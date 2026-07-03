# Gradient Correctness Verifier

Verify that a candidate training system produces the same accumulated gradients as the reference before `optimizer.step()`. This is the primary correctness gate for vibe-train.

## Correctness criterion

```python
torch.allclose(grad_candidate_fp32, grad_reference_fp32, rtol=1e-3, atol=1e-5)
```

All named parameter gradients must pass. Both reference and candidate must use:
- Same initial checkpoint (weights)
- Same tokenized batches (same seed, same step)
- Same data order

## Capturing gradients with FSDP1

```python
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

# After loss.backward(), BEFORE optimizer.step():
grads = {}
with FSDP.summon_full_params(model, with_grads=True, rank0_only=True):
    for name, param in model.named_parameters():
        if param.grad is not None:
            grads[name] = param.grad.detach().float().cpu()

if dist.get_rank() == 0:
    torch.save(grads, path)
```

`rank0_only=True`: only rank 0 gets full unsharded tensors; other ranks receive zeros. Guard `torch.save` with `if rank == 0`.

`with_grads=True`: required to access the unsharded gradient alongside the parameter.

## Capturing gradients with FSDP2

```python
from torch.distributed.tensor import DTensor

grads = {}
for name, param in model.named_parameters():
    g = param.grad
    if g is None:
        continue
    if isinstance(g, DTensor):
        g = g.full_tensor()  # all-gather gradient shards to rank 0
    if dist.get_rank() == 0:
        grads[name] = g.detach().float().cpu()

if dist.get_rank() == 0:
    torch.save(grads, path)
```

## Comparing two gradient dicts

```python
def check_gradients(ref: dict, cand: dict, rtol=1e-3, atol=1e-5) -> bool:
    failures = []
    for name in ref:
        if name not in cand:
            failures.append(f"missing: {name}")
            continue
        ok = torch.allclose(cand[name].float(), ref[name].float(), rtol=rtol, atol=atol)
        if not ok:
            diff = (cand[name] - ref[name]).abs()
            failures.append(f"FAIL {name}: max_abs={diff.max():.2e}")
    return len(failures) == 0, failures
```

## Tolerance selection

`rtol=1e-3, atol=1e-5` is the vibe-train starting tolerance (from proposal).

| Optimization | Expected gradient change | Still passes? |
|---|---|---|
| BF16 baseline → BF16 baseline (same seed) | 0 | yes |
| BF16 → FP8 (torchao Float8) | < 1e-3 relative | usually yes |
| Different attention kernel (FA2 vs SDPA) | < 1e-6 | yes |
| Different reduce_dtype (BF16 vs FP32) | can exceed 1e-3 | may fail; use FP32 reduce |
| Different gradient accumulation order | 0 (same total) | yes |

If verification fails on FP8:
1. Tighten investigation to projections with large activation ranges (q_proj, k_proj).
2. Check that `sync_float8_amax_and_scale_history` is called at the right point.
3. Try `rtol=5e-3` as a diagnostic loosened tolerance to understand the magnitude.

## Common sources of gradient divergence

| Root cause | Symptom | Fix |
|---|---|---|
| `reduce_dtype=BF16` instead of FP32 | All grads slightly off; error grows with layer depth | Set `reduce_dtype=torch.float32` in MixedPrecision |
| Different data batch | Completely different grads | Fix seed and data-order logic |
| Missing `no_sync()` in accumulation | Grads reduced mid-accumulation | Add `model.no_sync()` context |
| `use_reentrant=True` in AC + FSDP | Specific layer grads wrong | Switch to `use_reentrant=False` |
| Float8 scale not synced | Random-looking divergence | Call `sync_float8_amax_and_scale_history` |
| Non-deterministic NCCL ops | Small stochastic error | Set `NCCL_DETERMINISTIC=1` (slow) for debugging |

## Run-to-run reproducibility

For exact gradient matching between reference and candidate:
```bash
CUBLAS_WORKSPACE_CONFIG=:4096:8 PYTHONHASHSEED=42 \
torchrun --nproc_per_node=8 train.py --seed 42 ...
```

`CUBLAS_WORKSPACE_CONFIG` enables deterministic cuBLAS. Set `torch.use_deterministic_algorithms(True)` for full determinism (some ops have no deterministic kernel and will raise).
