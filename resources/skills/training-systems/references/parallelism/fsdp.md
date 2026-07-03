# FSDP1 — FullyShardedDataParallel

Shard model parameters, gradients, and optimizer states across all ranks to eliminate the per-rank memory footprint of a full model replica. Use for training models that exceed single-GPU memory.

## Prerequisites

- `torch.distributed` initialized with NCCL backend
- One CUDA device per rank (`torch.cuda.set_device(local_rank)`)
- `torch >= 2.0`

## Sharding strategies

| Strategy | Params | Grads | Optimizer | When to use |
|---|---|---|---|---|
| `FULL_SHARD` | sharded | sharded | sharded | default; maximum memory savings |
| `SHARD_GRAD_OP` | sharded during fwd+bwd only; full after | sharded | sharded | saves memory during compute only |
| `NO_SHARD` | full (DDP-like) | full | full | baseline / debugging |
| `HYBRID_SHARD` | sharded within node, replicated across nodes | sharded within node | sharded within node | multi-node with NVLink within node |

For single-node 8×H100 target: use `FULL_SHARD`.

## Mixed precision

```python
from torch.distributed.fsdp import MixedPrecision

mp = MixedPrecision(
    param_dtype=torch.bfloat16,   # weights stored and communicated in BF16
    reduce_dtype=torch.float32,   # gradient all-reduce in FP32 (more stable)
    buffer_dtype=torch.bfloat16,  # buffers (e.g. freqs_cis) in BF16
)
```

`reduce_dtype=torch.float32` is critical for training stability — gradient accumulation error compounds without it.

## Wrapping a transformer

Always wrap at the `TransformerBlock` level, not the full model. This minimizes the all-gather granularity and maximizes overlap.

```python
import functools
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy

auto_wrap = functools.partial(
    transformer_auto_wrap_policy,
    transformer_layer_cls={TransformerBlock},
)

model = FSDP(
    model,
    auto_wrap_policy=auto_wrap,
    mixed_precision=mp,
    sharding_strategy=ShardingStrategy.FULL_SHARD,
    device_id=torch.device(f"cuda:{local_rank}"),
    use_orig_params=True,  # required for torch.compile and named param access
)
```

`use_orig_params=True` is required when:
- Using `torch.compile` (FSDP+compile needs original param views)
- Accessing parameter names for gradient saving (`summon_full_params`)
- Using per-parameter optimizer settings

## Saving gradients before optimizer.step()

```python
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

# After loss.backward(), before optimizer.step():
grads = {}
with FSDP.summon_full_params(model, with_grads=True, rank0_only=True):
    for name, param in model.named_parameters():
        if param.grad is not None:
            grads[name] = param.grad.detach().float().cpu()
if rank == 0:
    torch.save(grads, "grads.pt")
```

`rank0_only=True` — only rank 0 gets the full unsharded tensors; other ranks get zeros. Always guard the `torch.save` with `if rank == 0`.

## Optimizer

```python
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=3e-4,
    betas=(0.9, 0.95),
    weight_decay=0.1,
    fused=True,  # fused CUDA kernel; requires CUDA device
)
```

FSDP shards the optimizer state automatically when wrapping. Do not call `optimizer.zero_grad()` before the first step — FSDP starts with no optimizer state.

## Gradient clipping

```python
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

Works correctly with FSDP because `model.parameters()` under FSDP returns the local shard of each parameter; `clip_grad_norm_` handles the distributed norm correctly.

## Activation checkpointing with FSDP

Apply activation checkpointing **before** wrapping with FSDP:

```python
from torch.utils.checkpoint import checkpoint

def make_ac_forward(fn):
    def ac_fwd(x, freqs):
        return checkpoint(fn, x, freqs, use_reentrant=False)
    return ac_fwd

for block in model.layers:
    block.attn.forward = make_ac_forward(block.attn.forward)

model = FSDP(model, ...)
```

Using `use_reentrant=False` is required in modern PyTorch for correctness under FSDP.

## Communication-compute overlap

FSDP1 overlaps:
- **Forward all-gather**: pre-gathers the next layer's params while the current layer is computing (controlled by `forward_prefetch=True`).
- **Backward reduce-scatter**: overlaps gradient reduce-scatter with backward compute.

```python
model = FSDP(
    model,
    ...,
    forward_prefetch=True,         # overlap all-gather with forward compute
    backward_prefetch=BackwardPrefetch.BACKWARD_PRE,  # default; pre-fetches next layer during backward
    limit_all_gathers=True,        # limits inflight all-gathers to control memory
)
```

## Common pitfalls

1. **Not using `use_orig_params=True`** when calling `torch.compile` or saving named gradients.
2. **Saving gradients after `optimizer.step()`** — gradients are zeroed by the optimizer; save before.
3. **Applying AC after FSDP wrap** — checkpoint hooks must be registered on the unwrapped module.
4. **Missing `torch.cuda.synchronize()`** before measuring step time — CUDA execution is asynchronous.
5. **`reduce_dtype` omitted** — defaults to `param_dtype`, causing gradient accumulation in BF16 and potential instability at large batch sizes.

## Migration to FSDP2

See [`fsdp2.md`](fsdp2.md). FSDP2 uses `fully_shard()` instead of the `FSDP(model, ...)` wrapper and is composable with `torch.compile`. For new systems on PyTorch ≥ 2.4, prefer FSDP2.
