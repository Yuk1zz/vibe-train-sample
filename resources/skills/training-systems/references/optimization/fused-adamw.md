# Fused AdamW

Replace the elementwise Python-loop AdamW with a fused CUDA kernel that applies the weight update in a single kernel launch, reducing Python overhead and kernel launch latency.

## Options

| Variant | API | Backend | Notes |
|---|---|---|---|
| `fused=True` | `torch.optim.AdamW(fused=True)` | Single fused CUDA kernel | Best for FSDP1; mutually exclusive with `foreach` |
| `foreach=True` | `torch.optim.AdamW(foreach=True)` | Batched element-wise ops | Best for FSDP2; avoids per-parameter kernel launches |
| default | `torch.optim.AdamW()` | Python loop | Baseline; slowest |

## fused=True (FSDP1)

```python
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=3e-4,
    betas=(0.9, 0.95),
    eps=1e-8,
    weight_decay=0.1,
    fused=True,
)
```

Requirements:
- CUDA device (not MPS, not CPU)
- `torch >= 2.0`
- Parameters must be contiguous float tensors

The fused kernel applies the Adam update, weight decay, and grad clamp in a single pass over memory, vs 5+ separate elementwise kernels in the default implementation.

## foreach=True (FSDP2 / composable)

```python
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=3e-4,
    betas=(0.9, 0.95),
    eps=1e-8,
    weight_decay=0.1,
    foreach=True,  # batch all param updates into fewer CUDA calls
    fused=False,   # mutually exclusive with foreach
)
```

`foreach` issues batched CUDA calls for all parameters at once, avoiding Python loop overhead per-parameter. Preferred under FSDP2 because FSDP2's `DTensor` parameters work correctly with `foreach` but may have edge cases with `fused`.

## Throughput impact

On Llama 3.1 8B, 8B parameters → ~16 GB of BF16 weight memory → optimizer step touches ~64 GB (params + first moment + second moment). Fused AdamW reduces optimizer step time by ~15–25% vs default.

## Interaction with FSDP

FSDP shards optimizer state automatically — each rank only stores optimizer state for its local param shard. This applies to both `fused` and `foreach` variants.

```
Per-rank memory with FSDP full_shard:
  - Params:           8B × 2B / 8 ranks = 2 GB BF16
  - Optimizer (m, v): 2 × 8B × 4B / 8 ranks = 8 GB FP32
  - Total: ~10 GB optimizer memory per rank
```

## Gradient clipping with fused optimizer

Clip before calling `optimizer.step()`:

```python
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
optimizer.step()
```

`fused=True` does not support `maximize=True` or custom clip logic inside the kernel. External `clip_grad_norm_` is the correct approach.

## Pitfalls

1. **`fused=True` and `foreach=True` together** — raises `ValueError`. Pick one.
2. **`fused=True` on CPU params** — silently falls back to default implementation or raises. Ensure all params are on CUDA.
3. **`foreach=True` with mixed-precision params** — `foreach` internally casts if param dtypes differ. Ensure consistency.
4. **Forgetting `zero_grad()` after `step()`** — stale gradients accumulate silently.
5. **`zero_grad(set_to_none=True)` vs `set_to_none=False`** — `set_to_none=True` (default since PyTorch 2.0) saves memory by deallocating grad tensors. May cause issues if other code checks `param.grad is not None` before FSDP gathers. Use `set_to_none=False` if you check gradient existence explicitly.
