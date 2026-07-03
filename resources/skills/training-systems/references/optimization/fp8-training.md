# Float8 / FP8 Training

Train linear layers in FP8 (E4M3/E5M2) to reduce memory bandwidth and increase MMA throughput on H100/H200. TorchTitan reports +41% tokens/sec on Llama 3.1 8B vs BF16+compile alone (9,409 vs 6,674 tok/sec).

## Prerequisites

- H100 / H200 (sm_90+) — FP8 MMA requires Hopper tensor cores
- `torch >= 2.4`
- `torchao >= 0.5` or `transformer-engine`

## torchao Float8 (recommended)

`torchao` provides `Float8Linear` as a drop-in replacement for `nn.Linear` with automatic delayed scaling.

```python
from torchao.float8 import convert_to_float8_training, Float8LinearConfig

config = Float8LinearConfig(
    enable_fsdp_float8_all_gather=True,   # all-gather in FP8 (reduces comm volume)
    precompute_float8_dynamic_scale_for_fsdp=True,  # amortize scale computation
)

model = convert_to_float8_training(
    model,
    config=config,
    module_filter_fn=lambda mod, fqn: isinstance(mod, nn.Linear),
)
```

Apply **before** FSDP wrap and before `torch.compile`.

## Delayed scaling

`torchao` uses per-tensor delayed scaling: the scale factor from step N-1 is used for step N. This avoids the per-tensor all-reduce that dynamic scaling requires, at the cost of one-step lag in scale accuracy.

```python
# Manual delayed scaling (advanced)
from torchao.float8 import sync_float8_amax_and_scale_history

# In training loop, after backward, before optimizer.step:
sync_float8_amax_and_scale_history(model)
optimizer.step()
```

When using `convert_to_float8_training`, this is handled automatically by `Float8Linear`.

## FSDP + Float8 all-gather

With `enable_fsdp_float8_all_gather=True`, FSDP gathers parameters in FP8 instead of BF16, halving the all-gather communication volume:

```python
# Must be enabled before FSDP wrapping
config = Float8LinearConfig(enable_fsdp_float8_all_gather=True)
model = convert_to_float8_training(model, config=config)
model = FSDP(model, ...)  # or fully_shard
```

This is the primary reason Float8 improves throughput beyond just faster matmuls — communication is the bottleneck at large scale.

## Dtypes

| dtype | Exponent bits | Mantissa bits | Range | Used for |
|---|---|---|---|---|
| `torch.float8_e4m3fn` | 4 | 3 | ±448 | weights, activations (forward) |
| `torch.float8_e5m2` | 5 | 2 | ±57344 | gradients (backward) |

The asymmetry — E4M3 for forward, E5M2 for backward — matches the different dynamic ranges of activations vs gradients.

## transformer-engine alternative

`transformer-engine` (NVIDIA) provides `te.Linear` with FP8 support and built-in delayed scaling:

```python
import transformer_engine.pytorch as te

# Replace nn.Linear with te.Linear
class MyAttention(nn.Module):
    def __init__(self):
        self.q_proj = te.Linear(hidden, heads * head_dim)
        ...
```

`transformer-engine` requires matching CUDA version and is less composable with `torch.compile` than `torchao`.

## Correctness impact

FP8 training changes the gradient numerics. The vibe-train accuracy checker uses:
```
rtol=1e-3, atol=1e-5
```
Float8 training typically produces gradients within this tolerance vs BF16 reference. If verification fails, tighten the tolerance investigation to the attention projections (`q_proj`, `k_proj`, `v_proj`) which have the most extreme activation ranges.

## Pitfalls

1. **Applying Float8 after FSDP wrap** — `convert_to_float8_training` must see `nn.Linear` modules before FSDP replaces them with FSDP wrappers.
2. **Forgetting `sync_float8_amax_and_scale_history`** — scale history doesn't advance; stale scales degrade accuracy.
3. **FP8 on non-Hopper hardware** — silently falls back to BF16 or raises. Check `torch.cuda.get_device_capability() >= (9, 0)`.
4. **`enable_fsdp_float8_all_gather` without FSDP** — flag is ignored; no error but no benefit.
5. **Loss spikes at step 1** — delayed scaling uses a scale of 1.0 initially; may cause overflow on the first step. Add a warmup phase in BF16 before enabling FP8.
