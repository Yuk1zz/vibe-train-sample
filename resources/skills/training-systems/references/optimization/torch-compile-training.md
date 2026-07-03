# torch.compile for Training

Compile the training forward+backward graph to fuse elementwise ops, eliminate Python overhead, and unlock Inductor kernel generation. TorchTitan reports +6.6% tokens/sec on Llama 3.1 8B vs eager FSDP.

## Basic usage

```python
model = torch.compile(model, fullgraph=False, mode="reduce-overhead")
```

Compile is applied to the model before the training loop. The first few steps are slow (compilation); throughput stabilizes after ~5–10 steps.

## Mode selection

| Mode | What it does | When to use |
|---|---|---|
| `"default"` | standard Inductor compilation | correctness baseline |
| `"reduce-overhead"` | enables CUDA graph capture for the compiled graph | training with fixed shapes (seq_len, batch) |
| `"max-autotune"` | exhaustive kernel search + `reduce-overhead` | maximum throughput; longer compile time |
| `"max-autotune-no-cudagraphs"` | `max-autotune` without CUDA graphs | when CUDA graphs are incompatible (AC, DDP) |

For training: start with `"reduce-overhead"`. Use `"max-autotune"` once the system is stable.

## FSDP1 + compile

With FSDP1, `torch.compile` sees graph breaks at FSDP module boundaries unless `use_orig_params=True`:

```python
model = FSDP(model, use_orig_params=True, ...)  # required
model = torch.compile(model)
```

Each FSDP-wrapped submodule boundary is still a separate compiled graph. This gives partial speedup but not `fullgraph=True` behavior.

## FSDP2 + compile (preferred)

FSDP2 uses `DTensor` operations that Dynamo traces through, enabling a single compiled graph across all layers:

```python
for block in model.layers:
    fully_shard(block, mesh=mesh, mp_policy=mp_policy)
fully_shard(model, mesh=mesh, mp_policy=mp_policy)

model = torch.compile(model, fullgraph=True, mode="reduce-overhead")
```

## fullgraph=True

Forces a single graph with no Python fallback. Fails on non-traceable control flow. Check for breaks first:

```python
import torch._dynamo
torch._dynamo.explain(model)(input_ids)  # lists graph breaks and reasons
```

Common break sources in transformers:
- `if self.training:` branches — use `model.train()` and avoid runtime mode checks
- `print()` / `logging` inside forward
- `.item()` calls (sync from GPU to CPU)
- Non-tensor Python containers iterated with dynamic length
- Custom autograd functions with Python-side backward

## Dynamic shapes

Fixed seq_len and batch size → no dynamic shapes needed → fastest compilation.

If seq_len varies:
```python
model = torch.compile(model, dynamic=True)  # trace once; generates guards
```

`dynamic=True` re-uses the compiled graph for different shapes if guards pass; recompiles only on guard failure.

## Selective compilation

Compile only the forward pass (skip AC recompute):

```python
# Wrap the AC checkpoint call to prevent compile from capturing recompute
@torch.compiler.disable
def ac_fwd(fn, x, freqs):
    return checkpoint(fn, x, freqs, use_reentrant=False)
```

Or compile at the `TransformerBlock` level:

```python
for block in model.layers:
    block.forward = torch.compile(block.forward, mode="reduce-overhead")
```

## Interaction with Float8

`torchao` Float8Linear is `torch.compile`-compatible. Apply Float8 conversion before compile:

```python
from torchao.float8 import convert_to_float8_training
model = convert_to_float8_training(model, ...)
model = torch.compile(model, ...)
```

## Pitfalls

1. **Compiling before FSDP wrap** — FSDP modifies parameter storage; compile the wrapped model.
2. **`use_reentrant=True` AC with compile** — graph break + wrong gradients. Use `use_reentrant=False`.
3. **First-step slowdown** — compile happens at first forward. Exclude steps 0–4 from throughput measurement.
4. **`reduce-overhead` + AC** — CUDA graph capture is incompatible with AC's Python autograd hooks. Use `"default"` mode when AC is active, or `"max-autotune-no-cudagraphs"`.
5. **Debug with `TORCH_COMPILE_DEBUG=1`** to see which ops are fused vs not.
