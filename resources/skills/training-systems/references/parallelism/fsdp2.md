# FSDP2 — fully_shard (PyTorch 2.x DTensor-based sharding)

FSDP2 is the PyTorch 2.x successor to FSDP1. Uses `DTensor` under the hood, exposes a `fully_shard()` functional API instead of the `FSDP(model, ...)` wrapper, and is composable with `torch.compile` without graph breaks.

## Prerequisites

- `torch >= 2.4` (FSDP2 stabilized; `fully_shard` available in `torch.distributed.fsdp`)
- One CUDA device per rank, NCCL initialized
- `use_orig_params` is implicit (always on in FSDP2)

## Core API

```python
from torch.distributed.fsdp import fully_shard, MixedPrecisionPolicy

mp_policy = MixedPrecisionPolicy(
    param_dtype=torch.bfloat16,
    reduce_dtype=torch.float32,
)

# Apply per-TransformerBlock (innermost first, then the root)
for block in model.layers:
    fully_shard(block, mesh=device_mesh, mp_policy=mp_policy)
fully_shard(model, mesh=device_mesh, mp_policy=mp_policy)
```

**Order matters**: shard inner modules before outer ones. The reverse order of FSDP1's `auto_wrap_policy`.

## Device mesh

```python
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh

dist.init_process_group("nccl")
device_mesh = init_device_mesh("cuda", (dist.get_world_size(),))
```

For single-node 8-GPU (1D FSDP): `init_device_mesh("cuda", (8,))`.

## foreach optimizer (FSDP2-native)

FSDP2 supports `foreach`-style optimizer that operates on all sharded params at once, reducing Python overhead:

```python
from torch.distributed.fsdp import FSDPModule

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=3e-4, betas=(0.9, 0.95), weight_decay=0.1,
    foreach=True,   # batch parameter updates; works with FSDP2 sharding
    fused=False,    # fused and foreach are mutually exclusive
)
```

`foreach=True` is preferred over `fused=True` under FSDP2 for better overlap with async comm.

## torch.compile integration

FSDP2 is designed to work with `torch.compile` without graph breaks across module boundaries:

```python
model = torch.compile(model, fullgraph=True, mode="reduce-overhead")
```

Under FSDP1 + `torch.compile`, each FSDP boundary introduced a graph break. FSDP2 uses `DTensor` ops that Dynamo can trace through, enabling a single compiled graph across the full model.

**Known issue**: `fullgraph=True` may fail if your model has non-traceable Python control flow. Use `fullgraph=False` (default) first, then promote to `fullgraph=True` once the graph is stable.

## Async-allreduce overlap

FSDP2 overlaps gradient reduce-scatter with backward compute automatically. Additionally, it supports async all-gather prefetch during forward:

```python
fully_shard(block, mesh=device_mesh, mp_policy=mp_policy,
            reshard_after_forward=True)  # default; re-shards params after each module's forward
```

Setting `reshard_after_forward=False` on the last block can reduce latency at the cost of memory.

## Saving gradients (FSDP2)

FSDP2 uses `DTensor` gradients. To gather full gradients on rank 0:

```python
from torch.distributed.tensor import DTensor

grads = {}
for name, param in model.named_parameters():
    if param.grad is not None:
        g = param.grad
        if isinstance(g, DTensor):
            g = g.full_tensor()  # gathers shards to rank 0 (rank0_only via placements)
        if dist.get_rank() == 0:
            grads[name] = g.detach().float().cpu()
if dist.get_rank() == 0:
    torch.save(grads, "grads.pt")
```

## Migration from FSDP1

| FSDP1 | FSDP2 |
|---|---|
| `FSDP(model, auto_wrap_policy=..., mixed_precision=mp, ...)` | `fully_shard(block); fully_shard(model, mp_policy=mp)` |
| `MixedPrecision(param_dtype=..., reduce_dtype=...)` | `MixedPrecisionPolicy(param_dtype=..., reduce_dtype=...)` |
| `ShardingStrategy.FULL_SHARD` | default in `fully_shard` |
| `summon_full_params(model, with_grads=True)` | `param.grad.full_tensor()` for DTensor grads |
| `use_orig_params=True` | always on |
| `forward_prefetch=True` | implicit async all-gather |

## Pitfalls

1. **Order of `fully_shard` calls** — inner before outer. Reversing this gives incorrect sharding.
2. **`foreach=True` and `fused=True` are mutually exclusive** — pick one.
3. **`fullgraph=True` with dynamic shapes** — graph breaks on `if seq_len != ...`; use `torch.compiler.disable()` guards.
4. **DTensor grad vs Tensor grad** — check `isinstance(param.grad, DTensor)` before calling `.full_tensor()`.
