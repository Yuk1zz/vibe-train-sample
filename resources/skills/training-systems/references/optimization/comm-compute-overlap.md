# Communication-Compute Overlap

Hide NCCL all-gather and reduce-scatter latency behind GPU compute. On 8×H100 NVLink, the interconnect is fast (~900 GB/s NVLink 4.0) but not free — at seq_len=4096 and 8B params, each all-gather of one TransformerBlock moves ~2 GB per step.

## FSDP1 overlap levers

### Forward all-gather prefetch

FSDP pre-gathers the next layer's parameters while computing the current layer's forward:

```python
model = FSDP(
    model,
    forward_prefetch=True,  # default: False in FSDP1
    ...
)
```

Effective when per-layer compute time > per-layer all-gather time. At batch_size=16 and seq_len=4096, this is usually the case for Llama 3.1 8B layers.

### Backward prefetch strategy

```python
from torch.distributed.fsdp import BackwardPrefetch

model = FSDP(
    model,
    backward_prefetch=BackwardPrefetch.BACKWARD_PRE,  # default; best overlap
    # BackwardPrefetch.BACKWARD_POST: overlap with current layer (less overlap)
    # None: no prefetch (save memory, worse throughput)
    ...
)
```

`BACKWARD_PRE` prefetches the all-gather for layer N while computing the backward for layer N+1. This is the standard TorchTitan configuration.

### limit_all_gathers

```python
model = FSDP(model, limit_all_gathers=True, ...)
```

Limits the number of in-flight all-gathers to prevent memory spikes from prefetching too many layers' parameters simultaneously. Trade memory for potential overlap reduction.

## FSDP2 overlap

FSDP2 uses the `DTensor`-based async all-gather that's composable with `torch.compile`:

```python
# Overlap is automatic in FSDP2; tune via reshard_after_forward
fully_shard(block, mesh=mesh, mp_policy=mp_policy,
            reshard_after_forward=True)  # re-shard immediately after fwd (default)
# reshard_after_forward=False: keep full params in memory after fwd (less re-gather in bwd)
```

Setting `reshard_after_forward=False` on the last few blocks reduces backward all-gather overhead at the cost of peak memory.

## Async gradient reduction (no_sync)

When using gradient accumulation, disable gradient reduce-scatter for all but the last micro-step:

```python
for micro_step in range(accumulation_steps):
    ctx = model.no_sync() if micro_step < accumulation_steps - 1 else contextlib.nullcontext()
    with ctx:
        logits = model(input_ids[micro_step])
        loss = F.cross_entropy(...) / accumulation_steps
        loss.backward()
optimizer.step()
optimizer.zero_grad()
```

`model.no_sync()` skips the all-reduce / reduce-scatter until the final micro-step, saving `(accumulation_steps - 1) / accumulation_steps` of the communication volume.

## NCCL stream and CPU-side overlap

FSDP launches NCCL ops on a separate CUDA stream from compute. Avoid CPU-side blocking between steps:

```python
# Wrong: .item() syncs CPU and GPU, stalls the pipeline
if loss.item() > 10.0:  # forces sync
    ...

# Right: accumulate loss in a tensor, log periodically
losses.append(loss.detach())
if step % 100 == 0:
    avg_loss = torch.stack(losses).mean().item()  # one sync per 100 steps
    losses.clear()
```

## Overlap checklist for Llama 3.1 8B on 8×H100

- [ ] `forward_prefetch=True` (FSDP1) or automatic (FSDP2)
- [ ] `backward_prefetch=BACKWARD_PRE` (FSDP1)
- [ ] `limit_all_gathers=True` if seeing OOM from prefetch
- [ ] `no_sync()` during gradient accumulation micro-steps
- [ ] No `.item()` / CPU-sync inside the hot training loop
- [ ] `torch.cuda.synchronize()` only at step boundaries for timing, not inside the loop

## Profiling overlap

Use Nsight Systems to verify overlap is happening:

```bash
nsys profile -t cuda,nvtx --capture-range=cudaProfilerApi \
    torchrun --nproc_per_node=8 train.py --steps 5
```

Look for NCCL kernels (`ncclAllReduce`, `ncclReduceScatter`, `ncclAllGather`) running in parallel with `volta_*` / `ampere_*` / `hopper_*` compute kernels on the timeline. If they are sequential, prefetch is not working.
