# Training Throughput Benchmark

Measure tokens/sec, step time, and peak GPU memory for a distributed training system.

## Primary metric

```
tokens/sec = (local_batch_size × seq_len × world_size) / step_time_s
           = tokens_per_step / step_time_s
```

For the vibe-train Stage 1 target: `65,536 tokens / step_time_s`.

## Correct step time measurement

```python
# WRONG: GPU execution is async; time.perf_counter() returns before GPU finishes
t0 = time.perf_counter()
loss.backward()
elapsed = time.perf_counter() - t0  # measures CPU enqueue time, not GPU time

# CORRECT: synchronize before stopping the timer
t0 = time.perf_counter()
loss.backward()
optimizer.step()
torch.cuda.synchronize()
elapsed = time.perf_counter() - t0
```

Synchronize after `optimizer.step()` to include the full step: forward + backward + optimizer update.

## Warmup exclusion

The first N steps include:
- JIT compilation (`torch.compile` graph capture)
- NCCL warmup
- CUDA allocator warmup

Exclude the first 5–10 steps from throughput statistics:

```python
step_times = []
for step in range(1, total_steps + 1):
    t0 = time.perf_counter()
    # ... training step ...
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    if step > warmup_steps:
        step_times.append(elapsed)

mean_step = sum(step_times) / len(step_times)
tokens_per_sec = tokens_per_step / mean_step
```

## Peak GPU memory

```python
# Reset memory stats before the benchmark window
torch.cuda.reset_peak_memory_stats()

# ... run steps ...

peak_mem_gb = torch.cuda.max_memory_allocated() / 1e9
```

Report per-rank peak memory. Peak memory determines the maximum batch size / seq_len the system can support.

## Distributed timing

In a multi-rank run, all ranks should report similar step times (FSDP synchronizes at all-reduce boundaries). Report rank 0's timing:

```python
if dist.get_rank() == 0:
    print(f"step {step}: {tokens_per_step / elapsed:,.0f} tok/sec")
```

## Output JSON schema (vibe-train contract)

The benchmark expects the candidate to produce a JSON file at `--output-json`:

```json
{
  "tokens_per_sec": 7142.0,
  "mean_step_ms": 9176.5,
  "peak_gpu_memory_gb": 62.4,
  "final_loss": 8.31,
  "samples_per_sec": 1.74
}
```

Minimum required: `tokens_per_sec > 0`.

## VibeTrainModel interface (import mode)

```python
class VibeTrainModel:
    @classmethod
    def from_config(cls, config_path: str, device: str, dtype) -> "VibeTrainModel":
        ...

    def train(self, num_steps: int) -> dict:
        # Returns: {"tokens_per_sec": float, "mean_step_ms": float,
        #            "final_loss": float, "peak_gpu_memory_gb": float}
        ...
```

## Reference targets

| Config | tok/sec | Ratio |
|---|---|---|
| TorchTitan FSDP (BF16) | 6,258 | 1.00× |
| + torch.compile | 6,674 | 1.07× |
| + torch.compile + Float8 | 9,409 | 1.50× |

## Nsight Systems profiling

```bash
nsys profile \
  -t cuda,nvtx,nccl \
  --capture-range=cudaProfilerApi \
  --capture-range-end=repeat:5 \
  -o profile_output \
  torchrun --nproc_per_node=8 train.py --steps 10
```

Add NVTX annotations around the training step:
```python
torch.cuda.nvtx.range_push("step")
# ... step code ...
torch.cuda.nvtx.range_pop()
```

## PyTorch Profiler (Python-level)

```python
from torch.profiler import profile, ProfilerActivity, schedule

with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    schedule=schedule(wait=5, warmup=2, active=3),
    on_trace_ready=torch.profiler.tensorboard_trace_handler("./prof"),
    record_shapes=True,
    with_stack=True,
) as prof:
    for step in range(10):
        # ... step ...
        prof.step()
```

Look for: top CUDA kernel time, communication vs compute ratio, idle gaps between kernels.
