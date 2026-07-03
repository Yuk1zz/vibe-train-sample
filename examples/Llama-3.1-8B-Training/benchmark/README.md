# Benchmark — Training Throughput

Measures tokens/sec, step time, and peak GPU memory for a candidate training system.

## Primary metric

**tokens/sec** = `global_batch_size × seq_len / step_time`
= `16 × 4096 / step_time` = 65,536 tokens/step / step_time_s

## Usage

### Import mode (candidate implements VibeTrainModel)

```bash
python benchmark.py --import-mode --steps 50 --warmup 10 --output-json result.json
```

The candidate must export:
```python
class VibeTrainModel:
    @classmethod
    def from_config(cls, config_path, device, dtype) -> "VibeTrainModel": ...
    def train(self, num_steps: int) -> dict: ...
    # train() must return {"tokens_per_sec": float, "mean_step_ms": float,
    #                      "final_loss": float, "peak_gpu_memory_gb": float}
```

### CLI mode (candidate accepts --output-json)

```bash
python benchmark.py \
    --cmd "torchrun --nproc_per_node=8 train.py" \
    --steps 50 --warmup 10 --output-json result.json
```

The candidate must write to `--output-json` with at minimum `{"tokens_per_sec": float}`.

## Output JSON schema

```json
{
  "config": {"steps": 50, "warmup_steps": 10, "wall_clock_s": 120.3},
  "tokens_per_sec": 7142.0,
  "mean_step_ms": 9176.5,
  "peak_gpu_memory_gb": 62.4,
  "final_loss": 8.31
}
```

## Reference targets

| Config | tok/sec |
|---|---|
| TorchTitan FSDP | 6,258 |
| + torch.compile | 6,674 |
| + torch.compile + Float8 | 9,409 |
