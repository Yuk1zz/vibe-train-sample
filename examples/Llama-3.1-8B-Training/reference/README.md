# Reference — Llama 3.1 8B FSDP Trainer

TorchTitan-style FSDP baseline for Llama 3.1 8B pretraining on 8×H100/H200.

## What this implements

- Explicit Llama 3.1 8B architecture (attention, MLP, RMSNorm, RoPE — no HuggingFace model classes)
- 1D FSDP (FullyShardedDataParallel) over 8 GPUs, BF16 mixed precision
- Selective activation checkpointing on attention layers only
- Fused AdamW optimizer
- Llama3-style scaled RoPE
- Gradient capture hook (saves accumulated grads before `optimizer.step()` at a configurable step)

## Running

```bash
# Basic training (20 steps, no grad save)
torchrun --nproc_per_node=8 reference.py --steps 20

# Save gradients at step 10 for accuracy checking
torchrun --nproc_per_node=8 reference.py --steps 20 \
  --save-grads /tmp/reference_grads.pt --grad-step 10
```

## Reference throughput targets (TorchTitan published)

| Config | tok/sec |
|---|---|
| FSDP | 6,258 |
| + torch.compile | 6,674 |
| + torch.compile + Float8 | 9,409 |
