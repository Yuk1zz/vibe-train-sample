# Objective — Llama 3.1 8B training (8×H100/H200, FSDP)

Maximize **training throughput (tokens/sec)** on a single node of 8×H100/H200 GPUs while preserving gradient correctness relative to the TorchTitan FSDP reference.

## Target model

Llama 3.1 8B — decoder-only Transformer:
- Layers: 32, Hidden size: 4096, Attention heads: 32, KV heads: 8
- Attention: grouped-query attention (GQA)
- MLP: SwiGLU feedforward block
- Normalization: RMSNorm
- Position embedding: RoPE (Llama3-style scaled)
- Context length: 4096 (training sequence length)

## Workload (fixed — do not change)

Stage 1 (initial reproduction):
- Parallelism: 1D FSDP over 8 GPUs
- Local batch size per GPU: 2 sequences
- Global batch size: 16 sequences
- Sequence length: 4096 tokens
- Tokens per optimizer update: 65,536

The reference and candidate systems must use the same sequence length, same tokenized batches, same data order, same seed, and same tokens/update.

## Correctness criterion

The candidate passes if its accumulated gradients (before `optimizer.step()`) match the reference gradients in FP32:

```python
torch.allclose(grad_candidate_fp32, grad_reference_fp32, rtol=1e-3, atol=1e-5)
```

All parameter gradients must pass this check.

## Reference targets (TorchTitan published, 8-GPU Llama 3.1 8B)

| Configuration | tok/sec |
|---|---|
| FSDP baseline | 6,258 |
| + torch.compile | 6,674 |
| + torch.compile + Float8 | 9,409 |

The first milestone is to reproduce the FSDP baseline, then beat it.

## Evaluation

1. Run 50–100 steps; confirm no loss divergence or NaNs.
2. Compare accumulated gradients before `optimizer.step()` at step 10 against the reference (see `accuracy_checker/`).
3. Measure throughput with the benchmark script (see `benchmark/`).
4. A candidate is accepted only if it passes gradient verification.
