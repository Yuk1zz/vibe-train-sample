---
name: training-systems
description: >-
  LLM training system optimization. Activate when a task touches distributed training,
  FSDP/FSDP2, activation checkpointing, torch.compile for training, FP8/Float8 training,
  FlashAttention backward pass, communication-compute overlap, gradient accumulation,
  fused AdamW, training profiling (Nsight/PyTorch Profiler), or gradient correctness verification.
---

# training-systems

This skill bundles curated reference material for **LLM training system optimization**, focused on throughput optimization while preserving gradient correctness. The first target is Llama 3.1 8B on 8×H100 using TorchTitan as the reference.

## How to use this skill

1. Read this file once to learn what's covered.
2. Identify the one or two topics that match the active task.
3. Open `references/<tier>/<topic>.md` directly. Each is self-contained.

## Default-on optimizations (distributed LLM training on NVIDIA)

Three techniques every production training system on NVIDIA ships before exploring further:

1. **FSDP2** — see [`references/parallelism/fsdp2.md`](references/parallelism/fsdp2.md). The PyTorch 2.x successor to FSDP1 with `DTensor`-based sharding, `foreach` optimizer support, and cleaner async-allreduce overlap. Prefer FSDP2 over FSDP1 for new systems.
2. **FlashAttention (training)** — see [`references/optimization/flashattention-training.md`](references/optimization/flashattention-training.md). Required for any training system: eliminates the O(T²) activation memory footprint of naive attention.
3. **torch.compile** — see [`references/optimization/torch-compile-training.md`](references/optimization/torch-compile-training.md). Compile the training step to fuse elementwise ops, remove Python overhead, and unlock inductor kernel fusion. TorchTitan shows +6.6% throughput on Llama 3.1 8B vs eager FSDP alone.

## Reference index

### Parallelism

- [`references/parallelism/fsdp.md`](references/parallelism/fsdp.md) — FSDP1 (FullyShardedDataParallel): sharding strategies, mixed precision, `transformer_auto_wrap_policy`, grad capture, `summon_full_params`.

- [`references/parallelism/fsdp2.md`](references/parallelism/fsdp2.md) — FSDP2 / `torch.distributed.tensor.parallel`: DTensor sharding, `fully_shard`, `foreach` optimizer, async-allreduce overlap, migration from FSDP1.

### Optimization techniques

- [`references/optimization/activation-checkpointing.md`](references/optimization/activation-checkpointing.md) — Selective vs full activation checkpointing: which layers to checkpoint, `torch.utils.checkpoint`, TorchTitan's selective-AC policy (attention only), memory vs recompute tradeoff.

- [`references/optimization/torch-compile-training.md`](references/optimization/torch-compile-training.md) — `torch.compile` for training: `fullgraph=True`, `mode="reduce-overhead"`, integration with FSDP2, dynamic shapes, common graph-break sources in training loops.

- [`references/optimization/fp8-training.md`](references/optimization/fp8-training.md) — Float8 / FP8 training: `torchao` Float8Linear, delayed scaling, E4M3/E5M2, TorchTitan Float8 recipe, +41% throughput on H100 vs BF16+compile.

- [`references/optimization/comm-compute-overlap.md`](references/optimization/comm-compute-overlap.md) — Communication-compute overlap: FSDP prefetch, `limit_all_gathers`, backward-prefetch strategies, async gradient reduction, NCCL stream management.

- [`references/optimization/gradient-accumulation.md`](references/optimization/gradient-accumulation.md) — Gradient accumulation and microbatch tuning: `no_sync()` context, accumulation steps, effective batch size, interaction with FSDP and torch.compile.

- [`references/optimization/fused-adamw.md`](references/optimization/fused-adamw.md) — Fused AdamW: `torch.optim.AdamW(fused=True)`, `foreach=True` variant, memory and throughput impact, interaction with FSDP2's `foreach` optimizer support.

### Tooling

- [`references/tooling/gradient-verifier.md`](references/tooling/gradient-verifier.md) — Gradient correctness verification: `torch.allclose` on FP32 gradients before `optimizer.step()`, `FSDP.summon_full_params(with_grads=True)`, tolerance selection (rtol=1e-3, atol=1e-5), common sources of gradient divergence.

- [`references/tooling/training-benchmark.md`](references/tooling/training-benchmark.md) — Training throughput benchmarking: measuring tokens/sec, step time, peak GPU memory, warmup strategies, wall-clock vs GPU-timer measurement, `torch.cuda.synchronize()` placement.

### Models

- [`references/models/llama-training.md`](references/models/llama-training.md) — Llama 3.1 8B training specifics: architecture details, RoPE scaling (Llama3 style), GQA, weight initialization, FSDP wrap policy for TransformerBlock, known numerical sensitivities.

## Out of scope

Kernel implementation (writing CUDA / Triton / CUTLASS kernels). For that, use the separate `agent-gpu-skills` collection.

Inference / serving optimization: use the `serving-systems` skill.
