# Gradient Accumulation and Microbatch Tuning

Increase effective batch size without increasing per-GPU memory by accumulating gradients over multiple micro-steps before calling `optimizer.step()`.

## Effective batch size

```
effective_global_batch = local_batch_size × world_size × accumulation_steps
```

For the vibe-train target (fixed workload): 65,536 tokens/update = 16 seqs × 4096 tokens.
With accumulation_steps=N: local_batch_size = 16 / (world_size × N) = 2 / N per GPU.

**Do not change** the total tokens/update — only the micro-step split is an optimization.

## Basic pattern

```python
accumulation_steps = 4  # example: 4 micro-steps of local_bs=1 → effective local_bs=4
optimizer.zero_grad()

for micro_step in range(accumulation_steps):
    input_ids, labels = get_micro_batch(step, micro_step, rank)

    with model.no_sync() if micro_step < accumulation_steps - 1 else contextlib.nullcontext():
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = model(input_ids)
            loss = F.cross_entropy(logits.view(-1, vocab_size), labels.view(-1))
            loss = loss / accumulation_steps  # scale loss to match full-batch gradient

        loss.backward()

torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
optimizer.step()
optimizer.zero_grad()
```

## no_sync() with FSDP

`model.no_sync()` suppresses FSDP's gradient reduce-scatter on all but the final micro-step:

```python
# FSDP1
from contextlib import contextmanager
ctx = model.no_sync()  # returns a context manager

# FSDP2
# fully_shard modules also support .no_sync() via FSDPModule protocol
```

Without `no_sync()`, FSDP performs reduce-scatter after every micro-step's backward, wasting N-1 reductions that will be overwritten.

## Loss scaling

Scale the loss by `1 / accumulation_steps` before `backward()`. This ensures the accumulated gradient equals the gradient of the full-batch loss (not N× larger):

```python
loss = loss / accumulation_steps
loss.backward()  # gradient += (full_batch_loss_gradient / accumulation_steps)
# After accumulation_steps steps: gradient == full_batch_gradient
```

**Gradient clipping** must happen after accumulation, before the optimizer step.

## torch.compile + gradient accumulation

`torch.compile` traces the forward+backward of one micro-step. The `no_sync()` context switches FSDP from "reduce-scatter" to "accumulate" mode, which changes the graph. This causes recompilation on the first and last micro-steps.

To avoid recompilation overhead:
```python
# Option 1: compile inside no_sync (compile sees one graph for sync, one for no-sync)
# Option 2: use torch.compiler.disable() around the no_sync boundary
# Option 3: avoid micro-batching at the cost of higher per-step memory
```

TorchTitan avoids this by not using gradient accumulation in the Stage 1 benchmark (local_bs=2 fits in memory).

## Memory vs throughput tradeoff

More accumulation_steps → smaller micro-batches → lower GPU utilization (SM efficiency drops below ~8 seqs on H100 for Llama 3.1 8B). Microbatch size ≥ 2 per GPU is generally the minimum for good SM utilization.

| accumulation_steps | local_batch (8 GPUs) | per-step memory | SM utilization |
|---|---|---|---|
| 1 | 2 seqs | highest | highest |
| 2 | 1 seq | medium | medium |
| 4 | 0.5 seq | lowest | low |

For Stage 1 (local_bs=2, no accumulation) stay at `accumulation_steps=1`.

## Pitfalls

1. **Forgetting to scale loss** — accumulated gradient is N× too large; gradient clipping masks the error but optimizer steps are wrong.
2. **Gradient clipping before accumulation is complete** — clip_grad_norm_ reads the partial gradient, not the full accumulated gradient.
3. **no_sync() in FSDP2 with compile** — FSDP2's async-allreduce is inside the compiled graph; `no_sync()` may break the compiled graph boundary. Test explicitly.
4. **Not resetting `zero_grad()` after `optimizer.step()`** — old gradients bleed into the next accumulation cycle.
