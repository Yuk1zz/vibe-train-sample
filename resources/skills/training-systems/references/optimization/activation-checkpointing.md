# Activation Checkpointing

Trade recomputation for activation memory. Instead of storing all intermediate activations for the backward pass, recompute them on demand during backward.

## Full vs selective checkpointing

| Mode | Memory saved | Recompute overhead | When to use |
|---|---|---|---|
| **Full AC** — checkpoint every TransformerBlock | ~70% of activation memory | ~30–35% slower | memory-constrained (can't fit even 1 layer's activations) |
| **Selective AC** — checkpoint only attention | ~40–50% of activation memory | ~10% slower | TorchTitan default; best throughput/memory tradeoff |
| **No AC** | 0 | 0 | fits in memory; maximize throughput |

For Llama 3.1 8B on H100-80GB with BF16: selective AC on attention layers is the TorchTitan default.

## Selective AC — attention only (TorchTitan policy)

Attention activations are the most memory-intensive (scale as O(T²) with sequence length). MLP activations are cheaper to store. Checkpoint attention, let MLP activations stay.

```python
from torch.utils.checkpoint import checkpoint

def make_ac_forward(fn):
    def ac_fwd(x, freqs_cis):
        return checkpoint(fn, x, freqs_cis, use_reentrant=False)
    return ac_fwd

# Apply before FSDP wrap
for block in model.layers:
    block.attn.forward = make_ac_forward(block.attn.forward)
```

**`use_reentrant=False` is required** for correctness under FSDP and `torch.compile`. The reentrant variant can silently produce wrong gradients when combined with FSDP's backward hooks.

## Full AC — checkpoint every block

```python
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    checkpoint_wrapper,
    CheckpointImpl,
    apply_activation_checkpointing,
)

apply_activation_checkpointing(
    model,
    checkpoint_wrapper_fn=functools.partial(
        checkpoint_wrapper,
        checkpoint_impl=CheckpointImpl.NO_REENTRANT,
    ),
    check_fn=lambda m: isinstance(m, TransformerBlock),
)
```

Apply before FSDP wrap. `CheckpointImpl.NO_REENTRANT` is the `use_reentrant=False` equivalent in the `checkpoint_wrapper` API.

## torch.compile + AC

`use_reentrant=False` is also required for `torch.compile` compatibility — reentrant AC inserts `torch.autograd.graph.saved_tensors_hooks` which breaks graph capture.

```python
# Wrong: reentrant AC + compile → graph break
checkpoint(fn, x, use_reentrant=True)  # DO NOT use with compile

# Correct
checkpoint(fn, x, use_reentrant=False)
```

## Memory estimation (Llama 3.1 8B)

Approximate activation memory per token per layer at BF16:
- Attention QKV + O projections: ~12× `hidden_size` bytes ≈ 98 KB/token/layer
- MLP gate/up/down projections: ~28× `hidden_size` bytes ≈ 229 KB/token/layer

At seq_len=4096, local_bs=2: ~32 GB activations per layer without AC, ~13 GB with selective AC.

## Pitfalls

1. **Applying AC after FSDP wrap** — AC hooks are on the unwrapped module. Wrapping with FSDP first loses the hooks.
2. **`use_reentrant=True` with FSDP** — silently wrong gradients in some configurations. Always `use_reentrant=False`.
3. **Double-checkpointing** — checkpointing a block that already has checkpointed sub-modules doubles recomputation. Apply AC at one level only.
4. **AC with `torch.cuda.graphs`** — checkpoint's recomputation uses Python autograd hooks, which are not capturable. Disable AC if using CUDA graphs.
