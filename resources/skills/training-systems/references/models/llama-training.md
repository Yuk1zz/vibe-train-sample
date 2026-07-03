# Llama 3.1 8B — Training Specifics

Key details for implementing and optimizing a Llama 3.1 8B training system.

## Architecture

| Param | Value |
|---|---|
| Layers | 32 |
| Hidden size | 4096 |
| Head dim | 128 |
| Attention heads | 32 |
| KV heads | 8 (GQA, 4 Q-heads per KV-head) |
| Intermediate size | 14336 |
| Vocab size | 128,256 |
| RMS norm eps | 1e-5 |
| Attention type | Grouped-query attention (GQA), causal |
| MLP | SwiGLU: `down_proj(silu(gate_proj(x)) * up_proj(x))` |
| Positional embedding | RoPE with Llama3-style long-context scaling |
| Context length | Up to 128K (training at 4096) |
| Tied embeddings | No (embed_tokens and lm_head are separate) |
| Attention bias | No |
| MLP bias | No |

## RoPE — Llama3 scaling

Standard RoPE uses `inv_freq = 1 / (theta ^ (2i / d))`. Llama 3.1 applies a frequency-dependent scaling that smoothly interpolates between the original and scaled frequencies:

```python
def build_llama3_rope(head_dim, max_seq, theta=500000.0,
                      factor=8.0, low_freq=1.0, high_freq=4.0, orig_max=8192):
    inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2).float() / head_dim))
    low_wavelen = orig_max / low_freq
    high_wavelen = orig_max / high_freq
    wavelen = 2 * math.pi / inv_freq

    # Scale low-frequency components (long-range)
    inv_freq = torch.where(wavelen > low_wavelen, inv_freq / factor, inv_freq)

    # Smooth interpolation in between
    smooth = (orig_max / wavelen - low_freq) / (high_freq - low_freq)
    inv_freq = torch.where(
        (wavelen >= high_wavelen) & (wavelen <= low_wavelen),
        (1 - smooth) * inv_freq / factor + smooth * inv_freq,
        inv_freq,
    )

    t = torch.arange(max_seq).float()
    freqs = torch.outer(t, inv_freq)
    return torch.cat([freqs, freqs], dim=-1)  # (max_seq, head_dim)
```

**Training at seq_len=4096**: The scaling factor barely activates at 4096 (original context is 8192), so for Stage 1 benchmarking, standard RoPE with `theta=500000` and no scaling is functionally equivalent. Use the full Llama3 scaling for correctness parity with TorchTitan.

## FSDP wrap policy

Wrap at the `TransformerBlock` level. Each block contains Attention + MLP + 2× RMSNorm:

```python
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy

auto_wrap = functools.partial(
    transformer_auto_wrap_policy,
    transformer_layer_cls={TransformerBlock},
)
```

Do NOT wrap at the `Attention` or `MLP` level — too fine-grained, excess all-gather overhead.

## Weight initialization

Llama 3 uses:
- Embedding: `std = 0.02`
- Linear layers (q, k, v, o, gate, up, down): `std = 0.02 / sqrt(2 * num_layers)` for the output projections (o_proj, down_proj)
- RMSNorm weight: ones

For training correctness verification, load the actual pretrained weights to match the reference gradient numerics.

## Memory footprint (8×H100, FSDP full_shard, BF16)

| Component | Total size | Per rank (÷8) |
|---|---|---|
| Parameters (BF16) | ~16 GB | ~2 GB |
| Optimizer state (FP32 m+v) | ~64 GB | ~8 GB |
| Activations (seq_len=4096, bs=2, selective AC) | ~20 GB | ~20 GB |
| Gradients (BF16 sharded) | ~16 GB | ~2 GB |
| **Total** | | **~32 GB** |

H100 80GB: comfortable with selective AC. Without AC: ~50+ GB, still fits but leaves less margin.

## Numerical sensitivities

1. **SwiGLU intermediate**: `silu(gate) * up` can produce large values at init. No issue in BF16+FP32 reduce but can cause FP8 scale issues.
2. **RMSNorm with BF16**: RMSNorm must upcast to FP32 internally for the variance computation (`x.float().pow(2).mean()`). BF16 variance is numerically unstable.
3. **GQA KV expansion**: `repeat_interleave` is correct; `expand + reshape` may produce non-contiguous tensors that break FlashAttention or cause extra copies.
4. **lm_head scale**: The lm_head weight norm grows during training. At step 0 with random weights, loss ≈ `log(128256) ≈ 11.76`. If loss is much lower, the weights were initialized incorrectly.
