# FlashAttention for Training

Use FlashAttention in the training forward+backward pass to reduce activation memory from O(T²) to O(T) and increase attention kernel throughput. Required for seq_len ≥ 2048 on H100.

## Why training needs FlashAttention

Without FlashAttention, the standard attention implementation stores the full N×N attention weight matrix for use in the backward pass:
- At seq_len=4096, BF16: one layer's attention matrix = `4096² × 2B × 32 heads = 1 GB`
- 32 layers × 1 GB = 32 GB just for attention weights, per sample

FlashAttention recomputes attention weights during backward (like activation checkpointing) without storing the N×N matrix, dropping attention activation memory to O(T).

## PyTorch SDPA (simplest)

PyTorch's `scaled_dot_product_attention` automatically dispatches to FlashAttention when available:

```python
import torch.nn.functional as F

# Inside forward:
out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
# → dispatches to FlashAttention on sm_80+ with float16/bfloat16
```

SDPA's Flash backend supports backward pass (gradients computed correctly). This is the recommended approach for new training systems — zero additional dependencies.

Verify dispatch:

```python
from torch.backends.cuda import sdp_kernel, SDPBackend
with sdp_kernel(enable_flash=True, enable_math=False, enable_mem_efficient=False):
    out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
# Raises if FlashAttention is not available
```

## Direct flash-attn (FA2)

```python
from flash_attn import flash_attn_func

# q, k, v: (batch, seqlen, nheads, head_dim) — note: seqlen before nheads
out = flash_attn_func(q, k, v, dropout_p=0.0, causal=True)
```

FA2's `flash_attn_func` supports training (forward + backward). The backward is fused and memory-efficient.

**Shape convention difference from SDPA**: FA2 uses `(B, T, H, D)`, SDPA uses `(B, H, T, D)`. Transpose before and after:

```python
q = self.q_proj(x).view(B, T, self.n_heads, self.head_dim)
k = self.k_proj(x).view(B, T, self.n_kv_heads, self.head_dim)
v = self.v_proj(x).view(B, T, self.n_kv_heads, self.head_dim)

out = flash_attn_func(q, k, v, causal=True)  # (B, T, H, D)
out = out.view(B, T, -1)
```

## GQA support

Both SDPA and FA2 support GQA (grouped-query attention) natively:

```python
# SDPA: expand K/V heads before calling
k = k.repeat_interleave(n_kv_groups, dim=1)  # (B, H_q, T, D)
v = v.repeat_interleave(n_kv_groups, dim=1)
out = F.scaled_dot_product_attention(q, k, v, is_causal=True)

# FA2: native GQA, no expansion needed
from flash_attn import flash_attn_func
out = flash_attn_func(q, k, v, causal=True)  # FA2 handles GQA internally when nheads_k < nheads_q
```

FA2's native GQA avoids materializing the expanded K/V tensors, saving memory.

## torch.compile + FlashAttention

SDPA with Flash backend: fully compatible with `torch.compile`. Dynamo traces through `F.scaled_dot_product_attention` and the Flash dispatch is inlined.

FA2 (`flash_attn_func`): registered as a custom op via `torch.library`; compile-compatible from FA2 ≥ 2.7 / PyTorch ≥ 2.4. Earlier versions cause graph breaks at the FA2 call site.

## FSDP + FlashAttention

No special handling needed. FSDP shards the attention projection weights; the kernel itself runs on the local shard of Q/K/V.

## Activation checkpointing + FlashAttention

FlashAttention already recomputes attention weights during backward. Do NOT apply `torch.utils.checkpoint` around the FA call itself — it doubles the recomputation without memory benefit.

Apply AC at the TransformerBlock level (checkpoint attention's projections + FA + output), not around FA alone.

## Pitfalls

1. **Wrong shape for FA2** — FA2 expects `(B, T, H, D)`, not `(B, H, T, D)`. Silently computes wrong output.
2. **`dropout_p > 0` in eval mode** — set `dropout_p=0.0` during eval.
3. **AC around FA alone** — redundant; FA already doesn't store the attention matrix.
4. **SDPA with `enable_math=True` fallback** — on architectures < sm_80, falls back to slow math kernel that stores the N×N matrix. Verify Flash backend is active.
5. **FA2 backward with non-contiguous tensors** — ensure Q, K, V are contiguous in memory before FA2 call.
