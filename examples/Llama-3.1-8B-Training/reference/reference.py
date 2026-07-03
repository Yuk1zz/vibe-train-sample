"""
TorchTitan-style FSDP reference trainer for Llama 3.1 8B.

Implements the baseline configuration from the TorchTitan Llama 3.1 8B benchmark:
  - 1D FSDP over 8 GPUs
  - BF16 mixed precision
  - Selective activation checkpointing (attention layers only)
  - AdamW optimizer
  - Fixed tokenized batches from FineWeb (synthetic fallback for testing)

Run with:
    torchrun --nproc_per_node=8 reference.py [--steps N] [--save-grads PATH] [--grad-step K]

The script saves accumulated pre-optimizer gradients at step --grad-step to --save-grads
for use by the accuracy checker.

Interface (importable):
    from reference import build_model, TrainingConfig
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    MixedPrecision,
    ShardingStrategy,
)
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from torch.utils.checkpoint import checkpoint as grad_checkpoint
import functools


# ---------------------------------------------------------------------------
# Model config
# ---------------------------------------------------------------------------

@dataclass
class ModelConfig:
    vocab_size: int = 128256
    hidden_size: int = 4096
    intermediate_size: int = 14336
    num_hidden_layers: int = 32
    num_attention_heads: int = 32
    num_key_value_heads: int = 8
    rms_norm_eps: float = 1e-5
    rope_theta: float = 500000.0
    max_position_embeddings: int = 131072

    @classmethod
    def from_json(cls, path: str | Path) -> "ModelConfig":
        data = json.loads(Path(path).read_text())
        fields = cls.__dataclass_fields__
        return cls(**{k: data[k] for k in fields if k in data})

    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.num_attention_heads

    @property
    def num_kv_groups(self) -> int:
        return self.num_attention_heads // self.num_key_value_heads


@dataclass
class TrainingConfig:
    seq_len: int = 4096
    local_batch_size: int = 2
    learning_rate: float = 3e-4
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    eps: float = 1e-8
    max_grad_norm: float = 1.0
    warmup_steps: int = 200
    seed: int = 42
    selective_ac: bool = True  # activation checkpoint attention layers only

    @classmethod
    def from_json(cls, path: str | Path) -> "TrainingConfig":
        data = json.loads(Path(path).read_text()).get("training", {})
        fields = cls.__dataclass_fields__
        return cls(**{k: data[k] for k in fields if k in data})


# ---------------------------------------------------------------------------
# RoPE
# ---------------------------------------------------------------------------

def _build_llama3_rope_freqs(head_dim: int, max_seq: int, theta: float,
                              factor: float = 8.0, low_freq_factor: float = 1.0,
                              high_freq_factor: float = 4.0,
                              orig_max_seq: int = 8192) -> torch.Tensor:
    inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
    low_freq_wavelen = orig_max_seq / low_freq_factor
    high_freq_wavelen = orig_max_seq / high_freq_factor
    wavelen = 2 * math.pi / inv_freq
    inv_freq = torch.where(wavelen > low_freq_wavelen, inv_freq / factor, inv_freq)
    smooth = (orig_max_seq / wavelen - low_freq_factor) / (high_freq_factor - low_freq_factor)
    inv_freq = torch.where(
        (wavelen >= high_freq_wavelen) & (wavelen <= low_freq_wavelen),
        (1 - smooth) * inv_freq / factor + smooth * inv_freq,
        inv_freq,
    )
    t = torch.arange(max_seq, dtype=torch.float32)
    freqs = torch.outer(t, inv_freq)
    return torch.cat([freqs, freqs], dim=-1)  # (max_seq, head_dim)


def apply_rotary(x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
    cos = freqs_cis.cos().to(x.dtype)
    sin = freqs_cis.sin().to(x.dtype)
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    x_rot = torch.cat((-x2, x1), dim=-1)
    return x * cos + x_rot * sin


# ---------------------------------------------------------------------------
# Model layers
# ---------------------------------------------------------------------------

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = x.float().pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return (x.float() * norm).to(x.dtype) * self.weight


class Attention(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.n_heads = cfg.num_attention_heads
        self.n_kv_heads = cfg.num_key_value_heads
        self.n_kv_groups = cfg.num_kv_groups
        self.head_dim = cfg.head_dim
        self.scale = self.head_dim ** -0.5

        self.q_proj = nn.Linear(cfg.hidden_size, cfg.num_attention_heads * cfg.head_dim, bias=False)
        self.k_proj = nn.Linear(cfg.hidden_size, cfg.num_key_value_heads * cfg.head_dim, bias=False)
        self.v_proj = nn.Linear(cfg.hidden_size, cfg.num_key_value_heads * cfg.head_dim, bias=False)
        self.o_proj = nn.Linear(cfg.num_attention_heads * cfg.head_dim, cfg.hidden_size, bias=False)

    def forward(self, x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
        B, T, _ = x.shape
        q = self.q_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)

        q = apply_rotary(q, freqs_cis)
        k = apply_rotary(k, freqs_cis)

        # Expand KV heads for GQA
        k = k.repeat_interleave(self.n_kv_groups, dim=1)
        v = v.repeat_interleave(self.n_kv_groups, dim=1)

        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        out = out.transpose(1, 2).contiguous().view(B, T, -1)
        return self.o_proj(out)


class MLP(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.gate_proj = nn.Linear(cfg.hidden_size, cfg.intermediate_size, bias=False)
        self.up_proj = nn.Linear(cfg.hidden_size, cfg.intermediate_size, bias=False)
        self.down_proj = nn.Linear(cfg.intermediate_size, cfg.hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class TransformerBlock(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.attn = Attention(cfg)
        self.mlp = MLP(cfg)
        self.input_layernorm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)

    def forward(self, x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.input_layernorm(x), freqs_cis)
        x = x + self.mlp(self.post_attention_layernorm(x))
        return x


class LlamaModel(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.embed_tokens = nn.Embedding(cfg.vocab_size, cfg.hidden_size)
        self.layers = nn.ModuleList([TransformerBlock(cfg) for _ in range(cfg.num_hidden_layers)])
        self.norm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.lm_head = nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=False)

        # Tie embed + lm_head weights (Llama 3.1 does NOT tie, but we keep separate)
        self.register_buffer(
            "freqs_cis",
            _build_llama3_rope_freqs(cfg.head_dim, cfg.max_position_embeddings, cfg.rope_theta),
            persistent=False,
        )

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        B, T = input_ids.shape
        x = self.embed_tokens(input_ids)
        freqs = self.freqs_cis[:T]
        for layer in self.layers:
            x = layer(x, freqs)
        x = self.norm(x)
        return self.lm_head(x)


def build_model(cfg: ModelConfig) -> LlamaModel:
    return LlamaModel(cfg)


# ---------------------------------------------------------------------------
# Selective activation checkpointing (attention layers only)
# ---------------------------------------------------------------------------

def _apply_selective_ac(model: LlamaModel) -> None:
    for block in model.layers:
        orig_forward = block.attn.forward

        def make_ac_forward(fn):
            def ac_forward(x, freqs_cis):
                return grad_checkpoint(fn, x, freqs_cis, use_reentrant=False)
            return ac_forward

        block.attn.forward = make_ac_forward(orig_forward)


# ---------------------------------------------------------------------------
# FSDP wrapping
# ---------------------------------------------------------------------------

def wrap_fsdp(model: LlamaModel, device: torch.device) -> FSDP:
    mp = MixedPrecision(
        param_dtype=torch.bfloat16,
        reduce_dtype=torch.float32,
        buffer_dtype=torch.bfloat16,
    )
    auto_wrap = functools.partial(
        transformer_auto_wrap_policy,
        transformer_layer_cls={TransformerBlock},
    )
    return FSDP(
        model,
        auto_wrap_policy=auto_wrap,
        mixed_precision=mp,
        sharding_strategy=ShardingStrategy.FULL_SHARD,
        device_id=device,
        use_orig_params=True,
    )


# ---------------------------------------------------------------------------
# Synthetic data loader (FineWeb tokenized placeholder)
# ---------------------------------------------------------------------------

def make_batch(local_bs: int, seq_len: int, vocab_size: int,
               seed: int, step: int, rank: int) -> tuple[torch.Tensor, torch.Tensor]:
    g = torch.Generator()
    g.manual_seed(seed + step * 1000 + rank)
    ids = torch.randint(0, vocab_size, (local_bs, seq_len + 1), generator=g)
    return ids[:, :-1], ids[:, 1:]


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train(
    steps: int,
    save_grads: Optional[str],
    grad_step: int,
    config_path: Optional[str],
) -> dict:
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)
    torch.manual_seed(42 + rank)

    cfg_path = config_path or (Path(__file__).parent / "config.json")
    model_cfg = ModelConfig.from_json(cfg_path)
    train_cfg = TrainingConfig.from_json(cfg_path)

    # Build model
    model = build_model(model_cfg).to(device)
    if train_cfg.selective_ac:
        _apply_selective_ac(model)
    model = wrap_fsdp(model, device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train_cfg.learning_rate,
        betas=(train_cfg.beta1, train_cfg.beta2),
        eps=train_cfg.eps,
        weight_decay=train_cfg.weight_decay,
        fused=True,
    )

    tokens_per_step = train_cfg.local_batch_size * train_cfg.seq_len * world_size
    step_times: list[float] = []

    for step in range(1, steps + 1):
        t0 = time.perf_counter()
        input_ids, labels = make_batch(
            train_cfg.local_batch_size, train_cfg.seq_len,
            model_cfg.vocab_size, train_cfg.seed, step, rank,
        )
        input_ids = input_ids.to(device)
        labels = labels.to(device)

        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = model(input_ids)
            loss = F.cross_entropy(logits.view(-1, model_cfg.vocab_size), labels.view(-1))

        loss.backward()

        # Save gradients BEFORE optimizer step at the target step
        if save_grads and step == grad_step and rank == 0:
            _save_gradients(model, save_grads, model_cfg)

        torch.nn.utils.clip_grad_norm_(model.parameters(), train_cfg.max_grad_norm)
        optimizer.step()
        optimizer.zero_grad()

        torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0
        step_times.append(elapsed)

        if rank == 0 and step % 10 == 0:
            tps = tokens_per_step / elapsed
            print(f"step {step:4d}  loss={loss.item():.4f}  tok/s={tps:,.0f}  step_ms={elapsed*1000:.1f}")

    dist.destroy_process_group()

    mean_step = sum(step_times[5:]) / max(len(step_times[5:]), 1)  # skip warmup
    return {
        "tokens_per_sec": tokens_per_step / mean_step,
        "mean_step_ms": mean_step * 1000,
        "tokens_per_step": tokens_per_step,
    }


def _save_gradients(model: FSDP, path: str, cfg: ModelConfig) -> None:
    grads: dict[str, torch.Tensor] = {}
    with FSDP.summon_full_params(model, with_grads=True, rank0_only=True):
        for name, param in model.named_parameters():
            if param.grad is not None:
                grads[name] = param.grad.detach().float().cpu()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(grads, path)
    print(f"[reference] saved {len(grads)} gradient tensors to {path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="TorchTitan-style FSDP reference trainer")
    parser.add_argument("--steps", type=int, default=20, help="Number of training steps")
    parser.add_argument("--save-grads", type=str, default=None,
                        help="Path to save gradient tensors (.pt) for accuracy checking")
    parser.add_argument("--grad-step", type=int, default=10,
                        help="Step at which to capture gradients (before optimizer.step)")
    parser.add_argument("--config", type=str, default=None, help="Path to config.json")
    args = parser.parse_args()

    results = train(args.steps, args.save_grads, args.grad_step, args.config)
    if int(os.environ.get("RANK", "0")) == 0:
        print("\n--- Training complete ---")
        for k, v in results.items():
            print(f"  {k}: {v:,.1f}")


if __name__ == "__main__":
    main()
