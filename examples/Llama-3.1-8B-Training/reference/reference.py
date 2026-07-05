"""
TorchTitan reference trainer for Llama 3.1 8B.

Uses TorchTitan's Transformer model directly (torchtitan.models.llama3) rather than
reimplementing it. This guarantees the reference and the candidate are compared against
the real TorchTitan architecture, not a hand-rolled approximation.

Requirements:
    pip install torchtitan          # requires PyTorch >= 2.5

Run with:
    torchrun --nproc_per_node=8 reference.py [--steps N] [--save-grads PATH] [--grad-step K]

The script saves accumulated pre-optimizer gradients at step --grad-step to --save-grads
for use by the accuracy checker.

Interface (importable):
    from reference import TrainingConfig, make_batch
"""

from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    MixedPrecision,
    ShardingStrategy,
)
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
import functools

from torchtitan.models.llama3.model.model import Transformer, TransformerBlock
from torchtitan.models.llama3.model.args import TransformerModelArgs, RoPEScalingArgs


# ---------------------------------------------------------------------------
# Training hyper-parameters (separate from model architecture)
# ---------------------------------------------------------------------------

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
    seed: int = 42

    @classmethod
    def from_json(cls, path: str | Path) -> "TrainingConfig":
        data = json.loads(Path(path).read_text()).get("training", {})
        fields = cls.__dataclass_fields__
        return cls(**{k: data[k] for k in fields if k in data})


def _make_model_args(seq_len: int) -> TransformerModelArgs:
    """TorchTitan TransformerModelArgs for Llama 3.1 8B (matches torchtitan llama3_1_8b config)."""
    return TransformerModelArgs(
        dim=4096,
        n_layers=32,
        n_heads=32,
        n_kv_heads=8,
        vocab_size=128256,
        multiple_of=1024,
        ffn_dim_multiplier=1.3,
        norm_eps=1e-5,
        rope_theta=500000.0,
        rope_scaling_args=RoPEScalingArgs(
            scaling_factor=8.0,
            low_freq_factor=1.0,
            high_freq_factor=4.0,
            original_max_position_embeddings=8192,
        ),
        max_seq_len=seq_len,
        depth_init=True,
        attn_type="sdpa",
    )


# ---------------------------------------------------------------------------
# Synthetic data loader
# ---------------------------------------------------------------------------

def make_batch(
    local_bs: int, seq_len: int, vocab_size: int, seed: int, step: int, rank: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Deterministic synthetic batch — identical inputs on reference and candidate."""
    g = torch.Generator()
    g.manual_seed(seed + step * 1000 + rank)
    ids = torch.randint(0, vocab_size, (local_bs, seq_len + 1), generator=g)
    return ids[:, :-1], ids[:, 1:]


# ---------------------------------------------------------------------------
# FSDP wrapping
# ---------------------------------------------------------------------------

def wrap_fsdp(model: Transformer, device: torch.device) -> FSDP:
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
# Gradient capture
# ---------------------------------------------------------------------------

def _save_gradients(model: FSDP, path: str) -> None:
    grads: dict[str, torch.Tensor] = {}
    with FSDP.summon_full_params(model, with_grads=True, rank0_only=True):
        for name, param in model.named_parameters():
            if param.grad is not None:
                grads[name] = param.grad.detach().float().cpu()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(grads, path)
    print(f"[reference] saved {len(grads)} gradient tensors to {path}")


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
    train_cfg = TrainingConfig.from_json(cfg_path)
    model_args = _make_model_args(seq_len=train_cfg.seq_len)

    # Build TorchTitan model and call its own init_weights for canonical initialization
    with torch.device(device):
        model = Transformer(model_args)
    model.init_weights()
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
            model_args.vocab_size, train_cfg.seed, step, rank,
        )
        input_ids = input_ids.to(device)
        labels = labels.to(device)

        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = model(input_ids)
            loss = F.cross_entropy(logits.view(-1, model_args.vocab_size), labels.view(-1))

        loss.backward()

        # Save pre-optimizer gradients at the target step (rank 0 only, gathered via FSDP)
        if save_grads and step == grad_step and rank == 0:
            _save_gradients(model, save_grads)

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

    mean_step = sum(step_times[5:]) / max(len(step_times[5:]), 1)
    return {
        "tokens_per_sec": tokens_per_step / mean_step,
        "mean_step_ms": mean_step * 1000,
        "tokens_per_step": tokens_per_step,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="TorchTitan Llama 3.1 8B FSDP reference trainer")
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
