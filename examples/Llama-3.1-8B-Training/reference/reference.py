"""
TorchTitan reference trainer for Llama 3.1 8B.

Uses TorchTitan's full training infrastructure directly — same model, same
parallelization stack (FSDP2, torch.compile, Float8) as the TorchTitan paper.

Three benchmark tiers:
  --mode baseline  : FSDP2 + BF16 + selective activation checkpointing
  --mode compile   : baseline + torch.compile (per-TransformerBlock)
  --mode fp8       : compile  + Float8 linear layers (via torchao)

Run with:
    torchrun --nproc_per_node=8 reference.py --mode baseline --steps 35 --warmup 5
    torchrun --nproc_per_node=8 reference.py --mode compile  --steps 50 --warmup 20
    torchrun --nproc_per_node=8 reference.py --mode fp8      --steps 50 --warmup 20

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

# TorchTitan model
from torchtitan.models.llama3.model.model import Transformer
from torchtitan.models.llama3.model.args import TransformerModelArgs, RoPEScalingArgs

# TorchTitan parallelization infrastructure
from torchtitan.models.llama3.infra.parallelize import apply_fsdp, apply_compile
from torchtitan.distributed import ParallelDims
from torchtitan.distributed.activation_checkpoint import apply_ac
from torchtitan.config.job_config import (
    ActivationCheckpoint as ACConfig,
    Compile as CompileConfig,
)


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
    """TorchTitan TransformerModelArgs for Llama 3.1 8B."""
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
# Gradient capture (FSDP2 / DTensor-aware)
# ---------------------------------------------------------------------------

def _save_gradients(model: torch.nn.Module, path: str, rank: int) -> None:
    """Gather sharded FSDP2 gradients to rank 0 and save."""
    from torch.distributed.tensor import Replicate

    grads: dict[str, torch.Tensor] = {}
    for name, param in model.named_parameters():
        if param.grad is None:
            continue
        grad = param.grad
        if hasattr(grad, "redistribute"):
            # DTensor: all-gather shards onto every rank, then take local copy
            full_grad = grad.redistribute(placements=[Replicate()]).to_local()
        else:
            full_grad = grad
        if rank == 0:
            grads[name] = full_grad.detach().float().cpu()

    if rank == 0:
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
    warmup: int = 5,
    mode: str = "baseline",
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

    # Build model directly on device with canonical TorchTitan initialization
    with torch.device(device):
        model = Transformer(model_args)
    model.init_weights()

    # ── Float8 (must happen before FSDP2) ─────────────────────────────────
    if mode == "fp8":
        from torchao.float8 import convert_to_float8_training, Float8LinearConfig
        convert_to_float8_training(model, config=Float8LinearConfig())
        if rank == 0:
            print("[reference] applied Float8 linear conversion")

    # ── FSDP2 device mesh (pure data-parallel, no TP/PP) ──────────────────
    parallel_dims = ParallelDims(
        dp_replicate=1,
        dp_shard=world_size,
        cp=1, tp=1, pp=1, ep=1, etp=1,
        world_size=world_size,
    )
    parallel_dims.build_mesh()
    dp_mesh = parallel_dims.get_mesh("fsdp")

    # ── Selective activation checkpointing (every 2nd layer — TorchTitan default) ──
    model_compile_enabled = mode in ("compile", "fp8")
    ac_config = ACConfig(mode="selective", selective_ac_option="2")
    apply_ac(model, ac_config, model_compile_enabled=model_compile_enabled)

    # ── torch.compile per-TransformerBlock (before FSDP2) ─────────────────
    if model_compile_enabled:
        compile_config = CompileConfig(enable=True, components=["model"], backend="inductor")
        apply_compile(model, compile_config)

    # ── FSDP2 ─────────────────────────────────────────────────────────────
    apply_fsdp(
        model,
        dp_mesh,
        param_dtype=torch.bfloat16,
        reduce_dtype=torch.float32,
        pp_enabled=False,
    )

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

        if save_grads and step == grad_step:
            _save_gradients(model, save_grads, rank)

        torch.nn.utils.clip_grad_norm_(model.parameters(), train_cfg.max_grad_norm)
        optimizer.step()
        optimizer.zero_grad()

        torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0
        step_times.append(elapsed)

        if rank == 0 and step % 10 == 0:
            tps = tokens_per_step / elapsed
            print(f"[{mode}] step {step:4d}  loss={loss.item():.4f}  tok/s={tps:,.0f}  step_ms={elapsed*1000:.1f}")

    dist.destroy_process_group()

    mean_step = sum(step_times[warmup:]) / max(len(step_times[warmup:]), 1)
    return {
        "mode": mode,
        "tokens_per_sec": tokens_per_step / mean_step,
        "mean_step_ms": mean_step * 1000,
        "tokens_per_step": tokens_per_step,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="TorchTitan Llama 3.1 8B reference trainer")
    parser.add_argument("--mode", choices=["baseline", "compile", "fp8"], default="baseline",
                        help="Benchmark tier: baseline (FSDP2+BF16), compile (+torch.compile), fp8 (+Float8)")
    parser.add_argument("--steps", type=int, default=35, help="Number of training steps")
    parser.add_argument("--warmup", type=int, default=5,
                        help="Steps to exclude from throughput average")
    parser.add_argument("--save-grads", type=str, default=None,
                        help="Path to save gradient tensors (.pt) for accuracy checking")
    parser.add_argument("--grad-step", type=int, default=10,
                        help="Step at which to capture gradients")
    parser.add_argument("--output-json", type=str, default=None,
                        help="Path to write benchmark results JSON (rank 0 only)")
    parser.add_argument("--config", type=str, default=None, help="Path to config.json")
    args = parser.parse_args()

    results = train(args.steps, args.save_grads, args.grad_step, args.config,
                    args.warmup, args.mode)

    if int(os.environ.get("RANK", "0")) == 0:
        print(f"\n--- [{args.mode}] complete ---")
        for k, v in results.items():
            print(f"  {k}: {v:,.1f}" if isinstance(v, float) else f"  {k}: {v}")
        if args.output_json:
            out = Path(args.output_json)
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
