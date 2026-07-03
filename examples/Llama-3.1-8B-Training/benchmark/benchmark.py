"""
Training throughput benchmark for vibe-train candidates.

Measures:
  - tokens/sec (primary metric)
  - step time (ms)
  - peak GPU memory per rank (GB)
  - loss (to confirm no divergence)

The benchmark drives the candidate training script and collects structured metrics.
The candidate must expose a VibeTrainModel interface OR accept CLI flags
--steps N --output-json PATH.

Usage (CLI mode — candidate exposes --output-json):
    torchrun --nproc_per_node=8 benchmark.py \\
        --cmd "torchrun --nproc_per_node=8 train.py" \\
        --steps 50 --warmup 10 --output-json bench_result.json

Usage (import mode — candidate exposes VibeTrainModel):
    python benchmark.py --import-mode --steps 50 --output-json bench_result.json

Exit codes: 0 = success, 1 = error
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path


# ---------------------------------------------------------------------------
# Import-mode: drives VibeTrainModel directly
# ---------------------------------------------------------------------------

def run_import_mode(args: argparse.Namespace) -> dict:
    try:
        from train import VibeTrainModel  # type: ignore
    except ImportError as exc:
        print(f"ERROR: Cannot import VibeTrainModel from train.py: {exc}")
        sys.exit(1)

    config_path = args.config or str(Path(__file__).parent.parent / "reference" / "config.json")
    device = args.device or "cuda"
    dtype_map = {"bfloat16": __import__("torch").bfloat16, "float16": __import__("torch").float16}
    dtype = dtype_map.get(args.dtype, __import__("torch").bfloat16)

    print(f"Initializing VibeTrainModel (config={config_path}, device={device}, dtype={dtype})")
    trainer = VibeTrainModel.from_config(config_path, device, dtype)

    # Warmup
    if args.warmup > 0:
        print(f"Warming up ({args.warmup} steps) ...")
        trainer.train(args.warmup)

    # Benchmark
    print(f"Benchmarking ({args.steps} steps) ...")
    t0 = time.perf_counter()
    result = trainer.train(args.steps)
    elapsed = time.perf_counter() - t0

    return _build_result(result, elapsed, args)


# ---------------------------------------------------------------------------
# CLI-mode: subprocess + structured JSON output from the candidate
# ---------------------------------------------------------------------------

def run_cli_mode(args: argparse.Namespace) -> dict:
    if not args.cmd:
        print("ERROR: --cmd is required in CLI mode")
        sys.exit(1)

    out_path = "/tmp/vibe_train_bench_raw.json"
    cmd = args.cmd.split() + [
        "--steps", str(args.steps + args.warmup),
        "--output-json", out_path,
    ]

    print(f"Running: {' '.join(cmd)}")
    t0 = time.perf_counter()
    proc = subprocess.run(cmd, capture_output=False, text=True)
    elapsed = time.perf_counter() - t0

    if proc.returncode != 0:
        print(f"ERROR: Training command failed with exit code {proc.returncode}")
        sys.exit(1)

    if not Path(out_path).exists():
        print(f"ERROR: Candidate did not write output JSON to {out_path}")
        print("  Candidate must accept --output-json PATH and write structured metrics.")
        sys.exit(1)

    raw = json.loads(Path(out_path).read_text())
    return _build_result(raw, elapsed, args)


# ---------------------------------------------------------------------------
# Result builder
# ---------------------------------------------------------------------------

def _build_result(raw: dict, wall_clock: float, args: argparse.Namespace) -> dict:
    tps = raw.get("tokens_per_sec", 0.0)
    step_ms = raw.get("mean_step_ms", 0.0)
    peak_mem = raw.get("peak_gpu_memory_gb", None)
    final_loss = raw.get("final_loss", None)
    samples_per_sec = raw.get("samples_per_sec", None)

    result = {
        "config": {
            "steps": args.steps,
            "warmup_steps": args.warmup,
            "wall_clock_s": wall_clock,
        },
        "tokens_per_sec": tps,
        "mean_step_ms": step_ms,
        "peak_gpu_memory_gb": peak_mem,
        "final_loss": final_loss,
        "samples_per_sec": samples_per_sec,
    }

    print()
    print("=" * 50)
    print("  Training Benchmark Results")
    print("=" * 50)
    print(f"  Steps:            {args.steps} (+ {args.warmup} warmup)")
    print(f"  Tokens/sec:       {tps:,.0f}")
    if step_ms:
        print(f"  Step time:        {step_ms:.1f} ms")
    if peak_mem is not None:
        print(f"  Peak GPU mem:     {peak_mem:.1f} GB")
    if final_loss is not None:
        print(f"  Final loss:       {final_loss:.4f}")

    # Reference targets for context
    print()
    print("  TorchTitan reference targets (8×H100, Llama 3.1 8B):")
    print("    FSDP:                    6,258 tok/sec")
    print("    + torch.compile:         6,674 tok/sec")
    print("    + torch.compile + FP8:   9,409 tok/sec")

    return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Training throughput benchmark")
    parser.add_argument("--steps", type=int, default=50,
                        help="Number of benchmark steps (after warmup)")
    parser.add_argument("--warmup", type=int, default=10,
                        help="Warmup steps (excluded from metrics)")
    parser.add_argument("--output-json", type=str, default=None,
                        help="Write structured results to this path")
    parser.add_argument("--import-mode", action="store_true",
                        help="Import VibeTrainModel directly instead of subprocess")
    parser.add_argument("--cmd", type=str, default=None,
                        help="Training command to run (CLI mode)")
    parser.add_argument("--config", type=str, default=None,
                        help="Path to config.json (import mode)")
    parser.add_argument("--device", type=str, default=None,
                        help="Device string (import mode)")
    parser.add_argument("--dtype", type=str, default="bfloat16",
                        help="Dtype string: bfloat16 or float16 (import mode)")
    args = parser.parse_args()

    if args.import_mode:
        result = run_import_mode(args)
    else:
        result = run_cli_mode(args)

    if args.output_json:
        Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output_json).write_text(json.dumps(result, indent=2))
        print(f"\nResults written to {args.output_json}")

    # Fail if no measurable throughput
    if result.get("tokens_per_sec", 0) <= 0:
        print("\nERROR: tokens_per_sec = 0; candidate is not reporting throughput correctly.")
        sys.exit(1)


if __name__ == "__main__":
    main()
