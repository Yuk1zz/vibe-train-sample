"""
Gradient correctness verifier for vibe-train candidates.

Compares accumulated gradients (before optimizer.step()) between the candidate
training system and the TorchTitan reference, using torch.allclose in FP32.

Primary check:
    torch.allclose(grad_candidate_fp32, grad_reference_fp32, rtol=1e-3, atol=1e-4)

Gradients are captured at step 1 (before any optimizer step) to avoid BF16 matmul
nondeterminism being amplified by AdamW over multiple steps.

Usage:
    # Step 1: Generate reference gradients (run once, save to disk)
    torchrun --nproc_per_node=<N> ../reference/reference.py \\
        --steps 1 --warmup 0 --save-grads /tmp/ref_grads.pt --grad-step 1

    # Step 2: Run candidate and save its gradients to the same step
    torchrun --nproc_per_node=<N> train.py \\
        --steps 1 --warmup 0 --save-grads /tmp/cand_grads.pt --grad-step 1

    # Step 3: Compare
    python checker.py --ref /tmp/ref_grads.pt --candidate /tmp/cand_grads.pt

    # Or run both and compare in one shot (requires VibeTrainModel interface)
    python checker.py --auto --model-dir /path/to/model

Exit codes: 0 = PASS, 1 = FAIL
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch


# Keep in sync with grad_rtol() / grad_atol() / grad_step() in templates/_config.j2
RTOL = 1e-3
ATOL = 1e-4
GRAD_STEP = 1


# ---------------------------------------------------------------------------
# Core comparison
# ---------------------------------------------------------------------------

def compare_gradients(
    ref_grads: dict[str, torch.Tensor],
    cand_grads: dict[str, torch.Tensor],
    rtol: float = RTOL,
    atol: float = ATOL,
    verbose: bool = True,
) -> tuple[bool, list[str]]:
    """Compare two gradient dicts. Returns (all_pass, list_of_failures)."""
    failures: list[str] = []
    missing_in_cand = set(ref_grads) - set(cand_grads)
    extra_in_cand = set(cand_grads) - set(ref_grads)

    if missing_in_cand:
        failures.append(f"Missing parameters in candidate: {sorted(missing_in_cand)}")
    if extra_in_cand and verbose:
        print(f"  [warn] Candidate has extra parameters (may be renamed): {sorted(extra_in_cand)}")

    common = set(ref_grads) & set(cand_grads)
    passed = 0
    total = len(common)

    for name in sorted(common):
        ref_g = ref_grads[name].float()
        cand_g = cand_grads[name].float()

        if ref_g.shape != cand_g.shape:
            failures.append(f"Shape mismatch for '{name}': ref={ref_g.shape} cand={cand_g.shape}")
            continue

        ok = torch.allclose(cand_g, ref_g, rtol=rtol, atol=atol)
        if ok:
            passed += 1
        else:
            max_abs_err = (cand_g - ref_g).abs().max().item()
            max_rel_err = ((cand_g - ref_g).abs() / (ref_g.abs() + 1e-8)).max().item()
            failures.append(
                f"MISMATCH '{name}': max_abs_err={max_abs_err:.2e}  max_rel_err={max_rel_err:.2e}"
            )

    if verbose:
        print(f"  Checked {total} parameter gradients: {passed} passed, {len(failures)} failed")

    return len(failures) == 0, failures


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare candidate vs reference gradients for gradient correctness."
    )
    parser.add_argument("--ref", type=str, required=True,
                        help="Path to reference gradient .pt file")
    parser.add_argument("--candidate", type=str, required=True,
                        help="Path to candidate gradient .pt file")
    parser.add_argument("--rtol", type=float, default=RTOL)
    parser.add_argument("--atol", type=float, default=ATOL)
    parser.add_argument("--verbose", action="store_true", default=True)
    args = parser.parse_args()

    ref_path = Path(args.ref)
    cand_path = Path(args.candidate)

    if not ref_path.exists():
        print(f"ERROR: Reference gradient file not found: {ref_path}")
        sys.exit(1)
    if not cand_path.exists():
        print(f"ERROR: Candidate gradient file not found: {cand_path}")
        sys.exit(1)

    print(f"Loading reference gradients from: {ref_path}")
    ref_grads: dict[str, torch.Tensor] = torch.load(ref_path, map_location="cpu", weights_only=True)
    print(f"  {len(ref_grads)} tensors loaded")

    print(f"Loading candidate gradients from: {cand_path}")
    cand_grads: dict[str, torch.Tensor] = torch.load(cand_path, map_location="cpu", weights_only=True)
    print(f"  {len(cand_grads)} tensors loaded")

    print(f"\nComparing gradients (rtol={args.rtol}, atol={args.atol}) ...")
    passed, failures = compare_gradients(ref_grads, cand_grads, args.rtol, args.atol, args.verbose)

    print()
    if passed:
        print("=" * 60)
        print("  GRADIENT CHECK PASSED")
        print("=" * 60)
        sys.exit(0)
    else:
        print("=" * 60)
        print("  GRADIENT CHECK FAILED")
        print("=" * 60)
        for msg in failures[:20]:
            print(f"  {msg}")
        if len(failures) > 20:
            print(f"  ... and {len(failures) - 20} more failures")
        sys.exit(1)


if __name__ == "__main__":
    main()
