"""Entry point for ``python -m vibe_train``.

Usage:
    python -m vibe_train [options]
    torchrun --nproc_per_node=1 -m vibe_train [options]   # local single-GPU test

Invokes the issue-tracker driven training loop (vibe_train equivalent of
``vibe-serve --outer-loop plain``). All flags mirror the plain loop; only
the defaults differ (training reference, training skills, training example).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Import shared infrastructure from vibe_serve (not modifying it, only reading)
from vibe_serve.cli import (
    load_config_and_skills,
    run_environment_spec_from_args,
    _resolve_run_dir,  # noqa: PLC2701 — internal helper, acceptable cross-package use
)
from vibe_serve.constants import PROJECT_ROOT

from vibe_train.constants import ComputeBackend, KNOWN_COMPUTE_BACKENDS
from vibe_train.loop import TrainLoopState, _load_state, run_train_loop


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m vibe_train",
        description=(
            "Issue-tracker driven training loop: perf_eval files optimization "
            "issues, implementer drains them one at a time, judge verifies "
            "gradient correctness."
        ),
    )

    # ------------------------------------------------------------------ #
    # Common args (mirrors vibe_serve.cli._add_common_args with training  #
    # defaults)                                                            #
    # ------------------------------------------------------------------ #
    parser.add_argument(
        "--ref",
        default="examples/Llama-3.1-8B-Training/reference",
        help="Path to reference implementation directory (default: examples/Llama-3.1-8B-Training/reference)",
    )
    parser.add_argument(
        "--exp-name",
        required=False,
        default="train-test",
        help="Experiment name (creates exp_env/<name>/)",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "agent.toml",
        help="Path to agent TOML config file (default: agent.toml)",
    )
    parser.add_argument(
        "--acc-checker",
        type=Path,
        default=Path("examples/Llama-3.1-8B-Training/accuracy_checker"),
        help="Path to directory containing accuracy checker code (default: examples/Llama-3.1-8B-Training/accuracy_checker).",
    )
    parser.add_argument(
        "--bench",
        type=Path,
        default=Path("examples/Llama-3.1-8B-Training/benchmark"),
        help="Path to directory containing benchmark code (default: examples/Llama-3.1-8B-Training/benchmark).",
    )
    parser.add_argument(
        "--nsys-profiler",
        type=Path,
        default=None,
        help="Path to directory containing nsys analysis script (analyze_nsys.py).",
    )
    parser.add_argument(
        "--profiler",
        choices=["nsys", "torch", "auto"],
        default="auto",
        help=(
            "Which profiler to use between rounds. "
            "'nsys' for NVIDIA Nsight Systems, "
            "'torch' for torch.profiler, "
            "'auto' picks torch when --modal is set, else nsys. Default: auto."
        ),
    )
    parser.add_argument(
        "--skills-dir",
        default=[Path("resources/skills/training-systems")],
        action="append",
        type=Path,
        help=(
            "Path to a skill source directory (can be repeated). "
            "Default: resources/skills/training-systems/."
        ),
    )
    parser.add_argument(
        "--no-skills",
        action="store_true",
        help="Disable skills entirely (ablation mode). Overrides --skills-dir.",
    )
    parser.add_argument(
        "--docker",
        action="store_true",
        help="Run agent operations inside a Docker container.",
    )
    parser.add_argument(
        "--docker-image",
        type=str,
        default=None,
        help="Docker image to use (with --docker or --modal).",
    )
    parser.add_argument(
        "--modal",
        action="store_true",
        help="Use Modal for remote GPU dispatch.",
    )
    parser.add_argument(
        "--modal-gpu",
        type=str,
        default="H100",
        help="Default Modal GPU spec (e.g. H100, A100). Default: H100.",
    )
    parser.add_argument(
        "--modal-model-volume",
        type=str,
        default=None,
        help="Name of a pre-existing Modal Volume holding model weights.",
    )
    parser.add_argument(
        "--modal-app",
        type=str,
        default="vibetrain",
        help="Default Modal App name. Default: vibetrain.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Pause for Enter at each step.",
    )
    parser.add_argument(
        "--git-tracking",
        action="store_true",
        help="Track workspace versions via git commits.",
    )
    parser.add_argument(
        "--agent-backend",
        choices=["deepagents", "cli"],
        default=None,
        help="Agent runner backend. Overrides [agent].backend in agent.toml.",
    )
    parser.add_argument(
        "--cli-provider",
        choices=["claude", "gemini", "codex", "opencode"],
        default=None,
        help="Which CLI coding-agent to drive when --agent-backend=cli.",
    )
    parser.add_argument(
        "--backend",
        type=ComputeBackend,
        choices=list(ComputeBackend),
        default=None,
        help=(
            "Compute backend to target. Overrides [backend].name in agent.toml. "
            f"Defaults to 'cuda'. Supported: {', '.join(KNOWN_COMPUTE_BACKENDS)}."
        ),
    )

    # ------------------------------------------------------------------ #
    # Train-loop-specific args                                             #
    # ------------------------------------------------------------------ #
    parser.add_argument("--max-rounds", type=int, default=5)
    parser.add_argument("--max-attempts-per-issue", type=int, default=3)
    parser.add_argument("--max-issues-per-perf-eval", type=int, default=3)
    parser.add_argument("--start-round", type=int, default=None, metavar="N")
    parser.add_argument(
        "--resume",
        nargs="?",
        const="latest",
        default=None,
        metavar="RUN_DIR",
        help=(
            "Resume an existing run. Pass the run-dir name under exp_env/ "
            "or omit the value to use the most recent run ('latest')."
        ),
    )

    return parser


def main(argv: list[str] | None = None) -> None:
    args = _build_parser().parse_args(argv)

    if args.modal and args.profiler == "nsys":
        print("Error: --modal only supports --profiler=torch.", file=sys.stderr)
        sys.exit(2)

    config, skills, backend = load_config_and_skills(args)

    existing = False
    exp_name = args.exp_name
    resume_state: TrainLoopState | None = None

    if args.resume is not None:
        run_dir_name = _resolve_run_dir(args.resume)
        exp_name = run_dir_name
        existing = True
        print(f"Resuming from: exp_env/{run_dir_name}/")

        exp_dir = PROJECT_ROOT / "exp_env" / run_dir_name
        log_dir = exp_dir / "logs"

        if args.start_round is not None:
            resume_state = TrainLoopState(
                round_idx=args.start_round - 1, bootstrap_done=True,
            )
        else:
            resume_state = _load_state(log_dir)
            if resume_state is not None:
                print(
                    f"Auto-detected state: iteration {resume_state.round_idx + 1}, "
                    f"phase '{resume_state.phase}'"
                    + (
                        f", current issue #{resume_state.current_issue_id}"
                        if resume_state.current_issue_id
                        else ""
                    )
                )
            else:
                resume_state = TrainLoopState(bootstrap_done=True)
                print(
                    "Warning: state.json not found. Starting fresh "
                    "(bootstrap will be skipped because existing run)."
                )

    success = run_train_loop(
        config=config,
        exp_name=exp_name,
        reference_path=args.ref,
        max_rounds=args.max_rounds,
        max_attempts_per_issue=args.max_attempts_per_issue,
        max_issues_per_perf_eval=args.max_issues_per_perf_eval,
        existing=existing,
        resume_state=resume_state,
        debug=args.debug,
        acc_checker=str(args.acc_checker) if args.acc_checker else None,
        bench=str(args.bench) if args.bench else None,
        nsys_profiler=str(args.nsys_profiler) if args.nsys_profiler else None,
        skills_dirs=skills,
        run_environment=run_environment_spec_from_args(args),
        agent_backend=args.agent_backend,
        cli_provider=args.cli_provider,
        backend=backend,
    )

    if success:
        print("\nTrain loop completed: no remaining open issues.")
    else:
        print(f"\nTrain loop did not complete after {args.max_rounds} rounds.")
        sys.exit(1)


if __name__ == "__main__":
    main()
