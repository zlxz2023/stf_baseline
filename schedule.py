"""Run training configurations sequentially.

Each configuration/seed pair runs in its own Python process.  The next task is
started only after the previous process exits, which also releases model and
CUDA state between tasks.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import sys
from typing import Iterable


MODEL_ORDER = ("AGCRN", "GTS", "GWNet", "STAEformer", "STID", "STWave")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run stf_baseline training jobs one after another."
    )
    parser.add_argument(
        "--models",
        nargs="+",
        choices=MODEL_ORDER,
        default=list(MODEL_ORDER),
        help="Models to schedule (default: all models).",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        help="Optional dataset names to include, for example METR-LA PEMS08.",
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=[2026],
        help="Random seeds; every seed is scheduled as a separate task.",
    )
    parser.add_argument(
        "--gpus",
        default="0",
        help="CUDA_VISIBLE_DEVICES value passed to every task (default: 0).",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Continue with later tasks when a task exits unsuccessfully.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the task queue without starting training.",
    )
    return parser.parse_args()


def iter_configs(root: Path, models: Iterable[str], datasets: set[str] | None):
    model_rank = {name: index for index, name in enumerate(MODEL_ORDER)}
    model_names = sorted(set(models), key=model_rank.__getitem__)
    configs = []
    for model in model_names:
        model_dir = root / "configs" / model
        for config_path in sorted(model_dir.glob("*.yaml"), key=lambda p: p.name.lower()):
            if datasets is not None and config_path.stem not in datasets:
                continue
            configs.append((model, config_path))
    return configs


def main() -> int:
    args = parse_args()
    root = Path(__file__).resolve().parent
    configs = iter_configs(root, args.models, set(args.datasets) if args.datasets else None)
    if args.datasets:
        missing = set(args.datasets) - {path.stem for _, path in configs}
        if missing:
            raise SystemExit("No matching configurations for dataset(s): " + ", ".join(sorted(missing)))
    tasks = [(model, path, seed) for model, path in configs for seed in args.seeds]

    if not tasks:
        raise SystemExit("No matching YAML configurations were found.")

    print(f"Queued {len(tasks)} task(s) under {root}", flush=True)
    if args.dry_run:
        for index, (model, path, seed) in enumerate(tasks, 1):
            print(f"[{index}/{len(tasks)}] {model} | {path.relative_to(root)} | seed={seed}")
        return 0

    python = sys.executable
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = args.gpus
    failures = []

    for index, (model, config_path, seed) in enumerate(tasks, 1):
        relative_config = config_path.relative_to(root)
        command = [
            python,
            "-u",
            str(root / "train.py"),
            "--model",
            model,
            "--cfg",
            str(relative_config),
            "--gpus",
            args.gpus,
            "--seed",
            str(seed),
        ]
        print(f"\n===== task {index}/{len(tasks)}: {model} | {relative_config} | seed={seed} =====", flush=True)
        completed = subprocess.run(command, cwd=root, env=env)
        if completed.returncode == 0:
            print(f"===== task {index}/{len(tasks)} finished successfully =====")
            continue

        failures.append((model, relative_config, seed, completed.returncode))
        print(f"===== task {index}/{len(tasks)} failed with exit code {completed.returncode} =====")
        if not args.continue_on_error:
            print("Stopping because --continue-on-error was not specified.")
            break

    completed_count = index - len(failures)
    print(f"\nSucceeded: {completed_count}; failed: {len(failures)}; pending: {len(tasks) - index}.")
    if failures:
        print("Failed task(s):")
        for model, config_path, seed, returncode in failures:
            print(f"  {model} | {config_path} | seed={seed} | exit={returncode}")
        return 1
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\nSchedule interrupted; no further tasks will be started.", flush=True)
        raise SystemExit(130)
