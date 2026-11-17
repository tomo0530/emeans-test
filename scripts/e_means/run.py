"""Run the frozen CPU or GPU e-means benchmark phase."""

import argparse
from pathlib import Path

import yaml
from project_module.e_means.contracts import ExperimentConfig
from project_module.e_means.runner import run_phase


def main() -> None:
    """Parse the experiment CLI and execute the requested phase."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--phase", choices=("cpu", "gpu"), required=True)
    parser.add_argument("--devices")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--retry-failed", action="store_true")
    parser.add_argument("--retry-reason")
    args = parser.parse_args()
    config = ExperimentConfig.model_validate(yaml.safe_load(args.config.read_text()))
    devices = [int(value) for value in args.devices.split(",")] if args.devices else None
    jobs = run_phase(
        config,
        args.data_root,
        args.output_root,
        phase=args.phase,
        devices=devices,
        resume=args.resume,
        dry_run=args.dry_run,
        retry_failed=args.retry_failed,
        retry_reason=args.retry_reason,
    )
    print(f"Frozen {len(jobs)} jobs; phase={args.phase}; dry_run={args.dry_run}")


if __name__ == "__main__":
    main()
