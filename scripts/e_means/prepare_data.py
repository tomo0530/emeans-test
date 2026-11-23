"""Prepare datasets for each configured condition and data seed."""

import argparse
from pathlib import Path

import yaml
from project_module.e_means.contracts import ExperimentConfig
from project_module.e_means.data import prepare_condition


def main() -> None:
    """Read experiment configuration and prepare its datasets."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    args = parser.parse_args()
    config = ExperimentConfig.model_validate(yaml.safe_load(args.config.read_text()))
    for condition in config.conditions:
        for seed in condition.data_seeds:
            print(
                f"Preparing {condition.dataset_id}/{condition.condition_id}, data_seed={seed}",
                flush=True,
            )
            prepare_condition(condition, args.data_root, seed)


if __name__ == "__main__":
    main()
