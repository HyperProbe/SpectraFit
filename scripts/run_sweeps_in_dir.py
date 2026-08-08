#!/usr/bin/env python3
import argparse
import yaml
from pathlib import Path
import sys

import wandb
import torch

# make sure your project root is on sys.path
sys.path.append(str(Path(__file__).parent.parent))

from src.models.pipeline.model_pipeline import train_sweep


def main():
    p = argparse.ArgumentParser(
        description="Launch W&B sweeps from all YAML configs in a given directory"
    )
    p.add_argument(
        "-d",
        "--config_dir",
        type=Path,
        help="Path to directory containing sweep YAML configs",
        default=Path(__file__).parent.parent / "configs/sweeps",
    )
    p.add_argument(
        "-p",
        "--project",
        type=str,
        help="W&B project name",
        default="biopsy-autoencoder",
    )
    p.add_argument(
        "-n", "--count", type=int, help="Number of runs to launch per sweep", default=10
    )
    args = p.parse_args()

    config_dir = args.config_dir.expanduser()
    if not config_dir.exists():
        print(f"Error: config directory not found at {config_dir}", file=sys.stderr)
        sys.exit(1)

    if not config_dir.is_dir():
        print(f"Error: {config_dir} is not a directory", file=sys.stderr)
        sys.exit(1)

    # Find all YAML files in the directory
    yaml_files = list(config_dir.glob("*.yaml")) + list(config_dir.glob("*.yml"))

    if not yaml_files:
        print(f"Error: No YAML files found in {config_dir}", file=sys.stderr)
        sys.exit(1)

    print(f"Found {len(yaml_files)} YAML config files in {config_dir}")
    for yaml_file in yaml_files:
        print(f"  - {yaml_file.name}")

    wandb.login()

    # Process each YAML file
    for config_path in yaml_files:
        print(f"\n{'='*60}")
        print(f"Processing config: {config_path.name}")
        print(f"{'='*60}")

        try:
            with open(config_path, "r") as f:
                sweep_config = yaml.safe_load(f)

            sweep_id = wandb.sweep(sweep_config, project=args.project)
            print(
                f"Started sweep {sweep_id!r} in project {args.project!r} for {config_path.name}"
            )
            print(f"Launching {args.count} runs...")

            wandb.agent(sweep_id, function=train_sweep, count=args.count)

            print(f"Completed sweep for {config_path.name}")

        except Exception as e:
            print(f"Error processing {config_path.name}: {e}", file=sys.stderr)
            continue

    print(f"\n{'='*60}")
    print("All sweeps completed!")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
