#!/usr/bin/env python3
import argparse
import yaml
from pathlib import Path

import wandb

# make sure your project root is on sys.path
import sys
sys.path.append(str(Path(__file__).parent.parent))

from src.models.pipeline.model_pipeline import train_sweep

def main():
    p = argparse.ArgumentParser(
        description="Launch a W&B sweep from a given YAML config"
    )
    p.add_argument(
        "-c", "--config",
        type=Path,
        help="Path to sweep YAML config",
        default=Path(__file__).parent.parent / "configs/sweeps/test_sweep.yaml"
    )
    p.add_argument(
        "-p", "--project",
        type=str,
        help="W&B project name",
        default="biopsy-autoencoder"
    )
    p.add_argument(
        "-n", "--count",
        type=int,
        help="Number of runs to launch",
        default=10
    )
    args = p.parse_args()

    config_path = args.config.expanduser()
    if not config_path.exists():
        print(f"Error: config file not found at {config_path}", file=sys.stderr)
        sys.exit(1)

    wandb.login()
    with open(config_path, "r") as f:
        sweep_config = yaml.safe_load(f)

    sweep_id = wandb.sweep(sweep_config, project=args.project)
    print(f"Started sweep {sweep_id!r} in project {args.project!r}. Launching {args.count} runs…")
    wandb.agent(sweep_id, function=train_sweep, count=args.count)


if __name__ == "__main__":
    main()
