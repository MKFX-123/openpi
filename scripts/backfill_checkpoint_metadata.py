"""Backfill self-contained metadata for a legacy checkpoint from a W&B run."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import tyro

from openpi.training import checkpoint_metadata


@dataclass(frozen=True)
class Args:
    checkpoint_dir: Path
    wandb_run_dir: Path
    base_config_name: str | None = None
    overwrite: bool = False


def main(args: Args) -> None:
    metadata_dir = args.checkpoint_dir / "metadata"
    if metadata_dir.exists() and not args.overwrite:
        raise FileExistsError(f"Metadata already exists: {metadata_dir}. Pass --overwrite to replace it.")

    config = checkpoint_metadata.restore_train_config_from_wandb(
        args.wandb_run_dir,
        base_config_name=args.base_config_name,
    )
    checkpoint_metadata.save(metadata_dir, config)
    print(f"Backfilled checkpoint metadata at {metadata_dir}")


if __name__ == "__main__":
    main(tyro.cli(Args))
