"""Compute normalization statistics for debiased hdf5 data.

This script computes the normalization statistics for hdf5 data. It computes
the mean, standard deviation, and quantiles (q01, q99) for state and actions.
The output path is determined by the config, similar to compute_norm_stats.py.
It applies the same padding to expand 14-dim data to 32-dim.
"""

import pathlib

import h5py
import numpy as np
import tqdm

import openpi.shared.normalize as normalize
import openpi.training.config as _config
import openpi.transforms as transforms


def compute_norm_stats(
    config_name: str,
    data_dir: str = "/mnt/public/jzc/debiased/epoch15/trajectory_chunks/",
):
    """Compute normalization statistics for hdf5 data.

    Args:
        config_name: Name of the config to use for determining output path and action_dim
        data_dir: Directory containing subdirectories with hdf5 files
    """
    # Get config to determine output path and action_dim
    config = _config.get_config(config_name)
    data_config = config.data.create(config.assets_dirs, config.model)

    # Get action_dim from the first ArxInputs transform
    action_dim = None
    for transform in data_config.data_transforms.inputs:
        if hasattr(transform, 'action_dim'):
            action_dim = transform.action_dim
            break

    if action_dim is None:
        action_dim = 32  # Default fallback
        print(f"Warning: Could not find action_dim in transforms, using default {action_dim}")

    print(f"Using action_dim={action_dim} for padding")

    # For multi-dataset, concatenate dataset names with underscores
    repo_ids = [repo_id.strip() for repo_id in data_config.repo_id.split(",") if repo_id.strip()]
    if len(repo_ids) > 1:
        asset_name = "_".join(repo_ids)
    else:
        asset_name = data_config.repo_id

    output_path = config.assets_dirs / asset_name
    output_path.mkdir(parents=True, exist_ok=True)

    # Find all hdf5 files
    data_path = pathlib.Path(data_dir)
    hdf5_files = list(data_path.glob("*/*.h5"))
    print(f"Found {len(hdf5_files)} hdf5 files in {data_dir}")

    if len(hdf5_files) == 0:
        print(f"No hdf5 files found in {data_dir}")
        return

    # Initialize running stats
    keys = ["state", "actions"]
    stats = {key: normalize.RunningStats() for key in keys}

    # Process each file
    for file_path in tqdm.tqdm(hdf5_files, desc="Processing files"):
        try:
            with h5py.File(file_path, "r") as f:
                # Get states and actions
                states = f["states"][:]  # (N, 14)
                action_chunks = f["action_chunks"][:]  # (N, 30, 14)

                # Pad to action_dim (14 -> 32)
                padded_states = transforms.pad_to_dim(states, action_dim)  # (N, 32)
                padded_actions = transforms.pad_to_dim(action_chunks, action_dim)  # (N, 30, 32)

                # Update stats
                stats["state"].update(np.asarray(padded_states))
                stats["actions"].update(np.asarray(padded_actions))
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            continue

    # Compute final statistics
    norm_stats = {key: stats.get_statistics() for key, stats in stats.items()}

    # Save to file
    print(f"Writing stats to: {output_path}")
    print(f"Datasets used: {repo_ids}")

    # Use the normalize.save function to maintain consistency with the original script
    normalize.save(output_path, norm_stats)

    print(f"Total files processed: {len(hdf5_files)}")
    print(f"\nPadded state shape: (N, {action_dim})")
    print(f"Padded actions shape: (N, 30, {action_dim})")
    print(f"\nState statistics:")
    print(f"  mean (first 5): {norm_stats['state'].mean[:5].tolist()}")
    print(f"  std (first 5): {norm_stats['state'].std[:5].tolist()}")
    print(f"  mean (last 5, should be ~0): {norm_stats['state'].mean[-5:].tolist()}")
    print(f"  std (last 5, should be ~0): {norm_stats['state'].std[-5:].tolist()}")
    print(f"\nAction statistics:")
    print(f"  mean (first 5): {norm_stats['actions'].mean[:5].tolist()}")
    print(f"  std (first 5): {norm_stats['actions'].std[:5].tolist()}")
    print(f"  mean (last 5, should be ~0): {norm_stats['actions'].mean[-5:].tolist()}")
    print(f"  std (last 5, should be ~0): {norm_stats['actions'].std[-5:].tolist()}")


if __name__ == "__main__":
    import tyro

    tyro.cli(compute_norm_stats)
