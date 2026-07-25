from collections.abc import Iterator, Sequence
import json
import logging
import multiprocessing
import os
from pathlib import Path
import typing
from typing import Literal, Protocol, SupportsIndex, TypeVar

import jax
import jax.numpy as jnp
import lerobot.common.datasets.lerobot_dataset as lerobot_dataset
import numpy as np
import torch

import openpi.models.model as _model
import openpi.training.config as _config
from openpi.training.droid_rlds_dataset import DroidRldsDataset
import openpi.transforms as _transforms

T_co = TypeVar("T_co", covariant=True)


class Dataset(Protocol[T_co]):
    """Interface for a dataset with random access."""

    def __getitem__(self, index: SupportsIndex) -> T_co:
        raise NotImplementedError("Subclasses of Dataset should implement __getitem__.")

    def __len__(self) -> int:
        raise NotImplementedError("Subclasses of Dataset should implement __len__.")


class IterableDataset(Protocol[T_co]):
    """Interface for an iterable dataset."""

    def __iter__(self) -> Iterator[T_co]:
        raise NotImplementedError("Subclasses of IterableDataset should implement __iter__.")

    def __len__(self) -> int:
        raise NotImplementedError("Subclasses of Dataset should implement __len__.")


class DataLoader(Protocol[T_co]):
    """Interface for a data loader."""

    def data_config(self) -> _config.DataConfig:
        """Get the data config for this data loader."""
        raise NotImplementedError("Subclasses of DataLoader should implement data_config.")

    def __iter__(self) -> Iterator[T_co]:
        raise NotImplementedError("Subclasses of DataLoader should implement __iter__.")


class FilteredLeRobotDataset(lerobot_dataset.LeRobotDataset):
    """Fixed LeRobotDataset that properly handles episode filtering.
    
    LeRobot has a bug where when using episodes=[131, 19, 81], the episode_data_index
    is correctly created with N entries (one per filtered episode), but hf_dataset's 
    episode_index column retains original values (e.g., 131, 19, 81). 
    The _get_query_indices method then tries to access episode_data_index["from"][131] 
    which is out of bounds.
    
    This subclass fixes the bug by maintaining an episode index mapping based on
    the actual order of episodes in the filtered dataset.
    """
    
    def __init__(self, *args, episodes: list[int] | None = None, **kwargs):
        # Store the episodes list before calling parent __init__
        self._filtered_episodes = episodes
        self._episode_index_map: dict[int, int] | None = None
        
        super().__init__(*args, episodes=episodes, **kwargs)
        
        # Build the episode index mapping after parent init
        # IMPORTANT: Map based on actual order in episode_data_index, not sorted order
        if episodes is not None and len(self.episode_data_index["from"]) > 0:
            self._episode_index_map = {}
            for new_idx in range(len(self.episode_data_index["from"])):
                # Get the first sample index for this episode in episode_data_index
                start_idx = self.episode_data_index["from"][new_idx].item()
                # Get the original episode index from hf_dataset
                orig_ep_idx = self.hf_dataset[start_idx]["episode_index"]
                if hasattr(orig_ep_idx, 'item'):
                    orig_ep_idx = orig_ep_idx.item()
                self._episode_index_map[orig_ep_idx] = new_idx
    
    def _get_query_indices(self, idx: int, ep_idx: int) -> tuple[dict[str, list[int | bool]]]:
        """Override to fix episode index mapping bug.
        
        ep_idx here is the ORIGINAL episode index from hf_dataset.
        We need to remap it to access episode_data_index correctly.
        """
        # Remap the episode index if we're using filtered episodes
        remapped_ep_idx = ep_idx
        if self._episode_index_map is not None:
            remapped_ep_idx = self._episode_index_map.get(ep_idx, ep_idx)
        
        # Now use remapped index to access episode_data_index
        ep_start = self.episode_data_index["from"][remapped_ep_idx]
        ep_end = self.episode_data_index["to"][remapped_ep_idx]
        
        query_indices = {
            key: [max(ep_start.item(), min(ep_end.item() - 1, idx + delta)) for delta in delta_idx]
            for key, delta_idx in self.delta_indices.items()
        }
        padding = {
            f"{key}_is_pad": torch.BoolTensor(
                [(idx + delta < ep_start.item()) | (idx + delta >= ep_end.item()) for delta in delta_idx]
            )
            for key, delta_idx in self.delta_indices.items()
        }
        return query_indices, padding
    
    def _query_videos(self, query_timestamps: dict[str, list[float]], ep_idx: int):
        """Override to use original episode index for video file paths.
        
        ep_idx here is the ORIGINAL episode index - we should NOT remap it
        because video files are named using the original episode indices.
        """
        # Do NOT remap ep_idx here - video files use original episode indices
        try:
            return super()._query_videos(query_timestamps, ep_idx)
        except Exception as e:
            video_paths = {
                key: str(self.root / self.meta.get_video_file_path(ep_idx, key))
                for key in self.meta.video_keys
            }
            raise RuntimeError(
                "Video decode failed. "
                f"repo_id={self.repo_id}, ep_idx={ep_idx}, "
                f"query_timestamps={query_timestamps}, video_paths={video_paths}"
            ) from e


class MultiDataset(Dataset[T_co]):
    """Dataset that combines multiple LeRobot datasets."""
    
    def __init__(self, datasets: Sequence[Dataset[T_co]]):
        self._datasets = list(datasets)
        self._cumulative_lengths = self._compute_cumulative_lengths()
        
    def _compute_cumulative_lengths(self) -> list[int]:
        """Compute cumulative lengths for efficient indexing."""
        cumulative_lengths = []
        total = 0
        for dataset in self._datasets:
            total += len(dataset)
            cumulative_lengths.append(total)
        return cumulative_lengths
        
    def __getitem__(self, index: SupportsIndex) -> T_co:
        idx = index.__index__()
        if idx < 0:
            idx += len(self)
        if idx < 0 or idx >= len(self):
            raise IndexError(f"Index {index} is out of range for dataset of length {len(self)}")
            
        # Find which dataset contains this index
        dataset_idx = 0
        for i, cumulative_length in enumerate(self._cumulative_lengths):
            if idx < cumulative_length:
                dataset_idx = i
                break
                
        # Calculate the offset within the selected dataset
        offset = idx
        if dataset_idx > 0:
            offset = idx - self._cumulative_lengths[dataset_idx - 1]
            
        return self._datasets[dataset_idx][offset]
        
    def __len__(self) -> int:
        if not self._cumulative_lengths:
            return 0
        return self._cumulative_lengths[-1]


class TransformedDataset(Dataset[T_co]):
    def __init__(self, dataset: Dataset, transforms: Sequence[_transforms.DataTransformFn]):
        self._dataset = dataset
        self._transform = _transforms.compose(transforms)

    def __getitem__(self, index: SupportsIndex) -> T_co:
        return self._transform(self._dataset[index])

    def __len__(self) -> int:
        return len(self._dataset)


def _int_column(values) -> np.ndarray:
    return np.fromiter(
        (int(value.item()) if hasattr(value, "item") else int(value) for value in values),
        dtype=np.int64,
        count=len(values),
    )


class LeRobotSampleFilterDataset(Dataset[T_co]):
    """Apply frame-window filters to a LeRobot dataset without modifying its rows."""

    def __init__(
        self,
        dataset: Dataset[T_co],
        *,
        action_horizon: int,
        state_history_size: int,
        state_future_size: int,
        state_step: int,
        filter_cross_task_action_chunks: bool,
        filter_issue_samples: bool,
    ):
        hf_dataset = getattr(dataset, "hf_dataset", None)
        if hf_dataset is None:
            raise TypeError("LeRobotSampleFilterDataset requires a LeRobot dataset with hf_dataset")
        if "episode_index" not in hf_dataset.column_names:
            raise ValueError("LeRobot dataset has no episode_index column")

        episode_indices = _int_column(hf_dataset["episode_index"])
        keep = np.ones(len(episode_indices), dtype=bool)
        self.task_filtered_samples = 0
        self.issue_filtered_samples = 0

        if filter_cross_task_action_chunks:
            if "task_index" not in hf_dataset.column_names:
                raise ValueError("LeRobot dataset has no task_index column")
            task_indices = _int_column(hf_dataset["task_index"])
            task_keep = np.ones(len(task_indices), dtype=bool)
            task_boundaries = np.flatnonzero(
                (episode_indices[1:] == episode_indices[:-1])
                & (task_indices[1:] != task_indices[:-1])
            ) + 1
            for boundary in task_boundaries:
                first = max(0, int(boundary) - action_horizon + 1)
                candidates = np.arange(first, boundary, dtype=np.int64)
                same_episode = episode_indices[candidates] == episode_indices[boundary]
                task_keep[candidates[same_episode]] = False
            self.task_filtered_samples = int((~task_keep).sum())
            keep &= task_keep

        if filter_issue_samples:
            if "frame_index" not in hf_dataset.column_names:
                raise ValueError("LeRobot dataset has no frame_index column")
            root = Path(str(getattr(dataset, "root")))
            metadata_path = root / "meta" / "data_quality" / "excluded_sample_ranges.json"
            if not metadata_path.is_file():
                raise FileNotFoundError(
                    f"Issue sample filtering requested but metadata is missing: {metadata_path}"
                )
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            frame_indices = _int_column(hf_dataset["frame_index"])
            issue_keep = np.ones(len(frame_indices), dtype=bool)
            min_delta = -state_history_size * state_step
            max_delta = max(action_horizon - 1, state_future_size * state_step)

            for episode in metadata.get("episodes", []):
                episode_index = int(episode["episode_index"])
                episode_mask = episode_indices == episode_index
                for start, end in episode.get("sample_ranges", []):
                    invalid_start = int(start) - max_delta
                    invalid_end = int(end) - min_delta
                    issue_keep[
                        episode_mask
                        & (frame_indices >= invalid_start)
                        & (frame_indices < invalid_end)
                    ] = False
            self.issue_filtered_samples = int((~issue_keep).sum())
            keep &= issue_keep

        self._dataset = dataset
        self._indices = np.flatnonzero(keep)
        self.total_samples = len(keep)
        self.filtered_samples = self.total_samples - len(self._indices)

    def __getitem__(self, index: SupportsIndex) -> T_co:
        mapped_index = index.__index__()
        if mapped_index < 0:
            mapped_index += len(self)
        if mapped_index < 0 or mapped_index >= len(self):
            raise IndexError(f"Index {index} is out of range for dataset of length {len(self)}")
        return self._dataset[int(self._indices[mapped_index])]

    def __len__(self) -> int:
        return len(self._indices)


class IterableTransformedDataset(IterableDataset[T_co]):
    def __init__(
        self,
        dataset: IterableDataset,
        transforms: Sequence[_transforms.DataTransformFn],
        *,
        is_batched: bool = False,
    ):
        self._dataset = dataset
        self._transform = _transforms.compose(transforms)
        self._is_batched = is_batched

    def __iter__(self):
        for sample in self._dataset:
            if self._is_batched:
                # Transforms are designed to be applied to individual samples. So we need to split the batch into
                # individual samples and apply the transform to each sample individually.
                batch_size = next(v.shape[0] for v in sample.values())

                # Split batch into individual samples using tree_map
                individual_samples = [jax.tree.map(lambda x: x[i], sample) for i in range(batch_size)]  # noqa: B023

                # Transform each sample
                transformed = [self._transform(s) for s in individual_samples]

                # Recombine batch with tree_map
                yield jax.tree.map(lambda *x: np.stack(x, axis=0), *transformed)
            else:
                yield self._transform(sample)

    def __len__(self) -> int:
        return len(self._dataset)


class FakeDataset(Dataset):
    def __init__(self, model_config: _model.BaseModelConfig, num_samples: int):
        self._num_samples = num_samples
        self._observation_spec, self._action_spec = model_config.inputs_spec()

    def __getitem__(self, index: SupportsIndex) -> dict:
        rng = jax.random.key(index.__index__())

        def make_from_spec(spec: jax.ShapeDtypeStruct):
            nonlocal rng
            rng, data_rng = jax.random.split(rng)
            # Remove the batch dimension.
            shape = spec.shape[1:]
            if spec.dtype == jnp.float32:
                return jax.random.uniform(data_rng, shape=shape, minval=-1.0, maxval=1.0)
            if spec.dtype == jnp.int32:
                return jax.random.randint(data_rng, shape=shape, minval=0, maxval=2048)
            return jnp.zeros(shape=shape, dtype=spec.dtype)

        observation = jax.tree.map(make_from_spec, self._observation_spec)
        action = jax.tree.map(make_from_spec, self._action_spec)

        return {
            **observation.to_dict(),
            "actions": action,
        }

    def __len__(self) -> int:
        return self._num_samples


def create_torch_dataset(
    data_config: _config.DataConfig,
    action_horizon: int,
    model_config: _model.BaseModelConfig,
    split: Literal["train", "val"] | None = None,
    val_ratio: float = 0.1,
    split_seed: int = 42,
) -> Dataset:
    """Create a dataset for training.
    
    Args:
        data_config: The data configuration.
        action_horizon: The action horizon.
        model_config: The model configuration.
        split: If "train" or "val", only load that split. If None, load all data.
        val_ratio: Ratio of validation data (default 0.1 means 10% validation).
        split_seed: Random seed for reproducible train/val splitting.
    """
    repo_id = data_config.repo_id
    if repo_id is None:
        raise ValueError("Repo ID is not set. Cannot create dataset.")
    if repo_id == "fake":
        return FakeDataset(model_config, num_samples=1024)
    
    state_history_size = getattr(data_config, 'state_history_size', 0)
    state_future_size = getattr(data_config, 'state_future_size', 0)
    state_step = getattr(data_config, 'state_step', 1)
    
    def _build_delta_timestamps(fps: float) -> dict[str, list[float]]:
        delta_ts = {
            key: [t / fps for t in range(action_horizon)] 
            for key in data_config.action_sequence_keys
        }
        if state_history_size > 0 or state_future_size > 0:
            delta_ts['state'] = [t * state_step / fps for t in range(-state_history_size, state_future_size + 1)]
        return delta_ts
    
    # Parse comma-separated repo_ids
    repo_ids = [repo_id.strip() for repo_id in repo_id.split(",") if repo_id.strip()]
    
    # If only one dataset, use the original logic for backward compatibility
    if len(repo_ids) == 1:
        dataset_meta = lerobot_dataset.LeRobotDatasetMetadata(repo_id)
        
        # Determine which episodes to load based on split
        episodes = None
        if split is not None:
            # Use metadata to get total number of episodes (no need to load full dataset)
            total_episodes = dataset_meta.total_episodes
            all_episode_indices = np.arange(total_episodes)
            
            # Shuffle with fixed seed for reproducibility
            rng = np.random.RandomState(split_seed)
            rng.shuffle(all_episode_indices)
            
            # Split into train/val
            val_size = int(total_episodes * val_ratio)
            if split == "val":
                episodes = all_episode_indices[:val_size].tolist()
                logging.info(f"Loading validation split: {len(episodes)} episodes out of {total_episodes}")
            else:  # train
                episodes = all_episode_indices[val_size:].tolist()
                logging.info(f"Loading training split: {len(episodes)} episodes out of {total_episodes}")
        
        # Use FilteredLeRobotDataset only when episodes are filtered (split is not None)
        # Otherwise use original LeRobotDataset to avoid any potential issues
        if episodes is not None:
            dataset = FilteredLeRobotDataset(
                data_config.repo_id,
                delta_timestamps=_build_delta_timestamps(dataset_meta.fps),
                episodes=episodes,
            )
        else:
            dataset = lerobot_dataset.LeRobotDataset(
                data_config.repo_id,
                delta_timestamps=_build_delta_timestamps(dataset_meta.fps),
            )

        if data_config.filter_cross_task_action_chunks or data_config.filter_issue_samples:
            dataset = LeRobotSampleFilterDataset(
                dataset,
                action_horizon=action_horizon,
                state_history_size=state_history_size,
                state_future_size=state_future_size,
                state_step=state_step,
                filter_cross_task_action_chunks=data_config.filter_cross_task_action_chunks,
                filter_issue_samples=data_config.filter_issue_samples,
            )
            logging.info(
                "Filtered %d/%d samples (task boundaries: %d, issue ranges: %d)",
                dataset.filtered_samples,
                dataset.total_samples,
                dataset.task_filtered_samples,
                dataset.issue_filtered_samples,
            )

        if data_config.prompt_from_task:
            dataset = TransformedDataset(dataset, [_transforms.PromptFromLeRobotTask(dataset_meta.tasks)])

        return dataset
    
    # Multiple datasets case
    datasets = []
    all_tasks = set()
    total_episodes_all = 0
    
    logging.info(f"Loading multiple datasets: {repo_ids}")
    
    for i, repo_id in enumerate(repo_ids):
        dataset_meta = lerobot_dataset.LeRobotDatasetMetadata(repo_id)
        all_tasks.update(dataset_meta.tasks)
        total_episodes_all += dataset_meta.total_episodes
        
        # Determine which episodes to load based on split
        episodes = None
        if split is not None:
            # Use metadata to get total number of episodes for this dataset
            total_episodes = dataset_meta.total_episodes
            all_episode_indices = np.arange(total_episodes)
            
            # Shuffle with fixed seed for reproducibility (different seed per dataset)
            rng = np.random.RandomState(split_seed + i)
            rng.shuffle(all_episode_indices)
            
            # Split into train/val
            val_size = max(1, int(total_episodes * val_ratio))
            if split == "val":
                episodes = all_episode_indices[:val_size].tolist()
                logging.info(f"Loading validation split for {repo_id}: {len(episodes)} episodes out of {total_episodes}")
            else:  # train
                episodes = all_episode_indices[val_size:].tolist()
                logging.info(f"Loading training split for {repo_id}: {len(episodes)} episodes out of {total_episodes}")
        else:
            logging.info(f"Loading full dataset for {repo_id}: {dataset_meta.total_episodes} episodes")
            
        # Create dataset
        if episodes is not None:
            dataset = FilteredLeRobotDataset(
                repo_id,
                delta_timestamps=_build_delta_timestamps(dataset_meta.fps),
                episodes=episodes,
            )
        else:
            dataset = lerobot_dataset.LeRobotDataset(
                repo_id,
                delta_timestamps=_build_delta_timestamps(dataset_meta.fps),
            )
        
        if data_config.filter_cross_task_action_chunks or data_config.filter_issue_samples:
            dataset = LeRobotSampleFilterDataset(
                dataset,
                action_horizon=action_horizon,
                state_history_size=state_history_size,
                state_future_size=state_future_size,
                state_step=state_step,
                filter_cross_task_action_chunks=data_config.filter_cross_task_action_chunks,
                filter_issue_samples=data_config.filter_issue_samples,
            )
            logging.info(
                "Filtered %d/%d samples from %s (task boundaries: %d, issue ranges: %d)",
                dataset.filtered_samples,
                dataset.total_samples,
                repo_id,
                dataset.task_filtered_samples,
                dataset.issue_filtered_samples,
            )

        if data_config.prompt_from_task:
            dataset = TransformedDataset(dataset, [_transforms.PromptFromLeRobotTask(dataset_meta.tasks)])
            
        datasets.append(dataset)
    
    # Combine all datasets
    combined_dataset = MultiDataset(datasets)
    logging.info(f"Combined {len(repo_ids)} datasets with total {len(combined_dataset)} samples")
    
    # If we have multiple datasets and prompt_from_task, log the tasks
    if data_config.prompt_from_task and len(datasets) > 1:
        logging.info(f"Combined dataset uses tasks from all datasets: {sorted(all_tasks)}")
    
    return combined_dataset


def create_rlds_dataset(
    data_config: _config.DataConfig,
    action_horizon: int,
    batch_size: int,
    *,
    shuffle: bool = False,
) -> Dataset:
    # At the moment, we only support DROID for RLDS datasets.
    return DroidRldsDataset(
        data_dir=data_config.rlds_data_dir,
        batch_size=batch_size,
        shuffle=shuffle,
        action_chunk_size=action_horizon,
        action_space=data_config.action_space,
        datasets=data_config.datasets,
    )


def transform_dataset(dataset: Dataset, data_config: _config.DataConfig, *, skip_norm_stats: bool = False) -> Dataset:
    """Transform the dataset by applying the data transforms."""
    norm_stats = {}
    if data_config.repo_id != "fake" and not skip_norm_stats:
        if data_config.norm_stats is None:
            raise ValueError(
                "Normalization stats not found. "
                "Make sure to run `scripts/compute_norm_stats.py --config-name=<your-config>`."
            )
        norm_stats = data_config.norm_stats

    return TransformedDataset(
        dataset,
        [
            *data_config.repack_transforms.inputs,
            *data_config.data_transforms.inputs,
            _transforms.Normalize(norm_stats, use_quantiles=data_config.use_quantile_norm),
            *data_config.model_transforms.inputs,
        ],
    )


def transform_iterable_dataset(
    dataset: IterableDataset,
    data_config: _config.DataConfig,
    *,
    skip_norm_stats: bool = False,
    is_batched: bool = False,
) -> IterableDataset:
    """Transform the dataset by applying the data transforms."""
    norm_stats = {}
    if data_config.repo_id != "fake" and not skip_norm_stats:
        if data_config.norm_stats is None:
            raise ValueError(
                "Normalization stats not found. "
                "Make sure to run `scripts/compute_norm_stats.py --config-name=<your-config>`."
            )
        norm_stats = data_config.norm_stats

    return IterableTransformedDataset(
        dataset,
        [
            *data_config.repack_transforms.inputs,
            *data_config.data_transforms.inputs,
            _transforms.Normalize(norm_stats, use_quantiles=data_config.use_quantile_norm),
            *data_config.model_transforms.inputs,
        ],
        is_batched=is_batched,
    )


def create_data_loader(
    config: _config.TrainConfig,
    *,
    sharding: jax.sharding.Sharding | None = None,
    shuffle: bool = False,
    num_batches: int | None = None,
    skip_norm_stats: bool = False,
    framework: Literal["jax", "pytorch"] = "jax",
    split: Literal["train", "val"] | None = None,
    val_ratio: float = 0.1,
    split_seed: int = 42,
    training: bool = True,
) -> DataLoader[tuple[_model.Observation, _model.Actions]]:
    """Create a data loader for training.

    Args:
        config: The training configuration.
        sharding: The sharding to use for the data loader (JAX only).
        shuffle: Whether to shuffle the data.
        num_batches: Determines the number of batches to return.
        skip_norm_stats: Whether to skip data normalization.
        framework: The framework to use ("jax" or "pytorch").
        split: If "train" or "val", only load that split. If None, load all data.
        val_ratio: Ratio of validation data (default 0.1 means 10% validation).
        split_seed: Random seed for reproducible train/val splitting.
        training: Whether to enable training-only data transforms such as augmentation.
    """
    data_config = (
        config.data.create_for_training(config.assets_dirs, config.model)
        if training
        else config.data.create(config.assets_dirs, config.model)
    )
    logging.info(f"data_config: {data_config}")

    if data_config.rlds_data_dir is not None:
        return create_rlds_data_loader(
            data_config,
            action_horizon=config.model.action_horizon,
            batch_size=config.batch_size,
            sharding=sharding,
            shuffle=shuffle,
            num_batches=num_batches,
            skip_norm_stats=skip_norm_stats,
            framework=framework,
        )
    return create_torch_data_loader(
        data_config,
        model_config=config.model,
        action_horizon=config.model.action_horizon,
        batch_size=config.batch_size,
        sharding=sharding,
        shuffle=shuffle,
        num_batches=num_batches,
        num_workers=config.num_workers,
        seed=config.seed,
        skip_norm_stats=skip_norm_stats,
        framework=framework,
        split=split,
        val_ratio=val_ratio,
        split_seed=split_seed,
    )


def create_torch_data_loader(
    data_config: _config.DataConfig,
    model_config: _model.BaseModelConfig,
    action_horizon: int,
    batch_size: int,
    *,
    sharding: jax.sharding.Sharding | None = None,
    skip_norm_stats: bool = False,
    shuffle: bool = False,
    num_batches: int | None = None,
    num_workers: int = 0,
    seed: int = 0,
    framework: str = "jax",
    split: Literal["train", "val"] | None = None,
    val_ratio: float = 0.1,
    split_seed: int = 42,
) -> DataLoader[tuple[_model.Observation, _model.Actions]]:
    """Create a data loader for training.

    Args:
        data_config: The data configuration.
        action_horizon: The action horizon.
        batch_size: The batch size.
        sharding: The sharding to use for the data loader. If None, the data loader will
            use a single device sharding.
        skip_norm_stats: Whether to skip data normalization.
        shuffle: Whether to shuffle the data.
        num_batches: Determines the number of batches to return. If the number exceeds the
            number of batches in the dataset, the data loader will loop over the dataset.
            If not provided, will iterate over the dataset indefinitely.
        num_workers: The number of worker processes to use. If zero, the data loader will
            execute in the main process.
        seed: The seed to use for shuffling the data.
        split: If "train" or "val", only load that split. If None, load all data.
        val_ratio: Ratio of validation data (default 0.1 means 10% validation).
        split_seed: Random seed for reproducible train/val splitting.
    """
    dataset = create_torch_dataset(data_config, action_horizon, model_config, split=split, val_ratio=val_ratio, split_seed=split_seed)
    dataset = transform_dataset(dataset, data_config, skip_norm_stats=skip_norm_stats)

    # Use TorchDataLoader for both frameworks
    # For PyTorch DDP, create DistributedSampler and divide batch size by world size
    # For JAX, divide by process count
    sampler = None
    if framework == "pytorch":
        if torch.distributed.is_initialized():
            sampler = torch.utils.data.distributed.DistributedSampler(
                dataset,
                num_replicas=torch.distributed.get_world_size(),
                rank=torch.distributed.get_rank(),
                shuffle=shuffle,
                drop_last=True,
            )
            local_batch_size = batch_size // torch.distributed.get_world_size()
        else:
            local_batch_size = batch_size
    else:
        local_batch_size = batch_size // jax.process_count()

    logging.info(f"local_batch_size: {local_batch_size}")
    data_loader = TorchDataLoader(
        dataset,
        local_batch_size=local_batch_size,
        sharding=None if framework == "pytorch" else sharding,
        shuffle=(sampler is None and shuffle),  # Don't shuffle if using sampler
        sampler=sampler,
        num_batches=num_batches,
        num_workers=num_workers,
        seed=seed,
        framework=framework,
    )

    return DataLoaderImpl(data_config, data_loader)


def create_rlds_data_loader(
    data_config: _config.DataConfig,
    action_horizon: int,
    batch_size: int,
    *,
    sharding: jax.sharding.Sharding | None = None,
    skip_norm_stats: bool = False,
    shuffle: bool = False,
    num_batches: int | None = None,
    framework: str = "jax",
) -> DataLoader[tuple[_model.Observation, _model.Actions]]:
    """Create an RLDS data loader for training.

    Note: This data loader requires some extra dependencies -- see examples/droid/README_train.md

    Args:
        data_config: The data configuration.
        action_horizon: The action horizon.
        batch_size: The batch size.
        sharding: The sharding to use for the data loader. If None, the data loader will
            use a single device sharding.
        skip_norm_stats: Whether to skip data normalization.
        shuffle: Whether to shuffle the data.
        num_batches: Determines the number of batches to return. If the number exceeds the
            number of batches in the dataset, the data loader will loop over the dataset.
            If not provided, will iterate over the dataset indefinitely.
    """
    if framework == "pytorch":
        raise NotImplementedError("PyTorch RLDS data loader is not supported yet")
    dataset = create_rlds_dataset(data_config, action_horizon, batch_size, shuffle=shuffle)
    dataset = transform_iterable_dataset(dataset, data_config, skip_norm_stats=skip_norm_stats, is_batched=True)

    data_loader = RLDSDataLoader(
        dataset,
        sharding=sharding,
        num_batches=num_batches,
    )

    return DataLoaderImpl(data_config, data_loader)


class TorchDataLoader:
    """Torch data loader implementation."""

    def __init__(
        self,
        dataset,
        local_batch_size: int,
        *,
        sharding: jax.sharding.Sharding | None = None,
        shuffle: bool = False,
        sampler: torch.utils.data.Sampler | None = None,
        num_batches: int | None = None,
        num_workers: int = 0,
        seed: int = 0,
        framework: str = "jax",
    ):
        """Create a PyTorch data loader.

        Args:
            dataset: The dataset to load.
            local_batch_size: The local batch size for each process.
            sharding: The sharding to use for the data loader.
            shuffle: Whether to shuffle the data.
            num_batches: If provided, determines the number of returned batches. If the
                number is larger than the number of batches in the dataset, the data loader
                will loop over the dataset. If not provided, will iterate over the dataset
                indefinitely.
            num_workers: The number of worker processes to use. If zero, the data loader will
                execute in the main process.
            seed: The seed to use for shuffling the data.
        """
        if jax.process_count() > 1:
            raise NotImplementedError("Data loading with multiple processes is not supported.")

        if len(dataset) < local_batch_size:
            raise ValueError(f"Local batch size ({local_batch_size}) is larger than the dataset size ({len(dataset)}).")

        # Store sharding - None for PyTorch, JAX sharding for JAX
        self._sharding = sharding
        if sharding is None and framework == "jax":
            # Use data parallel sharding by default for JAX only.
            self._sharding = jax.sharding.NamedSharding(
                jax.sharding.Mesh(jax.devices(), ("B",)),
                jax.sharding.PartitionSpec("B"),
            )
        self._num_batches = num_batches

        mp_context = None
        if num_workers > 0:
            mp_context = multiprocessing.get_context("spawn")

        generator = torch.Generator()
        generator.manual_seed(seed)
        self._data_loader = torch.utils.data.DataLoader(
            typing.cast(torch.utils.data.Dataset, dataset),
            batch_size=local_batch_size,
            shuffle=(sampler is None and shuffle),  # Don't shuffle if using sampler
            sampler=sampler,
            num_workers=num_workers,
            multiprocessing_context=mp_context,
            persistent_workers=num_workers > 0,
            collate_fn=_collate_fn,
            worker_init_fn=_worker_init_fn,
            drop_last=True,
            generator=generator,
        )

    @property
    def torch_loader(self) -> torch.utils.data.DataLoader:
        return self._data_loader

    def __iter__(self):
        num_items = 0
        while True:
            data_iter = iter(self._data_loader)
            while True:
                if self._num_batches is not None and num_items >= self._num_batches:
                    return
                try:
                    batch = next(data_iter)
                except StopIteration:
                    break  # We've exhausted the dataset. Create a new iterator and start over.
                num_items += 1
                # For JAX, convert to sharded arrays; for PyTorch, return torch tensors
                if self._sharding is not None:
                    yield jax.tree.map(lambda x: jax.make_array_from_process_local_data(self._sharding, x), batch)
                else:
                    yield jax.tree.map(torch.as_tensor, batch)


def _collate_fn(items):
    """Collate the batch elements into batched numpy arrays."""
    # Make sure to convert to numpy arrays before stacking since some of the incoming elements
    # may be JAX arrays.
    return jax.tree.map(lambda *xs: np.stack([np.asarray(x) for x in xs], axis=0), *items)


def _worker_init_fn(worker_id: int) -> None:
    """Tell JAX inside the worker process not to preallocate the GPU memory."""
    # NOTE: This is called after jax is imported inside the worker process. This
    # means that this approach will not work for selecting the backend.
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"


class RLDSDataLoader:
    """Shallow wrapper around the DROID data loader to make it compatible with openpi.

    All batching already happens in the DROID dataset, so we don't need to do anything here.
    """

    def __init__(
        self,
        dataset: DroidRldsDataset,
        *,
        sharding: jax.sharding.Sharding | None = None,
        num_batches: int | None = None,
    ):
        self._dataset = dataset
        self._num_batches = num_batches

        if jax.process_count() > 1:
            raise NotImplementedError("Data loading with multiple processes is not supported.")

        if sharding is None:
            # Use data parallel sharding by default.
            sharding = jax.sharding.NamedSharding(
                jax.sharding.Mesh(jax.devices(), ("B",)),
                jax.sharding.PartitionSpec("B"),
            )

        self._sharding = sharding
        self._num_batches = num_batches

    def __iter__(self):
        num_items = 0
        while True:
            data_iter = iter(self._dataset)
            while True:
                if self._num_batches is not None and num_items >= self._num_batches:
                    return
                try:
                    batch = next(data_iter)
                except StopIteration:
                    break  # We've exhausted the dataset. Create a new iterator and start over.
                num_items += 1
                yield jax.tree.map(lambda x: jax.make_array_from_process_local_data(self._sharding, x), batch)


class DataLoaderImpl(DataLoader):
    def __init__(self, data_config: _config.DataConfig, data_loader: TorchDataLoader | RLDSDataLoader):
        self._data_config = data_config
        self._data_loader = data_loader

    def data_config(self) -> _config.DataConfig:
        return self._data_config

    def __iter__(self):
        for batch in self._data_loader:
            yield _model.Observation.from_dict(batch), batch["actions"]
