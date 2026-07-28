from collections.abc import Iterator, Sequence
import json
import logging
import multiprocessing
import os
from pathlib import Path
import typing
from typing import Literal, Protocol, SupportsIndex, TypeVar

# JAX 生态系统
import jax                                    # [外部] JAX 核心库
import jax.numpy as jnp                       # [外部] JAX NumPy
import lerobot.common.datasets.lerobot_dataset as lerobot_dataset  # [外部] LeRobot 数据集
import numpy as np                            # [外部] NumPy
import torch                                  # [外部] PyTorch（用于 DataLoader）
import h5py                                   # [外部] HDF5 文件支持

# 项目内部导入
import openpi.models.model as _model          # [内部] 模型定义（Observation, Actions）
import openpi.training.config as _config      # [内部] 训练配置
from openpi.training.droid_rlds_dataset import DroidRldsDataset  # [内部] RLDS 数据集
import openpi.transforms as _transforms       # [内部] 数据变换

# 类型变量：协变类型参数（用于 Protocol）
T_co = TypeVar("T_co", covariant=True)


# ============================================================================
# Protocol 接口定义
# ============================================================================

class Dataset(Protocol[T_co]):
    """数据集接口（支持随机访问）。

    Protocol [标准库 typing]：结构化子类型，duck typing 的静态类型检查。
    任何实现 __getitem__ 和 __len__ 的类都自动符合此协议。

    典型实现：
        - LeRobotDataset: LeRobot 格式数据集
        - FakeDataset: 虚拟数据集（测试用）
        - MultiDataset: 多数据集组合
        - TransformedDataset: 应用变换的数据集
    """

    def __getitem__(self, index: SupportsIndex) -> T_co:
        """根据索引获取单个样本。

        Args:
            index: 样本索引（支持负数索引）

        Returns:
            单个样本数据（通常是字典格式）

        Raises:
            IndexError: 索引超出范围
        """
        raise NotImplementedError("Subclasses of Dataset should implement __getitem__.")

    def __len__(self) -> int:
        """返回数据集样本总数。

        Returns:
            样本总数
        """
        raise NotImplementedError("Subclasses of Dataset should implement __len__.")


class IterableDataset(Protocol[T_co]):
    """可迭代数据集接口（支持迭代访问）。

    与 Dataset 的区别：
        - Dataset: 支持随机访问，可以按索引获取任意样本
        - IterableDataset: 只能顺序迭代，不支持随机访问

    典型应用：
        - 数据流（无法预知总长度）
        - 实时生成的数据
        - RLDS 数据集
    """

    def __iter__(self) -> Iterator[T_co]:
        """返回数据集的迭代器。

        Returns:
            迭代器对象，逐个产生样本
        """
        raise NotImplementedError("Subclasses of IterableDataset should implement __iter__.")

    def __len__(self) -> int:
        """返回数据集样本总数（如果可确定）。

        Returns:
            样本总数（某些可迭代数据集可能无法预知）
        """
        raise NotImplementedError("Subclasses of Dataset should implement __len__.")


class DataLoader(Protocol[T_co]):
    """数据加载器接口。

    职责：
        - 批次化：将多个样本组合成一个批次
        - 迭代：提供无限循环的批次迭代
        - 配置访问：提供数据配置信息

    典型实现：
        - TorchDataLoader: 基于 PyTorch DataLoader 的包装
        - RLDSDataLoader: RLDS 数据集的专用加载器
    """

    def data_config(self) -> _config.DataConfig:
        """获取数据配置对象。

        Returns:
            数据配置对象，包含数据集信息、归一化统计量等
        """
        raise NotImplementedError("Subclasses of DataLoader should implement data_config.")

    def __iter__(self) -> Iterator[T_co]:
        """返回数据加载器的迭代器。

        Returns:
            迭代器对象，逐个产生批次数据
        """
        raise NotImplementedError("Subclasses of DataLoader should implement __iter__.")


# ============================================================================
# LeRobot 数据集扩展类
# ============================================================================

class FilteredLeRobotDataset(lerobot_dataset.LeRobotDataset):
    """修复了 LeRobot 数据集 episode 过滤 bug 的子类。

    LeRobot 的 Bug 详解：
        当使用 episodes=[131, 19, 81] 过滤数据时：
        1. episode_data_index 正确创建了 3 个条目（对应过滤后的 3 个 episode）
        2. 但 hf_dataset 的 episode_index 列保留了原始值（131, 19, 81）
        3. _get_query_indices 方法尝试访问 episode_data_index["from"][131]
        4. 结果：索引越界（只有 3 个条目，却访问索引 131）

    修复方案：
        维护一个 episode 索引映射表：
        - 原始 episode 索引 → 过滤后的连续索引
        - 示例：{131: 0, 19: 1, 81: 2}
        - 访问 episode_data_index 时使用重映射后的索引

    使用场景：
        - 训练/验证集划分（加载部分 episodes）
        - 任务特定的数据子集
    """

    def __init__(self, *args, episodes: list[int] | None = None, **kwargs):
        """初始化过滤后的 LeRobot 数据集。

        Args:
            *args, **kwargs: 传递给父类 LeRobotDataset 的参数
            episodes: 要加载的 episode 索引列表，None 表示加载全部
        """
        # 在调用父类 __init__ 之前保存 episodes 列表
        self._filtered_episodes = episodes
        # episode 索引映射表：原始索引 → 过滤后的索引
        self._episode_index_map: dict[int, int] | None = None

        # 调用父类 __init__（会创建 episode_data_index 和 hf_dataset）
        super().__init__(*args, episodes=episodes, **kwargs)

        # 在父类初始化后构建 episode 索引映射
        # 重要：基于 episode_data_index 中的实际顺序，而非排序后的顺序
        if episodes is not None and len(self.episode_data_index["from"]) > 0:
            self._episode_index_map = {}
            for new_idx in range(len(self.episode_data_index["from"])):
                # 获取此 episode 在 episode_data_index 中的第一个样本索引
                start_idx = self.episode_data_index["from"][new_idx].item()
                # 从 hf_dataset 获取原始 episode 索引
                orig_ep_idx = self.hf_dataset[start_idx]["episode_index"]
                if hasattr(orig_ep_idx, 'item'):
                    orig_ep_idx = orig_ep_idx.item()
                # 建立映射：原始索引 → 新索引
                self._episode_index_map[orig_ep_idx] = new_idx

    def _get_query_indices(self, idx: int, ep_idx: int) -> tuple[dict[str, list[int | bool]]]:
        """重写父类方法以修复 episode 索引映射 bug。

        Args:
            idx: 当前样本在全局数据集中的索引
            ep_idx: 当前样本所属的原始 episode 索引（来自 hf_dataset）

        Returns:
            (query_indices, padding) 元组
            - query_indices: 各个键的查询索引列表
            - padding: 各个键的填充掩码（标记越界的查询）

        逻辑详解：
            1. 使用映射表将原始 episode 索引转换为过滤后的索引
            2. 使用重映射后的索引访问 episode_data_index
            3. 计算查询索引时处理边界情况
        """
        # 如果使用过滤后的 episodes，重映射 episode 索引
        remapped_ep_idx = ep_idx
        if self._episode_index_map is not None:
            remapped_ep_idx = self._episode_index_map.get(ep_idx, ep_idx)

        # 使用重映射后的索引访问 episode_data_index
        ep_start = self.episode_data_index["from"][remapped_ep_idx]
        ep_end = self.episode_data_index["to"][remapped_ep_idx]

        # 构建查询索引：每个键（如 "action"）都有一个时间偏移列表
        query_indices = {
            key: [max(ep_start.item(), min(ep_end.item() - 1, idx + delta)) for delta in delta_idx]
            for key, delta_idx in self.delta_indices.items()
        }

        # 构建填充掩码：标记哪些查询超出了 episode 边界
        padding = {
            f"{key}_is_pad": torch.BoolTensor(
                [(idx + delta < ep_start.item()) | (idx + delta >= ep_end.item()) for delta in delta_idx]
            )
            for key, delta_idx in self.delta_indices.items()
        }
        return query_indices, padding

    def _query_videos(self, query_timestamps: dict[str, list[float]], ep_idx: int):
        """重写父类方法以使用原始 episode 索引查询视频文件。

        重要：这里的 ep_idx 是原始 episode 索引，不应该重映射！
        原因：视频文件使用原始 episode 索引命名（如 episode_0131.mp4）

        Args:
            query_timestamps: 查询时间戳字典
            ep_idx: 原始 episode 索引（不应重映射）

        Returns:
            查询到的视频帧数据
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


# ============================================================================
# 多数据集组合类
# ============================================================================

class MultiDataset(Dataset[T_co]):
    """组合多个 LeRobot 数据集。

    使用场景：
        - 跨任务训练（多个任务的数据集）
        - 跨机器人训练（不同机器人的数据集）
        - 跨环境训练（仿真+真实数据）

    实现原理：
        - 维护累积长度表：[len1, len1+len2, len1+len2+len3, ...]
        - 通过二分查找定位目标数据集
        - 计算目标数据集内的偏移量
    """

    def __init__(self, datasets: Sequence[Dataset[T_co]]):
        """初始化多数据集组合。

        Args:
            datasets: 要组合的数据集列表
        """
        self._datasets = list(datasets)
        # 预计算累积长度以提高索引效率
        self._cumulative_lengths = self._compute_cumulative_lengths()

    def _compute_cumulative_lengths(self) -> list[int]:
        """计算累积长度表。

        Returns:
            累积长度列表
            示例：[100, 250, 400] 表示：
                - dataset 0: 索引 0-99
                - dataset 1: 索引 100-249
                - dataset 2: 索引 250-399
        """
        cumulative_lengths = []
        total = 0
        for dataset in self._datasets:
            total += len(dataset)
            cumulative_lengths.append(total)
        return cumulative_lengths

    def __getitem__(self, index: SupportsIndex) -> T_co:
        """根据全局索引获取样本。

        Args:
            index: 全局索引（跨越所有数据集）

        Returns:
            对应数据集中的样本

        Raises:
            IndexError: 索引超出范围

        逻辑：
            1. 转换为非负索引
            2. 通过累积长度表找到目标数据集
            3. 计算目标数据集内的偏移量
            4. 返回目标数据集中的样本
        """
        # 转换为整数索引（处理负数索引）
        idx = index.__index__()
        if idx < 0:
            idx += len(self)
        if idx < 0 or idx >= len(self):
            raise IndexError(f"Index {index} is out of range for dataset of length {len(self)}")

        # 找到包含此索引的数据集
        dataset_idx = 0
        for i, cumulative_length in enumerate(self._cumulative_lengths):
            if idx < cumulative_length:
                dataset_idx = i
                break

        # 计算在目标数据集内的偏移量
        offset = idx
        if dataset_idx > 0:
            offset = idx - self._cumulative_lengths[dataset_idx - 1]

        # 返回目标数据集中的样本
        return self._datasets[dataset_idx][offset]

    def __len__(self) -> int:
        """返回所有数据集的总样本数。

        Returns:
            总样本数
        """
        if not self._cumulative_lengths:
            return 0
        return self._cumulative_lengths[-1]


# ============================================================================
# 数据变换包装类
# ============================================================================

class TransformedDataset(Dataset[T_co]):
    """应用变换的数据集包装器。

    用途：
        - 数据归一化
        - 动作类型转换（绝对动作 ↔ 相对动作）
        - 数据重打包（调整字段结构）
        - 模型特定的变换

    变换链：
        多个变换可以串联，按顺序应用：
        transforms = [transform1, transform2, transform3]
        compose(transforms) 创建组合变换函数
    """

    def __init__(self, dataset: Dataset, transforms: Sequence[_transforms.DataTransformFn]):
        """初始化变换数据集。

        Args:
            dataset: 原始数据集
            transforms: 变换函数列表（按顺序应用）
        """
        self._dataset = dataset
        # 组合多个变换为一个函数
        self._transform = _transforms.compose(transforms)

    def __getitem__(self, index: SupportsIndex) -> T_co:
        """获取样本并应用变换。

        Args:
            index: 样本索引

        Returns:
            变换后的样本
        """
        # 先从原始数据集获取样本，然后应用变换
        return self._transform(self._dataset[index])

    def __len__(self) -> int:
        """返回数据集样本数。

        Returns:
            样本总数（变换不改变数据集大小）
        """
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
    """可迭代数据集的变换包装器。

    与 TransformedDataset 的区别：
        - 支持批量数据的变换
        - 批量变换时会拆分-变换-重组
    """

    def __init__(
        self,
        dataset: IterableDataset,
        transforms: Sequence[_transforms.DataTransformFn],
        *,
        is_batched: bool = False,
    ):
        """初始化可迭代变换数据集。

        Args:
            dataset: 可迭代数据集
            transforms: 变换函数列表
            is_batched: 数据集是否已批次化
        """
        self._dataset = dataset
        self._transform = _transforms.compose(transforms)
        self._is_batched = is_batched

    def __iter__(self):
        """迭代并变换数据。

        如果数据已批次化，会拆分-变换-重组：
            1. 将批次拆分为单个样本
            2. 对每个样本应用变换
            3. 重新组合为批次

        原因：
            某些变换（如数据增强）需要独立处理每个样本
        """
        for sample in self._dataset:
            if self._is_batched:
                # 变换设计为应用于单个样本，因此需要拆分批次
                batch_size = next(v.shape[0] for v in sample.values())

                # 使用 tree_map 将批次拆分为单个样本
                individual_samples = [jax.tree.map(lambda x: x[i], sample) for i in range(batch_size)]

                # 对每个样本应用变换
                transformed = [self._transform(s) for s in individual_samples]

                # 使用 tree_map 重新组合为批次
                yield jax.tree.map(lambda *x: np.stack(x, axis=0), *transformed)
            else:
                # 非批次化数据，直接应用变换
                yield self._transform(sample)

    def __len__(self) -> int:
        """返回数据集样本数。

        Returns:
            样本总数
        """
        return len(self._dataset)


# ============================================================================
# 虚拟数据集（用于测试）
# ============================================================================

class FakeDataset(Dataset):
    """虚拟数据集（用于测试和调试）。

    特点：
        - 不依赖外部数据文件
        - 生成符合模型规范的随机数据
        - 用于快速验证模型代码

    使用场景：
        - 单元测试
        - 代码调试
        - 性能基准测试
    """

    def __init__(self, model_config: _model.BaseModelConfig, num_samples: int):
        """初始化虚拟数据集。

        Args:
            model_config: 模型配置（用于获取输入规范）
            num_samples: 样本数量
        """
        self._num_samples = num_samples
        # 从模型配置获取输入规范（形状和类型）
        self._observation_spec, self._action_spec = model_config.inputs_spec()

    def __getitem__(self, index: SupportsIndex) -> dict:
        """生成虚拟样本。

        Args:
            index: 样本索引（用作随机种子）

        Returns:
            虚拟样本字典（observation + actions）

        生成规则：
            - float32: 均匀分布 [-1.0, 1.0]
            - int32: 均匀分布 [0, 2048]
            - 其他类型: 全零
        """
        # 使用索引作为随机种子（确保可重现）
        rng = jax.random.key(index.__index__())

        def make_from_spec(spec: jax.ShapeDtypeStruct):
            """根据规范生成数据。"""
            nonlocal rng
            rng, data_rng = jax.random.split(rng)
            # 移除批次维度
            shape = spec.shape[1:]
            if spec.dtype == jnp.float32:
                # float32: 均匀分布 [-1.0, 1.0]
                return jax.random.uniform(data_rng, shape=shape, minval=-1.0, maxval=1.0)
            if spec.dtype == jnp.int32:
                # int32: 均匀分布 [0, 2048]
                return jax.random.randint(data_rng, shape=shape, minval=0, maxval=2048)
            # 其他类型: 全零
            return jnp.zeros(shape=shape, dtype=spec.dtype)

        # 生成观测和动作
        observation = jax.tree.map(make_from_spec, self._observation_spec)
        action = jax.tree.map(make_from_spec, self._action_spec)

        return {
            **observation.to_dict(),
            "actions": action,
        }

    def __len__(self) -> int:
        """返回虚拟样本总数。

        Returns:
            样本总数
        """
        return self._num_samples


# ============================================================================
# 速度去偏 HDF5 数据集
# ============================================================================

class VelocityDebiasHDF5Dataset(Dataset):
    """速度去偏 HDF5 数据集。

    特点：
        - 直接从 HDF5 文件读取去偏后的数据
        - 支持多目录扫描和文件级 train/val 划分
        - 实现随机访问和批量处理
        - 支持相机名称映射和数据格式转换

    数据格式：
        - action_chunks: (N, 30, 14) - 动作序列
        - face_images: (N, 3, 224, 224) - 正面相机
        - left_images: (N, 3, 224, 224) - 左手腕相机
        - right_images: (N, 3, 224, 224) - 右手腕相机
        - states: (N, 14) - 机器人状态
        - frame_indices: (N,) - 帧索引
    """

    def __init__(
        self,
        data_dirs: list[str],
        camera_mapping: dict[str, str] | None = None,
        split: Literal["train", "val"] = "train",
        val_ratio: float = 0.1,
        split_seed: int = 42,
    ):
        """初始化速度去偏 HDF5 数据集。

        Args:
            data_dirs: HDF5 文件目录列表
            camera_mapping: 相机名称映射（HDF5字段名 → 模型输入字段名）
            split: "train" 或 "val"
            val_ratio: 验证集比例
            split_seed: 划分随机种子
        """
        import pathlib

        self._data_dirs = data_dirs
        self._split = split
        self._val_ratio = val_ratio
        self._split_seed = split_seed

        # 默认相机映射
        self._camera_mapping = camera_mapping or {
            "face_images": "face_view",
            "left_images": "left_wrist_view",
            "right_images": "right_wrist_view",
        }

        # 发现并加载 HDF5 文件
        self._hdf5_files = self._discover_hdf5_files()

        if not self._hdf5_files:
            raise ValueError(f"No HDF5 files found in directories: {data_dirs}")

        # 构建 (file_path, sample_idx) 全局索引
        self._global_index = self._build_global_index()

        # 应用训练/验证划分
        self._global_index = self._split_data()

        logging.info(
            f"VelocityDebiasHDF5Dataset: found {len(self._hdf5_files)} files, "
            f"{len(self._global_index)} samples for {split} split"
        )

    def _discover_hdf5_files(self) -> list[str]:
        """发现所有 HDF5 文件。"""
        import pathlib

        files = []
        for data_dir in self._data_dirs:
            path = pathlib.Path(data_dir)
            if not path.exists():
                logging.warning(f"Directory not found: {data_dir}")
                continue

            # 搜索所有 .h5 文件
            for h5_file in path.glob("*.h5"):
                files.append(str(h5_file))

        return sorted(files)

    def _build_global_index(self) -> list[tuple[str, int]]:
        """构建全局索引 (file_path, sample_idx)。"""
        global_index = []

        for file_path in self._hdf5_files:
            try:
                with h5py.File(file_path, 'r') as f:
                    # 获取第一个维度的大小（样本数）
                    if 'action_chunks' in f:
                        num_samples = f['action_chunks'].shape[0]
                        for sample_idx in range(num_samples):
                            global_index.append((file_path, sample_idx))
            except Exception as e:
                logging.warning(f"Failed to read file {file_path}: {e}")

        logging.info(f"Total samples across all files: {len(global_index)}")
        return global_index

    def _split_data(self) -> list[tuple[str, int]]:
        """划分训练集和验证集。"""
        if not self._global_index:
            return []

        # 使用文件级别划分（同一文件的所有样本要么全在训练集，要么全在验证集）
        unique_files = list(set(idx[0] for idx in self._global_index))

        # 固定随机种子确保可重现
        rng = np.random.RandomState(self._split_seed)
        rng.shuffle(unique_files)

        # 划分文件
        split_idx = int(len(unique_files) * (1.0 - self._val_ratio))

        if self._split == "train":
            selected_files = set(unique_files[:split_idx])
        else:  # val
            selected_files = set(unique_files[split_idx:])

        # 过滤索引
        filtered_index = [idx for idx in self._global_index if idx[0] in selected_files]

        logging.info(
            f"Split {self._split}: {len(selected_files)} files, {len(filtered_index)} samples"
        )

        return filtered_index

    def __getitem__(self, index: SupportsIndex) -> dict:
        """获取单个样本。

        Args:
            index: 样本索引

        Returns:
            样本字典，包含：
                - images: dict of camera images
                - state: robot state
                - actions: action sequence
        """
        if index < 0 or index >= len(self):
            raise IndexError(f"Index {index} out of range [0, {len(self)})")

        file_path, sample_idx = self._global_index[index]

        try:
            with h5py.File(file_path, 'r') as f:
                sample = {}

                # 读取图像数据并应用相机映射
                images = {}
                for hdf5_key, model_key in self._camera_mapping.items():
                    if hdf5_key in f:
                        try:
                            # HDF5: (N, 3, 224, 224) → 获取单个样本
                            img_data = f[hdf5_key][sample_idx]
                            # 转换为 numpy 并确保类型正确
                            images[model_key] = np.asarray(img_data, dtype=np.float32)
                        except Exception as e:
                            # 跳过无法读取的相机（如 face_images 压缩问题）
                            logging.warning(f"Failed to read {hdf5_key}[{sample_idx}] from {file_path}: {e}")
                            continue

                sample["images"] = images

                # 读取状态数据
                if 'states' in f:
                    sample['state'] = np.asarray(
                        f['states'][sample_idx], dtype=np.float32
                    )

                # 读取动作数据
                if 'action_chunks' in f:
                    sample['actions'] = np.asarray(
                        f['action_chunks'][sample_idx], dtype=np.float32
                    )

                return sample

        except Exception as e:
            logging.error(f"Failed to load sample {index} from {file_path}: {e}")
            raise RuntimeError(f"Failed to load sample: {e}")

    def __len__(self) -> int:
        """返回数据集大小。"""
        return len(self._global_index)


# ============================================================================
# LeRobot 数据集创建函数
# ============================================================================

def create_torch_dataset(
    data_config: _config.DataConfig,
    action_horizon: int,
    model_config: _model.BaseModelConfig,
    split: Literal["train", "val"] | None = None,
    val_ratio: float = 0.1,
    split_seed: int = 42,
) -> Dataset:
    """创建 LeRobot 数据集（支持训练/验证划分）。

    Args:
        data_config: 数据配置对象
        action_horizon: 动作预测步数
        model_config: 模型配置对象
        split: "train"/"val"/None（加载训练集/验证集/全部）
        val_ratio: 验证集比例（默认 0.1 = 10%）
        split_seed: 随机种子（确保可重现的划分）

    Returns:
        数据集对象（Dataset）

    数据划分策略：
        - 按 episodes 划分（不是按 samples）
        - 使用固定种子打乱 episodes
        - 前 val_ratio 作为验证集，其余作为训练集

    多数据集支持：
        - repo_id 可以是逗号分隔的列表
        - 多个数据集会合并为 MultiDataset
        - 每个数据集独立划分训练/验证集
    """
    repo_id = data_config.repo_id

    # 优先检查 HDF5 数据集（移除对特定 repo_id 的依赖）
    hdf5_data_dirs = getattr(data_config, 'hdf5_data_dirs', None)
    if hdf5_data_dirs is not None:
        if not hdf5_data_dirs:
            raise ValueError("HDF5 data directories not specified in config")

        camera_mapping = getattr(data_config, 'hdf5_camera_mapping', None)
        hdf5_val_ratio = getattr(data_config, 'hdf5_val_ratio', 0.1)

        logging.info(f"Creating VelocityDebiasHDF5Dataset with dirs: {hdf5_data_dirs}")

        return VelocityDebiasHDF5Dataset(
            data_dirs=hdf5_data_dirs,
            camera_mapping=camera_mapping,
            split=split or "train",
            val_ratio=hdf5_val_ratio,
            split_seed=split_seed,
        )

    if repo_id is None:
        raise ValueError("Repo ID is not set. Cannot create dataset.")
    if repo_id == "fake":
        # 特殊标识：返回虚拟数据集
        return FakeDataset(model_config, num_samples=1024)

    # 获取状态序列配置（可选）
    state_history_size = getattr(data_config, 'state_history_size', 0)
    state_future_size = getattr(data_config, 'state_future_size', 0)
    state_step = getattr(data_config, 'state_step', 1)

    def _build_delta_timestamps(fps: float) -> dict[str, list[float]]:
        """构建 delta_timestamps 字典。

        Delta Timestamps:
            定义查询数据时的时间偏移量（秒）。
            例如：[0.0, 0.02, 0.04, ...] 表示查询当前及未来 2 个时间步的数据

        Args:
            fps: 数据集帧率（帧/秒）

        Returns:
            delta_timestamps 字典
        """
        # 动作序列：未来 action_horizon 步
        delta_ts = {
            key: [t / fps for t in range(action_horizon)]
            for key in data_config.action_sequence_keys
        }
        # 状态序列：历史 + 当前 + 未来（如果配置）
        if state_history_size > 0 or state_future_size > 0:
            delta_ts['state'] = [t * state_step / fps for t in range(-state_history_size, state_future_size + 1)]
        return delta_ts

    # 解析逗号分隔的 repo_ids
    repo_ids = [repo_id.strip() for repo_id in repo_id.split(",") if repo_id.strip()]

    # ======================================================================
    # 单数据集情况（向后兼容）
    # ======================================================================
    if len(repo_ids) == 1:
        # 读取数据集元数据（不需要加载完整数据集）
        dataset_meta = lerobot_dataset.LeRobotDatasetMetadata(repo_id)

        # 根据划分类型确定要加载的 episodes
        episodes = None
        if split is not None:
            # 使用元数据获取总 episode 数（无需加载完整数据集）
            total_episodes = dataset_meta.total_episodes
            all_episode_indices = np.arange(total_episodes)

            # 使用固定种子打乱（确保可重现）
            rng = np.random.RandomState(split_seed)
            rng.shuffle(all_episode_indices)

            # 划分训练/验证集
            val_size = int(total_episodes * val_ratio)
            if split == "val":
                episodes = all_episode_indices[:val_size].tolist()
                logging.info(f"Loading validation split: {len(episodes)} episodes out of {total_episodes}")
            else:  # train
                episodes = all_episode_indices[val_size:].tolist()
                logging.info(f"Loading training split: {len(episodes)} episodes out of {total_episodes}")

        # 创建数据集（使用 FilteredLeRobotDataset 仅在 episodes 被过滤时）
        # 否则使用原始 LeRobotDataset（避免潜在问题）
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

    # ======================================================================
    # 多数据集情况
    # ======================================================================
    datasets = []
    all_tasks = set()
    total_episodes_all = 0

    logging.info(f"Loading multiple datasets: {repo_ids}")

    # 为每个数据集创建独立的数据集对象
    for i, repo_id in enumerate(repo_ids):
        dataset_meta = lerobot_dataset.LeRobotDatasetMetadata(repo_id)
        all_tasks.update(dataset_meta.tasks)
        total_episodes_all += dataset_meta.total_episodes

        # 确定要加载的 episodes（与单数据集逻辑相同）
        episodes = None
        if split is not None:
            total_episodes = dataset_meta.total_episodes
            all_episode_indices = np.arange(total_episodes)

            # 使用不同的种子（避免所有数据集的打乱顺序相同）
            rng = np.random.RandomState(split_seed + i)
            rng.shuffle(all_episode_indices)

            # 划分训练/验证集（至少保留 1 个 episode）
            val_size = max(1, int(total_episodes * val_ratio))
            if split == "val":
                episodes = all_episode_indices[:val_size].tolist()
                logging.info(f"Loading validation split for {repo_id}: {len(episodes)} episodes out of {total_episodes}")
            else:  # train
                episodes = all_episode_indices[val_size:].tolist()
                logging.info(f"Loading training split for {repo_id}: {len(episodes)} episodes out of {total_episodes}")
        else:
            logging.info(f"Loading full dataset for {repo_id}: {dataset_meta.total_episodes} episodes")

        # 创建数据集
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

    # 合并所有数据集
    combined_dataset = MultiDataset(datasets)
    logging.info(f"Combined {len(repo_ids)} datasets with total {len(combined_dataset)} samples")

    # 如果有多个数据集且使用 prompt_from_task，记录任务列表
    if data_config.prompt_from_task and len(datasets) > 1:
        logging.info(f"Combined dataset uses tasks from all datasets: {sorted(all_tasks)}")

    return combined_dataset


# ============================================================================
# RLDS 数据集创建函数
# ============================================================================

def create_rlds_dataset(
    data_config: _config.DataConfig,
    action_horizon: int,
    batch_size: int,
    *,
    shuffle: bool = False,
) -> Dataset:
    """创建 RLDS 数据集（目前只支持 DROID）。

    RLDS (Reinforcement Learning Datasets):
        Google 的标准化 RL 数据集格式。
        支持多种数据集（DROID, BridgeData 等）。

    Args:
        data_config: 数据配置对象
        action_horizon: 动作预测步数
        batch_size: 批次大小
        shuffle: 是否打乱数据

    Returns:
        DroidRldsDataset 对象

    注意：
        目前只支持 DROID 数据集
        需要额外的依赖（见 examples/droid/README_train.md）
    """
    return DroidRldsDataset(
        data_dir=data_config.rlds_data_dir,
        batch_size=batch_size,
        shuffle=shuffle,
        action_chunk_size=action_horizon,
        action_space=data_config.action_space,
        datasets=data_config.datasets,
    )


# ============================================================================
# 数据变换应用函数
# ============================================================================

def transform_dataset(dataset: Dataset, data_config: _config.DataConfig, *, skip_norm_stats: bool = False) -> Dataset:
    """对数据集应用变换（适用于 Dataset）。

    Args:
        dataset: 原始数据集
        data_config: 数据配置对象
        skip_norm_stats: 是否跳过归一化统计量检查

    Returns:
        变换后的数据集

    变换顺序：
        1. repack_transforms: 重打包数据（调整字段结构）
        2. data_transforms: 数据变换（动作类型转换等）
        3. Normalize: 归一化（使用预计算的统计量）
        4. model_transforms: 模型特定变换

    Raises:
        ValueError: 如果缺少归一化统计量
    """
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
    """对可迭代数据集应用变换。

    Args:
        dataset: 原始可迭代数据集
        data_config: 数据配置对象
        skip_norm_stats: 是否跳过归一化统计量检查
        is_batched: 数据是否已批次化

    Returns:
        变换后的可迭代数据集

    与 transform_dataset 的区别：
        - 使用 IterableTransformedDataset
        - 支持批量数据的变换
    """
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


# ============================================================================
# 数据加载器工厂函数
# ============================================================================

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
    """创建数据加载器（统一入口）。

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

    # RLDS 数据集分支
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

    # LeRobot 数据集分支
    return create_torch_data_loader(
        data_config,
        model_config=config.model,
        action_horizon=int(config.model.action_horizon),
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
    """创建 PyTorch 数据加载器（用于 LeRobot 数据集）。

    Args:
        data_config: 数据配置对象
        model_config: 模型配置对象
        action_horizon: 动作预测步数
        batch_size: 批次大小
        sharding: JAX 分片策略
        skip_norm_stats: 是否跳过归一化
        shuffle: 是否打乱数据
        num_batches: 返回的批次数
        num_workers: 工作进程数（0 = 主进程）
        seed: 随机种子
        framework: "jax" 或 "pytorch"
        split: "train"/"val"/None
        val_ratio: 验证集比例
        split_seed: 随机种子

    Returns:
        DataLoader 对象

    分布式支持：
        - PyTorch DDP: 使用 DistributedSampler，batch_size 除以 world_size
        - JAX: batch_size 除以 process_count
    """
    # 创建数据集
    dataset = create_torch_dataset(data_config, int(action_horizon*data_config.scaler), model_config, split=split, val_ratio=val_ratio, split_seed=split_seed)
    # 应用数据变换
    dataset = transform_dataset(dataset, data_config, skip_norm_stats=skip_norm_stats)

    # ======================================================================
    # 分布式训练的批次大小调整
    # ======================================================================
    sampler = None
    if framework == "pytorch":
        # PyTorch 分布式式训练
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
        # JAX 训练
        local_batch_size = batch_size // jax.process_count()

    logging.info(f"local_batch_size: {local_batch_size}")

    # 创建 TorchDataLoader
    data_loader = TorchDataLoader(
        dataset,
        local_batch_size=local_batch_size,
        sharding=None if framework == "pytorch" else sharding,
        shuffle=(sampler is None and shuffle),  # 使用 sampler 时不打乱
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
    """创建 RLDS 数据加载器。

    Args:
        data_config: 数据配置对象
        action_horizon: 动作预测步数
        batch_size: 批次大小
        sharding: JAX 分片策略
        skip_norm_stats: 是否跳过归一化
        shuffle: 是否打乱数据
        num_batches: 返回的批次数
        framework: "jax" 或 "pytorch"（PyTorch 暂不支持）

    Returns:
        DataLoader 对象

    注意：
        - PyTorch RLDS 暂不支持
        - 需要额外依赖（见 examples/droid/README_train.md）
    """
    if framework == "pytorch":
        raise NotImplementedError("PyTorch RLDS data loader is not supported yet")

    # 创建 RLDS 数据集
    dataset = create_rlds_dataset(data_config, action_horizon, batch_size, shuffle=shuffle)
    # 应用变换（RLDS 数据已批次化）
    dataset = transform_iterable_dataset(dataset, data_config, skip_norm_stats=skip_norm_stats, is_batched=True)

    # 创建 RLDS 数据加载器
    data_loader = RLDSDataLoader(
        dataset,
        sharding=sharding,
        num_batches=num_batches,
    )

    return DataLoaderImpl(data_config, data_loader)


# ============================================================================
# PyTorch DataLoader 包装类
# ============================================================================

class TorchDataLoader:
    """PyTorch DataLoader 的 JAX/PyTorch 兼容包装器。

    特性：
        - 无限循环（训练时持续产生批次）
        - 自动分片（JAX 多 GPU 支持）
        - 多进程加载（加速数据预处理）
    """

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
        """初始化 PyTorch 数据加载器。

        Args:
            dataset: 数据集对象
            local_batch_size: 每个进程的批次大小
            sharding: JAX 分片策略
            shuffle: 是否打乱数据
            sampler: PyTorch 采样器（分布式训练用）
            num_batches: 返回的批次数（None = 无限）
            num_workers: 工作进程数（0 = 主进程）
            seed: 随机种子
            framework: "jax" 或 "pytorch"
        """
        if jax.process_count() > 1:
            raise NotImplementedError("Data loading with multiple processes is not supported.")

        if len(dataset) < local_batch_size:
            raise ValueError(f"Local batch size ({local_batch_size}) is larger than the dataset size ({len(dataset)}).")

        # 保存分片策略（PyTorch 为 None，JAX 为分片对象）
        self._sharding = sharding
        if sharding is None and framework == "jax":
            # JAX 默认使用数据并行分片
            self._sharding = jax.sharding.NamedSharding(
                jax.sharding.Mesh(jax.devices(), ("B",)),
                jax.sharding.PartitionSpec("B"),
            )
        self._num_batches = num_batches

        # 多进程配置
        mp_context = None
        if num_workers > 0:
            mp_context = multiprocessing.get_context("spawn")

        # 创建 PyTorch DataLoader
        generator = torch.Generator()
        generator.manual_seed(seed)
        self._data_loader = torch.utils.data.DataLoader(
            typing.cast(torch.utils.data.Dataset, dataset),
            batch_size=local_batch_size,
            shuffle=(sampler is None and shuffle),  # 使用 sampler 时不打乱
            sampler=sampler,
            num_workers=num_workers,
            multiprocessing_context=mp_context,
            persistent_workers=num_workers > 0,  # 保持工作进程活跃
            collate_fn=_collate_fn,              # 批次整理函数
            worker_init_fn=_worker_init_fn,       # 工作进程初始化函数
            drop_last=True,                       # 丢弃不完整的批次
            generator=generator,
        )

    @property
    def torch_loader(self) -> torch.utils.data.DataLoader:
        """获取底层的 PyTorch DataLoader。

        Returns:
            PyTorch DataLoader 对象
        """
        return self._data_loader

    def __iter__(self):
        """无限循环迭代批次。

        特性：
            - 无限循环（数据耗尽后重新开始）
            - 可选的批次限制（num_batches）
            - 自动转换为 JAX 分片数组或 PyTorch 张量
        """
        num_items = 0
        while True:
            data_iter = iter(self._data_loader)
            while True:
                # 检查批次限制
                if self._num_batches is not None and num_items >= self._num_batches:
                    return
                try:
                    batch = next(data_iter)
                except StopIteration:
                    # 数据耗尽，退出内层循环
                    break
                num_items += 1
                # 转换为 JAX 分片数组或 PyTorch 张量
                if self._sharding is not None:
                    # JAX: 从进程本地数据创建分片数组
                    yield jax.tree.map(lambda x: jax.make_array_from_process_local_data(self._sharding, x), batch)
                else:
                    # PyTorch: 转换为张量
                    yield jax.tree.map(torch.as_tensor, batch)


def _collate_fn(items):
    """批次整理函数：将多个样本组合成一个批次。

    Args:
        items: 样本列表（每个样本是字典）

    Returns:
        批次字典（值是 numpy 数组）

    实现：
        - 使用 jax.tree.map 遍历嵌套结构
        - 将每个字段堆叠为批次
        - 确保转换为 numpy 数组
    """
    # 确保在堆叠前转换为 numpy 数组（某些元素可能是 JAX 数组）
    return jax.tree.map(lambda *xs: np.stack([np.asarray(x) for x in xs], axis=0), *items)


def _worker_init_fn(worker_id: int) -> None:
    """工作进程初始化函数。

    告诉 JAX 在工作进程中不预分配 GPU 内存。

    Args:
        worker_id: 工作进程 ID

    注意：
        - 在 jax 导入后调用（无法选择后端）
        - 设置环境变量影响 JAX 行为
    """
    # 禁用 GPU 内存预分配
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    # 使用平台分配器
    os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"


# ============================================================================
# RLDS DataLoader 包装类
# ============================================================================

class RLDSDataLoader:
    """RLDS 数据加载器的兼容包装器。

    特点：
        - 轻量级包装（无额外批次化逻辑）
        - 批次化已在 DroidRldsDataset 中完成

    原因：
        RLDS 数据集在内部已完成批次化，
        因此这里只需要提供迭代接口和分片转换。
    """

    def __init__(
        self,
        dataset: DroidRldsDataset,
        *,
        sharding: jax.sharding.Sharding | None = None,
        num_batches: int | None = None,
    ):
        """初始化 RLDS 数据加载器。

        Args:
            dataset: DroidRldsDataset 对象
            sharding: JAX 分片策略
            num_batches: 返回的批次数
        """
        self._dataset = dataset
        self._num_batches = num_batches

        if jax.process_count() > 1:
            raise NotImplementedError("Data loading with multiple processes is not supported.")

        if sharding is None:
            # 默认使用数据并行分片
            sharding = jax.sharding.NamedSharding(
                jax.sharding.Mesh(jax.devices(), ("B",)),
                jax.sharding.PartitionSpec("B"),
            )

        self._sharding = sharding
        self._num_batches = num_batches

    def __iter__(self):
        """无限循环迭代批次。

        与 TorchDataLoader 类似：
            - 无限循环
            - 批次限制
            - 分片转换
        """
        num_items = 0
        while True:
            data_iter = iter(self._dataset)
            while True:
                if self._num_batches is not None and num_items >= self._num_batches:
                    return
                try:
                    batch = next(data_iter)
                except StopIteration:
                    break
                num_items += 1
                # 转换为 JAX 分片数组
                yield jax.tree.map(lambda x: jax.make_array_from_process_local_data(self._sharding, x), batch)


# ============================================================================
# DataLoader 实现（统一接口）
# ============================================================================

class DataLoaderImpl(DataLoader):
    """DataLoader 接口的实现。

    职责：
        - 包装底层数据加载器
        - 转换输出格式（dict → (Observation, Actions)）
        - 提供配置访问接口
    """

    def __init__(self, data_config: _config.DataConfig, data_loader: TorchDataLoader | RLDSDataLoader):
        """初始化 DataLoader 实现。

        Args:
            data_config: 数据配置对象
            data_loader: 底层数据加载器
        """
        self._data_config = data_config
        self._data_loader = data_loader

    def data_config(self) -> _config.DataConfig:
        """返回数据配置对象。

        Returns:
            数据配置对象
        """
        return self._data_config

    def __iter__(self):
        """迭代并转换输出格式。

        Yields:
            (Observation, Actions) 元组
            - Observation: 模型观测对象
            - Actions: 动作张量
        """
        for batch in self._data_loader:
            # 将字典转换为 Observation 对象
            yield _model.Observation.from_dict(batch), batch["actions"]
