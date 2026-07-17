"""Offline evaluation for X2Robot checkpoints on raw episode data."""

from __future__ import annotations

import dataclasses
import itertools
import json
import logging
from pathlib import Path
import re
from typing import Any

import cv2
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import tyro

from openpi.policies import policy_config as _policy_config
from openpi.training import checkpoint_metadata as _checkpoint_metadata


@dataclasses.dataclass
class Args:
    dataset_dir: str
    policy_dir: str
    move_steps: int = 15
    num_episodes: int | None = None
    split: str | None = None
    val_ratio: float = 0.1
    split_seed: int = 42
    output_dir: str = "offline_test_results"


@dataclasses.dataclass(frozen=True)
class EpisodeSpec:
    path: Path
    repo_id: str
    target_fps: float
    source_fps: float | None = None
    phase_boundaries: tuple[int, ...] | None = None
    phase_dim_index: int | None = None
    base_sm2sm_dim: int = 28
    phase_labels: dict[str, str] | None = None


VIDEO_FILENAMES = ("leftImg.mp4", "faceImg.mp4", "rightImg.mp4")


def source_indices_for_target_frames(
    source_frame_count: int,
    source_fps: float,
    target_fps: float,
) -> np.ndarray:
    """Map the converted dataset timeline back to raw frame indices."""
    target_frame_count = int(np.ceil(source_frame_count * target_fps / source_fps))
    target_indices = np.arange(target_frame_count, dtype=np.float64)
    source_indices = np.rint(target_indices * source_fps / target_fps).astype(np.int64)
    return np.clip(source_indices, 0, source_frame_count - 1)


def phase_ids_for_indices(frame_indices: np.ndarray, phase_boundaries: tuple[int, ...]) -> np.ndarray:
    boundaries = np.asarray(phase_boundaries, dtype=np.int64)
    phase_ids = np.searchsorted(boundaries, frame_indices, side="right") - 1
    return np.clip(phase_ids, 0, len(boundaries) - 1).astype(np.float32)


def load_episode_metadata(episode_path: Path) -> tuple[int, float]:
    episode_json_path = episode_path / f"{episode_path.name}.json"
    with episode_json_path.open("r", encoding="utf-8") as f:
        header = f.read(4096)
    total_match = re.search(r'"total"\s*:\s*(\d+)', header)
    fps_match = re.search(r'"fps"\s*:\s*([0-9.]+)', header)
    if total_match is not None and fps_match is not None:
        total_frames = int(total_match.group(1))
        source_fps = float(fps_match.group(1))
    else:
        payload = json.loads(episode_json_path.read_text(encoding="utf-8"))
        total_frames = int(payload.get("total", len(payload["data"])))
        source_fps = float(payload.get("fps", 30.0))
    if total_frames < 2:
        raise ValueError(f"episode has too few frames: {total_frames}")
    if source_fps <= 0:
        raise ValueError(f"invalid source fps: {source_fps}")
    return total_frames, source_fps


def load_phase_boundaries(
    annotation_path: Path,
    total_frames: int,
    ordered_key_frame_specs: list[list[Any]],
) -> tuple[int, ...]:
    annotation = json.loads(annotation_path.read_text(encoding="utf-8"))
    boundaries = []
    for spec in ordered_key_frame_specs:
        if not isinstance(spec, list | tuple) or len(spec) != 2:
            raise ValueError(f"invalid ordered key frame spec: {spec!r}")
        key, index = str(spec[0]), int(spec[1])
        values = annotation.get(key)
        if not isinstance(values, list) or len(values) <= index:
            raise ValueError(f"missing key frame: {key}[{index}]")
        try:
            frame_index = round(float(values[index]))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"invalid key frame: {key}[{index}]={values[index]!r}") from exc
        boundaries.append(min(max(frame_index, 0), total_frames - 1))
    if not boundaries:
        raise ValueError("ordered_key_frame_specs is empty")
    if any(left > right for left, right in itertools.pairwise(boundaries)):
        raise ValueError(f"key frames are not monotonic: {boundaries}")
    return tuple(boundaries)


def build_robot_parts(frame_data: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    slave = np.concatenate(
        [
            frame_data["follow_left_position"],
            frame_data["follow_left_rotation"],
            [frame_data["follow_left_gripper"]],
            frame_data["follow_right_position"],
            frame_data["follow_right_rotation"],
            [frame_data["follow_right_gripper"]],
        ]
    ).astype(np.float32)
    master = np.concatenate(
        [
            frame_data["master_left_position"],
            frame_data["master_left_rotation"],
            [frame_data["master_left_gripper"]],
            frame_data["master_right_position"],
            frame_data["master_right_rotation"],
            [frame_data["master_right_gripper"]],
        ]
    ).astype(np.float32)
    return slave, master


def build_state(frame_data: dict[str, Any], policy_mode: str, phase_id: float | None) -> np.ndarray:
    slave, master = build_robot_parts(frame_data)
    state = slave if policy_mode in {"s2s", "s2m"} else np.concatenate([slave, master])
    if phase_id is not None:
        state = np.concatenate([state, np.asarray([phase_id], dtype=np.float32)])
    return state


def extract_eval_prediction(
    actions: np.ndarray,
    policy_mode: str,
    slave_state_dim: int,
    output_dim: int,
    base_sm2sm_dim: int,
) -> np.ndarray:
    actions = np.asarray(actions)[:, :output_dim]
    if policy_mode in {"sm2sm", "smp2smp"}:
        robot_end = min(base_sm2sm_dim, output_dim)
        parts = [actions[:, slave_state_dim:robot_end]]
        if output_dim > base_sm2sm_dim:
            parts.append(actions[:, base_sm2sm_dim:output_dim])
        return np.concatenate(parts, axis=1)
    return actions


def build_eval_ground_truth(
    frame_data: dict[str, Any],
    policy_mode: str,
    phase_id: float | None,
) -> np.ndarray:
    slave, master = build_robot_parts(frame_data)
    robot = slave if policy_mode == "s2s" else master
    if phase_id is not None:
        robot = np.concatenate([robot, np.asarray([phase_id], dtype=np.float32)])
    return robot


def run_inference(
    policy: Any,
    episode: EpisodeSpec,
    *,
    policy_mode: str,
    state_history_size: int,
    state_future_size: int,
    state_step: int,
    slave_state_dim: int,
    output_dim: int,
    move_steps: int,
) -> tuple[np.ndarray, np.ndarray]:
    episode_name = episode.path.name
    with (episode.path / f"{episode_name}.json").open("r", encoding="utf-8") as f:
        episode_data = json.load(f)

    frames = episode_data["data"]
    source_fps = float(episode_data.get("fps", episode.source_fps or 30.0))
    videos = {
        key: cv2.VideoCapture(str(episode.path / filename))
        for key, filename in {"left": "leftImg.mp4", "face": "faceImg.mp4", "right": "rightImg.mp4"}.items()
    }
    try:
        video_frame_counts = [int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) for cap in videos.values()]
        total_source_frames = min([len(frames), *video_frame_counts])
        if total_source_frames < 2:
            raise ValueError(f"Episode {episode_name} has fewer than two synchronized frames")
        source_timeline = source_indices_for_target_frames(
            total_source_frames,
            source_fps,
            episode.target_fps,
        )
        phase_timeline = (
            phase_ids_for_indices(source_timeline, episode.phase_boundaries)
            if episode.phase_boundaries is not None
            else None
        )

        all_preds: list[np.ndarray] = []
        all_gts: list[np.ndarray] = []
        current_target_idx = 0
        while current_target_idx < len(source_timeline) - 1:
            current_source_idx = int(source_timeline[current_target_idx])
            images = {}
            for key, cap in videos.items():
                cap.set(cv2.CAP_PROP_POS_FRAMES, current_source_idx)
                success, frame = cap.read()
                if not success or frame is None:
                    raise RuntimeError(f"Failed to read {key} video frame {current_source_idx} in {episode_name}")
                image_key = f"{key}_wrist_view" if key != "face" else "face_view"
                images[image_key] = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            target_indices = np.asarray(
                [
                    np.clip(current_target_idx + offset * state_step, 0, len(source_timeline) - 1)
                    for offset in range(-state_history_size, state_future_size + 1)
                ],
                dtype=np.int64,
            )
            state_seq = np.stack(
                [
                    build_state(
                        frames[int(source_timeline[target_idx])],
                        policy_mode,
                        None if phase_timeline is None else float(phase_timeline[target_idx]),
                    )
                    for target_idx in target_indices
                ]
            )

            action_pred = policy.infer({"images": images, "prompt": "", "state": state_seq})["actions"]
            action_pred = extract_eval_prediction(
                action_pred,
                policy_mode,
                slave_state_dim,
                output_dim,
                episode.base_sm2sm_dim,
            )

            gt_length = min(move_steps, len(source_timeline) - current_target_idx - 1, len(action_pred))
            if gt_length <= 0:
                break
            gt_actions = []
            for offset in range(1, gt_length + 1):
                target_idx = current_target_idx + offset
                gt_actions.append(
                    build_eval_ground_truth(
                        frames[int(source_timeline[target_idx])],
                        policy_mode,
                        None if phase_timeline is None else float(phase_timeline[target_idx]),
                    )
                )
            all_preds.append(action_pred[:gt_length])
            all_gts.append(np.stack(gt_actions))
            current_target_idx += move_steps

        eval_dim = 14 + (1 if phase_timeline is not None else 0)
        return (
            np.concatenate(all_preds) if all_preds else np.zeros((0, eval_dim), dtype=np.float32),
            np.concatenate(all_gts) if all_gts else np.zeros((0, eval_dim), dtype=np.float32),
        )
    finally:
        for cap in videos.values():
            cap.release()


def compute_metrics(pred: np.ndarray, gt: np.ndarray, num_phases: int | None) -> dict[str, float]:
    metrics = {"robot_mae": float(np.mean(np.abs(pred[:, :14] - gt[:, :14])))}
    if pred.shape[1] > 14:
        metrics["phase_mae"] = float(np.mean(np.abs(pred[:, 14] - gt[:, 14])))
        max_phase = (num_phases - 1) if num_phases else int(np.max(gt[:, 14]))
        pred_phase = np.clip(np.rint(pred[:, 14]), 0, max_phase)
        metrics["phase_accuracy"] = float(np.mean(pred_phase == gt[:, 14]))
    return metrics


def plot_results(
    pred: np.ndarray,
    gt: np.ndarray,
    name: str,
    output_path: Path,
    target_fps: float,
) -> None:
    has_phase = pred.shape[1] > 14
    rows = 3 if has_phase else 2
    fig, axes = plt.subplots(rows, 3, figsize=(18, 5 * rows), squeeze=False)
    fig.suptitle(f"Episode: {name}", fontsize=16)
    time_axis = np.arange(len(gt)) / target_fps
    configs = [
        (0, 0, [0, 1, 2], "Left Arm XYZ", "Position"),
        (0, 1, [3, 4, 5], "Left Arm RPY", "Rotation"),
        (0, 2, [6], "Left Gripper", "Gripper"),
        (1, 0, [7, 8, 9], "Right Arm XYZ", "Position"),
        (1, 1, [10, 11, 12], "Right Arm RPY", "Rotation"),
        (1, 2, [13], "Right Gripper", "Gripper"),
    ]
    labels = {3: ["X", "Y", "Z"], 1: [""]}
    for row, col, indices, title, ylabel in configs:
        ax = axes[row, col]
        for label, idx in zip(labels[len(indices)], indices, strict=True):
            ax.plot(time_axis, gt[:, idx], "--", alpha=0.7, label=f"GT {label}")
            ax.plot(time_axis, pred[:, idx], alpha=0.7, label=f"Pred {label}")
        ax.set_title(title)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel(ylabel)
        ax.legend()
        ax.grid(visible=True, alpha=0.3)

    if has_phase:
        axes[2, 0].step(time_axis, gt[:, 14], where="post", linestyle="--", label="GT")
        axes[2, 0].plot(time_axis, pred[:, 14], alpha=0.8, label="Pred")
        axes[2, 0].set(title="Key State", xlabel="Time (s)", ylabel="Phase ID")
        axes[2, 0].legend()
        axes[2, 0].grid(visible=True, alpha=0.3)
        axes[2, 1].axis("off")
        axes[2, 2].axis("off")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def discover_episodes(
    dataset_dirs: list[Path],
    datasets_metadata: dict[str, Any],
    *,
    split: str | None,
    val_ratio: float,
    split_seed: int,
) -> list[EpisodeSpec]:
    metadata_entries = datasets_metadata.get("datasets", [])
    if not metadata_entries:
        raise ValueError("Checkpoint has no LeRobot dataset metadata")
    if len(metadata_entries) == 1:
        dataset_groups = [(metadata_entries[0], dataset_dirs)]
    elif len(metadata_entries) == len(dataset_dirs):
        dataset_groups = [(dataset, [dataset_dir]) for dataset, dataset_dir in zip(metadata_entries, dataset_dirs, strict=True)]
    else:
        raise ValueError(
            "For multi-dataset checkpoints, pass one --dataset-dir entry per metadata dataset in the same order"
        )

    specs: list[EpisodeSpec] = []
    for dataset_index, (dataset, group_dirs) in enumerate(dataset_groups):
        episode_paths = sorted(
            path
            for dataset_dir in group_dirs
            for path in dataset_dir.iterdir()
            if path.is_dir()
            and (path / f"{path.name}.json").is_file()
            and all((path / filename).is_file() for filename in VIDEO_FILENAMES)
        )
        key_state = dataset.get("key_state")
        if key_state:
            dataset_specs = []
            annotation_relative_path = Path(key_state["annotation_relative_path"])
            fallback_value = key_state.get("fallback_annotation_relative_path")
            fallback_relative_path = Path(fallback_value) if fallback_value else None
            ordered_key_frame_specs = key_state["ordered_key_frame_specs"]
            skipped_missing_annotation = 0
            skipped_invalid_annotation = 0
            for episode_path in episode_paths:
                annotation_path = episode_path / annotation_relative_path
                if not annotation_path.is_file() and fallback_relative_path is not None:
                    annotation_path = episode_path / fallback_relative_path
                if not annotation_path.is_file():
                    skipped_missing_annotation += 1
                    continue
                try:
                    total_frames, source_fps = load_episode_metadata(episode_path)
                    phase_boundaries = load_phase_boundaries(
                        annotation_path,
                        total_frames,
                        ordered_key_frame_specs,
                    )
                except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
                    skipped_invalid_annotation += 1
                    logging.warning("Skipping episode %s: %s", episode_path.name, exc)
                    continue
                dataset_specs.append(
                    EpisodeSpec(
                        path=episode_path,
                        repo_id=dataset["repo_id"],
                        target_fps=float(dataset["fps"]),
                        source_fps=source_fps,
                        phase_boundaries=phase_boundaries,
                        phase_dim_index=int(key_state["phase_dim_index"]),
                        base_sm2sm_dim=int(key_state["base_sm2sm_dim"]),
                        phase_labels=key_state.get("phase_labels"),
                    )
                )
            logging.info(
                "Dataset %s: skipped %d episodes without %s%s and %d with invalid annotations",
                dataset["repo_id"],
                skipped_missing_annotation,
                annotation_relative_path,
                f" or {fallback_relative_path}" if fallback_relative_path is not None else "",
                skipped_invalid_annotation,
            )
        else:
            dataset_specs = [
                EpisodeSpec(path=path, repo_id=dataset["repo_id"], target_fps=float(dataset["fps"]))
                for path in episode_paths
            ]

        if split:
            if split not in {"train", "val"}:
                raise ValueError(f"Invalid split {split!r}; expected 'train' or 'val'")
            rng = np.random.RandomState(split_seed + dataset_index)
            indices = np.arange(len(dataset_specs))
            rng.shuffle(indices)
            val_size = int(len(dataset_specs) * val_ratio)
            selected = indices[:val_size] if split == "val" else indices[val_size:]
            dataset_specs = [dataset_specs[index] for index in sorted(selected)]
        specs.extend(dataset_specs)
        logging.info("Dataset %s: selected %d episodes", dataset["repo_id"], len(dataset_specs))
    return specs


def main(args: Args) -> None:
    if args.move_steps <= 0:
        raise ValueError("move-steps must be positive")
    config = _checkpoint_metadata.load_train_config(args.policy_dir)
    datasets_metadata = _checkpoint_metadata.load_datasets(args.policy_dir)
    policy_mode = getattr(config.data, "mode", None)
    if policy_mode is None:
        raise ValueError("Checkpoint data config does not define an X2Robot policy mode")

    logging.info("Loading policy and config from %s", args.policy_dir)
    policy = _policy_config.create_trained_policy(config, args.policy_dir)
    episodes = discover_episodes(
        [Path(item.strip()) for item in args.dataset_dir.split(",")],
        datasets_metadata,
        split=args.split,
        val_ratio=args.val_ratio,
        split_seed=args.split_seed,
    )
    if args.num_episodes is not None:
        episodes = episodes[: args.num_episodes]
    logging.info("Evaluating %d episodes", len(episodes))

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    episode_metrics = []
    for episode in tqdm(episodes, desc="Processing"):
        pred, gt = run_inference(
            policy,
            episode,
            policy_mode=policy_mode,
            state_history_size=getattr(config.data, "state_history_size", 0),
            state_future_size=getattr(config.data, "state_future_size", 0),
            state_step=getattr(config.data, "state_step", 1),
            slave_state_dim=getattr(config.data, "slave_state_dim", 14),
            output_dim=getattr(config.data, "action_dim", config.model.action_dim),
            move_steps=args.move_steps,
        )
        if len(pred) == 0:
            logging.warning("No predictions produced for %s", episode.path.name)
            continue
        metrics = compute_metrics(pred, gt, len(episode.phase_labels) if episode.phase_labels else None)
        episode_metrics.append({"episode": episode.path.name, "frames": len(pred), **metrics})
        plot_results(pred, gt, episode.path.name, output_dir / f"{episode.path.name}.jpg", episode.target_fps)

    summary: dict[str, Any] = {"episodes": episode_metrics}
    if episode_metrics:
        metric_names = [key for key in episode_metrics[0] if key not in {"episode", "frames"}]
        summary["aggregate"] = {
            name: float(np.average([item[name] for item in episode_metrics], weights=[item["frames"] for item in episode_metrics]))
            for name in metric_names
        }
    (output_dir / "metrics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    logging.info("Done. Results saved to %s", output_dir)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main(tyro.cli(Args))
