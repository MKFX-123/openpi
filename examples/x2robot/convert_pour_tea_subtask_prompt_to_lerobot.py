#!/usr/bin/env python3
"""Convert X1Pro pour-tea data with frame-level subtask prompts to LeRobot.

The converter keeps the original 28-dimensional SM2SM state/action layout and
stores each subtask prompt as the frame's LeRobot ``task``.  It reuses the
optimized X2Robot conversion path: videos are transcoded directly and in
parallel with ffmpeg, while Parquet metadata is built without decoding or
writing temporary image frames.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import as_completed
from dataclasses import dataclass
import json
from pathlib import Path
import re
import shutil
import time
from typing import Any

from convert_x2robot_data_to_lerobot_v5 import ACTION_KEYS
from convert_x2robot_data_to_lerobot_v5 import FILE_CAMERA_MAPPING
from convert_x2robot_data_to_lerobot_v5 import STATE_KEYS
from convert_x2robot_data_to_lerobot_v5 import NoVideoIOLeRobotDataset
from convert_x2robot_data_to_lerobot_v5 import get_dim_from_keys
from convert_x2robot_data_to_lerobot_v5 import transcode_single_video
from lerobot.common.datasets.lerobot_dataset import HF_LEROBOT_HOME
import numpy as np
import tqdm
import tyro

import datasets

DEFAULT_DATASET_ROOT = Path("/mnt/public3/datasets/x1pro/pour_tea_training")
DEFAULT_ANNOTATION_RELATIVE_PATH = Path("anno/subtask_prompt.json")
DEFAULT_REPO_NAME = "pour_tea_x1pro_subtask_prompt_sm2sm_15hz"

STATE_DIM = get_dim_from_keys(STATE_KEYS)
ACTION_DIM = get_dim_from_keys(ACTION_KEYS)
if STATE_DIM != 28 or ACTION_DIM != 28:
    raise ValueError(f"Expected 28-dimensional SM2SM layout, got state={STATE_DIM}, action={ACTION_DIM}")


@dataclass(frozen=True)
class SubtaskSegment:
    segment_id: str
    start_frame: int
    end_frame: int
    prompt: str


@dataclass(frozen=True)
class EpisodeRecord:
    path: Path
    annotation_path: Path
    total_frames: int
    source_fps: float
    segments: tuple[SubtaskSegment, ...]


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def get_episode_json_path(episode_path: Path) -> Path:
    return episode_path / f"{episode_path.name}.json"


def get_episode_dirs(dataset_root: Path) -> list[Path]:
    return sorted(
        path
        for path in dataset_root.iterdir()
        if path.is_dir() and get_episode_json_path(path).is_file()
    )


def get_episode_metadata(episode_json_path: Path) -> tuple[int, float]:
    with episode_json_path.open("r", encoding="utf-8") as file:
        header = file.read(8192)
    total_match = re.search(r'"total"\s*:\s*(\d+)', header)
    fps_match = re.search(r'"fps"\s*:\s*([0-9]+(?:\.[0-9]+)?)', header)
    if total_match is not None and fps_match is not None:
        total = int(total_match.group(1))
        fps = float(fps_match.group(1))
    else:
        payload = load_json(episode_json_path)
        data = payload.get("data")
        if not isinstance(data, list):
            raise ValueError("episode JSON has no data list")
        total = int(payload.get("total", len(data)))
        fps = float(payload.get("fps", 30.0))
    if total < 2 or fps <= 0:
        raise ValueError(f"invalid episode total/fps: total={total}, fps={fps}")
    return total, fps


def load_subtask_segments(
    annotation_path: Path,
    total_frames: int,
    source_fps: float,
) -> tuple[SubtaskSegment, ...]:
    annotation = load_json(annotation_path)
    if annotation.get("annotation_unit") != "frame_index":
        raise ValueError("annotation_unit must be 'frame_index'")
    annotation_total = int(annotation.get("total_frames", total_frames))
    annotation_fps = float(annotation.get("fps", source_fps))
    if annotation_total != total_frames:
        raise ValueError(f"annotation total_frames={annotation_total}, episode total={total_frames}")
    if not np.isclose(annotation_fps, source_fps):
        raise ValueError(f"annotation fps={annotation_fps}, episode fps={source_fps}")

    raw_segments = annotation.get("segments")
    if not isinstance(raw_segments, list) or not raw_segments:
        raise ValueError("annotation has no segments")

    segments: list[SubtaskSegment] = []
    seen_ids: set[str] = set()
    for position, raw in enumerate(raw_segments):
        if not isinstance(raw, dict):
            raise ValueError(f"segment {position} is not an object")
        segment_id = raw.get("id")
        prompt = raw.get("prompt")
        if not isinstance(segment_id, str) or not segment_id:
            raise ValueError(f"segment {position} has an invalid id")
        if segment_id in seen_ids:
            raise ValueError(f"duplicate segment id: {segment_id}")
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError(f"segment {segment_id} has an empty prompt")
        try:
            start = int(raw["start_frame"])
            end = int(raw["end_frame"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"segment {segment_id} has invalid frame bounds") from exc
        if not 0 <= start < end <= total_frames:
            raise ValueError(f"segment {segment_id} bounds [{start}, {end}) outside [0, {total_frames})")
        if segments and start != segments[-1].end_frame:
            raise ValueError(
                f"segments are not contiguous: {segments[-1].segment_id} ends at "
                f"{segments[-1].end_frame}, {segment_id} starts at {start}"
            )
        segments.append(SubtaskSegment(segment_id, start, end, prompt.strip()))
        seen_ids.add(segment_id)

    if len(segments) != 9:
        raise ValueError(f"expected 9 subtask segments, got {len(segments)}")
    if segments[-1].end_frame != total_frames:
        raise ValueError(
            f"last subtask ends at {segments[-1].end_frame}, episode ends at {total_frames}"
        )
    return tuple(segments)


def discover_episodes(
    dataset_root: Path,
    annotation_relative_path: Path,
) -> tuple[list[EpisodeRecord], list[tuple[Path, str]]]:
    records: list[EpisodeRecord] = []
    skipped: list[tuple[Path, str]] = []
    for episode_path in get_episode_dirs(dataset_root):
        annotation_path = episode_path / annotation_relative_path
        if not annotation_path.is_file():
            skipped.append((episode_path, f"missing annotation: {annotation_relative_path}"))
            continue
        missing_cameras = [
            filename
            for filename in FILE_CAMERA_MAPPING.values()
            if not (episode_path / filename).is_file()
        ]
        if missing_cameras:
            skipped.append((episode_path, f"missing camera files: {', '.join(missing_cameras)}"))
            continue
        try:
            total_frames, source_fps = get_episode_metadata(get_episode_json_path(episode_path))
            segments = load_subtask_segments(annotation_path, total_frames, source_fps)
        except Exception as exc:
            skipped.append((episode_path, f"invalid annotation/episode JSON: {exc}"))
            continue
        records.append(
            EpisodeRecord(
                path=episode_path,
                annotation_path=annotation_path,
                total_frames=total_frames,
                source_fps=source_fps,
                segments=segments,
            )
        )
    return records, skipped


def source_indices_for_target_frames(
    target_frame_count: int,
    source_frame_count: int,
    source_fps: float,
    target_fps: int,
) -> np.ndarray:
    target_indices = np.arange(target_frame_count, dtype=np.float64)
    source_indices = np.rint(target_indices * source_fps / target_fps).astype(np.int64)
    return np.clip(source_indices, 0, source_frame_count - 1)


def prompts_for_source_indices(
    source_indices: np.ndarray,
    segments: tuple[SubtaskSegment, ...],
) -> np.ndarray:
    starts = np.asarray([segment.start_frame for segment in segments], dtype=np.int64)
    segment_indices = np.searchsorted(starts, source_indices, side="right") - 1
    # Some recordings contain a short unannotated prefix. Treat it as the first
    # subtask so every LeRobot frame has a valid language instruction.
    segment_indices = np.clip(segment_indices, 0, len(segments) - 1)
    prompts = np.asarray([segment.prompt for segment in segments], dtype=object)
    return prompts[segment_indices]


def load_episode_arrays_and_tasks(
    record: EpisodeRecord,
    target_frame_count: int,
    target_fps: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    payload = load_json(get_episode_json_path(record.path))
    data = payload.get("data")
    if not isinstance(data, list) or len(data) != record.total_frames:
        raise ValueError(
            f"episode data length mismatch for {record.path}: "
            f"expected {record.total_frames}, got {len(data) if isinstance(data, list) else 'missing'}"
        )
    source_indices = source_indices_for_target_frames(
        target_frame_count,
        record.total_frames,
        record.source_fps,
        target_fps,
    )

    all_keys = set(STATE_KEYS) | set(ACTION_KEYS)
    trajectories: dict[str, list[Any]] = {key: [] for key in all_keys}
    for frame_data in data:
        for key in all_keys:
            trajectories[key].append(frame_data[key])

    arrays: dict[str, np.ndarray] = {}
    for key, values in trajectories.items():
        array = np.asarray(values, dtype=np.float32)
        if "gripper" in key:
            array = array.reshape(-1, 1)
        arrays[key] = array

    states = np.concatenate([arrays[key][source_indices] for key in STATE_KEYS], axis=1)
    actions = np.concatenate([arrays[key][source_indices] for key in ACTION_KEYS], axis=1)
    tasks = prompts_for_source_indices(source_indices, record.segments)
    return states, actions, tasks


def cleanup_empty_images_dir(output_path: Path) -> None:
    images_dir = output_path / "images"
    if images_dir.is_dir():
        shutil.rmtree(images_dir)


def write_conversion_metadata(
    output_path: Path,
    dataset_root: Path,
    annotation_relative_path: Path,
    target_fps: int,
    records: list[EpisodeRecord],
) -> None:
    metadata_dir = output_path / "meta" / "subtask_prompt"
    metadata_dir.mkdir(parents=True, exist_ok=True)
    unique_prompts = list(
        dict.fromkeys(segment.prompt for record in records for segment in record.segments)
    )
    metadata = {
        "encoding": "frame_level_lerobot_task",
        "source_dataset_root": str(dataset_root),
        "annotation_relative_path": str(annotation_relative_path),
        "source_state_dim": STATE_DIM,
        "source_action_dim": ACTION_DIM,
        "target_fps": target_fps,
        "unique_prompts": unique_prompts,
        "episodes": [
            {
                "episode_index": episode_index,
                "source_path": str(record.path),
                "annotation_path": str(record.annotation_path),
                "total_frames": record.total_frames,
                "source_fps": record.source_fps,
                "segments": [
                    {
                        "id": segment.segment_id,
                        "start_frame": segment.start_frame,
                        "end_frame": segment.end_frame,
                        "prompt": segment.prompt,
                    }
                    for segment in record.segments
                ],
            }
            for episode_index, record in enumerate(records)
        ],
    }
    with (metadata_dir / "conversion.json").open("w", encoding="utf-8") as file:
        json.dump(metadata, file, ensure_ascii=False, indent=2)
        file.write("\n")


def main(
    dataset_root: Path = DEFAULT_DATASET_ROOT,
    repo_name: str = DEFAULT_REPO_NAME,
    annotation_relative_path: Path = DEFAULT_ANNOTATION_RELATIVE_PATH,
    *,
    push_to_hub: bool = False,
    private: bool = False,
    debug: bool = False,
    debug_episodes: int = 3,
    low_resolution: bool = True,
    num_workers: int = 10,
    target_fps: int = 15,
    video_codec: str = "h264",
    overwrite: bool = False,
) -> None:
    datasets.disable_progress_bars()
    dataset_root = dataset_root.resolve()
    if not dataset_root.is_dir():
        raise FileNotFoundError(f"Dataset root does not exist: {dataset_root}")
    if annotation_relative_path.is_absolute() or ".." in annotation_relative_path.parts:
        raise ValueError("annotation_relative_path must be a safe relative path")
    if video_codec not in {"av1", "h264"}:
        raise ValueError(f"Unsupported video codec: {video_codec}")
    if target_fps <= 0 or debug_episodes <= 0 or num_workers <= 0:
        raise ValueError("target_fps, debug_episodes, and num_workers must be positive")

    output_path = HF_LEROBOT_HOME / repo_name
    print(f"HF_LEROBOT_HOME: {HF_LEROBOT_HOME}")
    print(f"Dataset root: {dataset_root}")
    print(f"Subtask annotation: {annotation_relative_path}")
    print(f"Output: {output_path}")
    print(f"Target fps: {target_fps}; codec: {video_codec}; workers: {num_workers}")
    print(f"SM2SM dimensions: state={STATE_DIM}, action={ACTION_DIM}")

    if output_path.exists():
        if not overwrite:
            raise FileExistsError(f"{output_path} already exists. Pass --overwrite to replace it.")
        print(f"Removing existing output dataset: {output_path}")
        shutil.rmtree(output_path)

    records, skipped_before_transcode = discover_episodes(dataset_root, annotation_relative_path)
    total_valid_records = len(records)
    if debug:
        records = records[:debug_episodes]
        print(f"Debug mode: processing {len(records)} episodes")
    if not records:
        raise RuntimeError(f"No valid subtask-annotated episodes found under {dataset_root}")
    print(f"Valid annotated episodes: {total_valid_records}; selected: {len(records)}")
    if skipped_before_transcode:
        print(f"Skipped before transcode: {len(skipped_before_transcode)}")
        for path, reason in skipped_before_transcode[:20]:
            print(f"  - {path.name}: {reason}")
        if len(skipped_before_transcode) > 20:
            print(f"  ... and {len(skipped_before_transcode) - 20} more")

    target_size = (320, 240) if low_resolution else (640, 480)
    image_shape = (target_size[1], target_size[0], 3)
    dataset = NoVideoIOLeRobotDataset.create(
        repo_id=repo_name,
        robot_type="ARX",
        fps=target_fps,
        root=output_path,
        features={
            "face_view": {
                "dtype": "video",
                "shape": image_shape,
                "names": ["height", "width", "channel"],
            },
            "left_wrist_view": {
                "dtype": "video",
                "shape": image_shape,
                "names": ["height", "width", "channel"],
            },
            "right_wrist_view": {
                "dtype": "video",
                "shape": image_shape,
                "names": ["height", "width", "channel"],
            },
            "state": {"dtype": "float32", "shape": (STATE_DIM,), "names": ["state"]},
            "actions": {"dtype": "float32", "shape": (ACTION_DIM,), "names": ["actions"]},
        },
        image_writer_threads=0,
        image_writer_processes=0,
    )
    dataset._skip_all_media = True  # noqa: SLF001

    total_start = time.time()
    transcode_start = time.time()
    transcode_tasks = [
        (
            str(record.path),
            episode_index,
            camera_name,
            video_filename,
            output_path,
            target_size,
            target_fps,
            video_codec,
        )
        for episode_index, record in enumerate(records)
        for camera_name, video_filename in FILE_CAMERA_MAPPING.items()
    ]
    episode_frame_counts: dict[int, dict[str, int]] = {}
    episode_failures: dict[int, list[str]] = {}

    print("Transcoding videos directly with ffmpeg...")
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = {
            executor.submit(transcode_single_video, *task): task[:3]
            for task in transcode_tasks
        }
        with tqdm.tqdm(total=len(transcode_tasks), desc="Transcoding videos") as progress:
            for future in as_completed(futures):
                try:
                    episode_index, camera_name, frame_count = future.result()
                    episode_frame_counts.setdefault(episode_index, {})[camera_name] = frame_count
                except Exception as exc:
                    _episode_path, episode_index, camera_name = futures[future]
                    summary = "\n".join(str(exc).splitlines()[-8:])
                    episode_failures.setdefault(episode_index, []).append(
                        f"{camera_name} transcode failed: {summary}"
                    )
                progress.update(1)
    transcode_seconds = time.time() - transcode_start

    successful_indices = [
        index for index in range(len(records)) if index not in episode_failures
    ]
    failed_indices = sorted(episode_failures)
    for episode_index in failed_indices:
        for camera_name in FILE_CAMERA_MAPPING:
            video_path = (
                output_path
                / "videos"
                / "chunk-000"
                / camera_name
                / f"episode_{episode_index:06d}.mp4"
            )
            if video_path.exists():
                video_path.unlink()

    for new_index, old_index in enumerate(successful_indices):
        if new_index == old_index:
            continue
        for camera_name in FILE_CAMERA_MAPPING:
            source = (
                output_path / "videos" / "chunk-000" / camera_name / f"episode_{old_index:06d}.mp4"
            )
            destination = (
                output_path / "videos" / "chunk-000" / camera_name / f"episode_{new_index:06d}.mp4"
            )
            if source.exists():
                if destination.exists():
                    destination.unlink()
                source.rename(destination)

    processed_records = [records[index] for index in successful_indices]
    processed_frame_counts = [
        min(episode_frame_counts[index].values()) for index in successful_indices
    ]
    if failed_indices:
        print(f"Skipped after transcode: {len(failed_indices)}")
        for index in failed_indices[:20]:
            print(f"  - {records[index].path.name}")
            for reason in episode_failures[index]:
                print(f"      {reason}")
    if not processed_records:
        raise RuntimeError("No episodes remained after video transcoding")

    build_start = time.time()
    dummy_image = np.zeros(image_shape, dtype=np.uint8)
    print("Building frame metadata and per-frame tasks...")
    for record, video_frame_count in tqdm.tqdm(
        zip(processed_records, processed_frame_counts, strict=True),
        total=len(processed_records),
        desc="Building dataset",
    ):
        states, actions, tasks = load_episode_arrays_and_tasks(
            record,
            target_frame_count=video_frame_count,
            target_fps=target_fps,
        )
        num_frames = len(states)
        dataset._video_frame_count = num_frames - 1  # noqa: SLF001
        for frame_index in range(num_frames - 1):
            dataset.add_frame(
                {
                    "face_view": dummy_image,
                    "left_wrist_view": dummy_image,
                    "right_wrist_view": dummy_image,
                    "state": states[frame_index],
                    "actions": actions[frame_index + 1],
                    "task": str(tasks[frame_index]),
                }
            )
        dataset.save_episode()
    build_seconds = time.time() - build_start

    print("Finalizing video metadata...")
    dataset.finalize_video_info()
    cleanup_empty_images_dir(output_path)
    write_conversion_metadata(
        output_path,
        dataset_root,
        annotation_relative_path,
        target_fps,
        processed_records,
    )

    total_seconds = time.time() - total_start
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Valid annotated episodes: {total_valid_records}")
    print(f"Episodes selected:        {len(records)}")
    print(f"Episodes processed:       {len(processed_records)}")
    print(f"Skipped before transcode: {len(skipped_before_transcode)}")
    print(f"Skipped after transcode:  {len(failed_indices)}")
    print(f"Transcode time:           {transcode_seconds:.2f}s")
    print(f"Metadata build time:      {build_seconds:.2f}s")
    print(f"Total time:               {total_seconds:.2f}s")
    print(f"Dataset saved at:         {output_path}")
    print("=" * 60)

    if push_to_hub:
        dataset.push_to_hub(private=private)


if __name__ == "__main__":
    tyro.cli(main)
