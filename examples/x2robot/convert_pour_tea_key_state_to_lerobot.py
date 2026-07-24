#!/usr/bin/env python3
"""Convert X1Pro pour-tea key-state annotations to a LeRobot dataset.

This keeps the original SM2SM robot state/action layout and appends one scalar
phase dimension. The appended value is a label id in [0, 7], derived from eight
ordered key frames in ``anno/subtask_gemini+heuristic.json``.
"""

from __future__ import annotations

import json
import re
import shutil
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import datasets
import numpy as np
import tqdm
import tyro
from lerobot.common.datasets.lerobot_dataset import HF_LEROBOT_HOME

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from examples.x2robot.convert_x2robot_data_to_lerobot_v5 import (
    ACTION_KEYS,
    FILE_CAMERA_MAPPING,
    STATE_KEYS,
    NoVideoIOLeRobotDataset,
    get_dim_from_keys,
    transcode_video_ffmpeg,
)


DEFAULT_DATASET_ROOT = Path("/mnt/public3/datasets/x1pro/pour_tea_training")
DEFAULT_ANNOTATION_RELATIVE_PATH = Path("anno/subtask_gemini+heuristic.json")
DEFAULT_PROMPT_RELATIVE_PATH = Path("anno/prompt.txt")
DEFAULT_ISSUE_ANNOTATION_RELATIVE_PATH = Path("anno/pour_tea_issues_auto.json")
DEFAULT_REPO_NAME = "pour_tea_x1pro_key_state_sm2sm"

ORDERED_KEY_FRAME_SPECS = [
    ("0", 0),
    ("1", 0),
    ("2", 1),
    ("3", 0),
    ("4", 0),
    ("4", 1),
    ("4", 2),
    ("5", 0),
]

PHASE_LABELS = [
    "start_remove_lid",
    "start_pour_water",
    "stop_pour_water",
    "start_replace_lid",
    "start_first_cup",
    "finish_first_cup",
    "finish_second_cup",
    "finish_third_cup_and_reset",
]

BASE_STATE_DIM = get_dim_from_keys(STATE_KEYS)
BASE_ACTION_DIM = get_dim_from_keys(ACTION_KEYS)
if BASE_STATE_DIM != BASE_ACTION_DIM:
    raise ValueError(f"SM2SM state/action dims differ: {BASE_STATE_DIM} vs {BASE_ACTION_DIM}")

PHASE_DIM_INDEX = BASE_STATE_DIM
AUGMENTED_DIM = BASE_STATE_DIM + 1


@dataclass(frozen=True)
class EpisodeRecord:
    path: Path
    key_frame_path: Path
    issue_ranges: list[tuple[int, int]]
    prompt: str
    phase_boundaries: list[int]
    total_frames: int
    source_fps: float


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def get_episode_json_path(episode_path: Path) -> Path:
    return episode_path / f"{episode_path.name}.json"


def get_episode_dirs(dataset_root: Path) -> list[Path]:
    return sorted(
        path
        for path in dataset_root.iterdir()
        if path.is_dir() and get_episode_json_path(path).is_file()
    )


def get_episode_metadata(episode_json_path: Path) -> tuple[int, float]:
    with episode_json_path.open("r", encoding="utf-8") as f:
        header = f.read(4096)
    total_match = re.search(r'"total"\s*:\s*(\d+)', header)
    fps_match = re.search(r'"fps"\s*:\s*([0-9.]+)', header)
    if total_match is not None and fps_match is not None:
        total = int(total_match.group(1))
        fps = float(fps_match.group(1))
    else:
        payload = load_json(episode_json_path)
        total = payload.get("total")
        if total is None:
            total = len(payload["data"])
        total = int(total)
        fps = float(payload.get("fps", 30.0))
    if total < 2:
        raise ValueError(f"episode has too few frames: {total}")
    if fps <= 0:
        raise ValueError(f"invalid fps: {fps}")
    return total, fps


def load_phase_boundaries(key_frame_path: Path, total_frames: int) -> list[int]:
    key_frames = load_json(key_frame_path)
    boundaries: list[int] = []
    missing: list[str] = []

    for key, index in ORDERED_KEY_FRAME_SPECS:
        values = key_frames.get(key)
        if not isinstance(values, list) or len(values) <= index:
            missing.append(f"{key}[{index}]")
            continue
        try:
            frame_idx = int(round(float(values[index])))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"invalid key frame {key}[{index}]={values[index]!r}") from exc
        boundaries.append(min(max(frame_idx, 0), total_frames - 1))

    if missing:
        raise ValueError(f"missing key frames: {', '.join(missing)}")
    if any(boundaries[i] > boundaries[i + 1] for i in range(len(boundaries) - 1)):
        raise ValueError(f"key frames are not monotonic: {boundaries}")

    return boundaries


def phase_ids_for_indices(frame_indices: np.ndarray, phase_boundaries: list[int]) -> np.ndarray:
    boundaries = np.asarray(phase_boundaries, dtype=np.int64)
    phase_ids = np.searchsorted(boundaries, frame_indices, side="right") - 1
    return np.clip(phase_ids, 0, len(phase_boundaries) - 1).astype(np.float32)


def load_issue_ranges(
    issue_annotation_path: Path,
    issue_label: str,
    total_frames: int,
) -> list[tuple[int, int]]:
    if not issue_annotation_path.is_file():
        return []
    annotation = load_json(issue_annotation_path)
    values = annotation.get(issue_label, [])
    if not isinstance(values, list) or len(values) % 2 != 0:
        raise ValueError(f"issue label {issue_label!r} must contain start/end pairs")

    ranges = []
    for index in range(0, len(values), 2):
        start = min(max(int(round(float(values[index]))), 0), total_frames)
        end = min(max(int(round(float(values[index + 1]))), 0), total_frames)
        if start < end:
            ranges.append((start, end))
    return ranges


def target_issue_ranges(
    record: EpisodeRecord,
    target_frame_count: int,
    target_fps: int,
) -> list[tuple[int, int]]:
    """Map raw issue intervals to LeRobot rows, including each row's next-frame action."""
    if not record.issue_ranges or target_frame_count < 2:
        return []

    source_indices = source_indices_for_target_frames(
        target_frame_count,
        record.total_frames,
        record.source_fps,
        target_fps,
    )
    bad_source_frames = np.zeros(record.total_frames, dtype=bool)
    for start, end in record.issue_ranges:
        bad_source_frames[start:end] = True

    bad_rows = bad_source_frames[source_indices[:-1]] | bad_source_frames[source_indices[1:]]
    changes = np.flatnonzero(bad_rows[1:] != bad_rows[:-1]) + 1
    ranges: list[tuple[int, int]] = []
    run_start = 0
    for run_end in [*changes.tolist(), len(bad_rows)]:
        if bad_rows[run_start]:
            ranges.append((run_start, run_end))
        run_start = run_end
    return ranges


def transcode_single_video_with_codec(
    episode_path: str,
    episode_index: int,
    camera_name: str,
    video_filename: str,
    output_root: Path,
    target_size: tuple[int, int],
    fps: int,
    video_codec: str,
) -> tuple[int, str, int]:
    video_path = Path(episode_path) / video_filename
    output_path = output_root / "videos" / "chunk-000" / camera_name / f"episode_{episode_index:06d}.mp4"

    if video_codec == "av1":
        vcodec = "libsvtav1"
        crf = 30
    elif video_codec == "h264":
        vcodec = "libx264"
        crf = 23
    else:
        raise ValueError(f"Unsupported video codec: {video_codec}. Expected 'av1' or 'h264'.")

    num_frames = transcode_video_ffmpeg(
        str(video_path),
        output_path,
        target_size,
        fps,
        vcodec=vcodec,
        pix_fmt="yuv420p",
        g=2,
        crf=crf,
    )
    return episode_index, camera_name, num_frames


def source_indices_for_target_frames(
    target_frame_count: int,
    source_frame_count: int,
    source_fps: float,
    target_fps: int,
) -> np.ndarray:
    target_indices = np.arange(target_frame_count, dtype=np.float64)
    source_indices = np.rint(target_indices * source_fps / target_fps).astype(np.int64)
    return np.clip(source_indices, 0, source_frame_count - 1)


def load_json_data_with_phase(
    episode_path: Path,
    phase_boundaries: list[int],
    target_frame_count: int,
    target_fps: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Load robot arrays and append scalar phase id to state/action."""
    payload = load_json(get_episode_json_path(episode_path))
    data = payload["data"]
    source_fps = float(payload.get("fps", 30.0))
    source_indices = source_indices_for_target_frames(target_frame_count, len(data), source_fps, target_fps)
    phases = phase_ids_for_indices(source_indices, phase_boundaries)

    all_keys = set(STATE_KEYS) | set(ACTION_KEYS)
    trajectories: dict[str, list[Any]] = {key: [] for key in all_keys}
    for frame_data in data:
        for key in all_keys:
            trajectories[key].append(frame_data[key])

    arrays = {}
    for key, values in trajectories.items():
        arr = np.asarray(values, dtype=np.float32)
        if "gripper" in key:
            arr = arr.reshape(-1, 1)
        arrays[key] = arr

    phase_column = phases.reshape(-1, 1)
    state_array = np.concatenate([*[arrays[key][source_indices] for key in STATE_KEYS], phase_column], axis=1)
    action_array = np.concatenate([*[arrays[key][source_indices] for key in ACTION_KEYS], phase_column], axis=1)
    return state_array, action_array


def discover_episodes(
    dataset_root: Path,
    annotation_relative_path: Path,
    prompt_relative_path: Path,
    issue_annotation_relative_path: Path | None,
    issue_label: str,
    fallback_annotation_relative_path: Path | None = None,
) -> tuple[list[EpisodeRecord], list[tuple[Path, str]]]:
    episode_records: list[EpisodeRecord] = []
    skipped: list[tuple[Path, str]] = []

    for episode_path in get_episode_dirs(dataset_root):
        prompt_path = episode_path / prompt_relative_path
        primary_key_frame_path = episode_path / annotation_relative_path
        fallback_key_frame_path = (
            episode_path / fallback_annotation_relative_path
            if fallback_annotation_relative_path is not None
            else None
        )
        key_frame_path = primary_key_frame_path
        if not key_frame_path.is_file() and fallback_key_frame_path is not None:
            key_frame_path = fallback_key_frame_path

        missing_cameras = [
            video_filename
            for video_filename in FILE_CAMERA_MAPPING.values()
            if not (episode_path / video_filename).is_file()
        ]
        if missing_cameras:
            skipped.append((episode_path, f"missing camera files: {', '.join(missing_cameras)}"))
            continue
        if not key_frame_path.is_file():
            annotation_candidates = [str(annotation_relative_path)]
            if fallback_annotation_relative_path is not None:
                annotation_candidates.append(str(fallback_annotation_relative_path))
            skipped.append((episode_path, f"missing annotation: {', '.join(annotation_candidates)}"))
            continue
        if not prompt_path.is_file():
            skipped.append((episode_path, f"missing prompt: {prompt_relative_path}"))
            continue

        try:
            total_frames, source_fps = get_episode_metadata(get_episode_json_path(episode_path))
            phase_boundaries = load_phase_boundaries(key_frame_path, total_frames)
            issue_ranges = (
                load_issue_ranges(
                    episode_path / issue_annotation_relative_path,
                    issue_label,
                    total_frames,
                )
                if issue_annotation_relative_path is not None
                else []
            )
            prompt = prompt_path.read_text(encoding="utf-8").strip()
            if not prompt:
                raise ValueError("prompt is empty")
        except Exception as exc:
            skipped.append((episode_path, f"invalid annotation/json/prompt: {exc}"))
            continue

        episode_records.append(
            EpisodeRecord(
                path=episode_path,
                key_frame_path=key_frame_path,
                issue_ranges=issue_ranges,
                prompt=prompt,
                phase_boundaries=phase_boundaries,
                total_frames=total_frames,
                source_fps=source_fps,
            )
        )

    return episode_records, skipped


def write_phase_metadata(
    output_path: Path,
    dataset_root: Path,
    annotation_relative_path: Path,
    prompt_relative_path: Path,
    fallback_annotation_relative_path: Path | None,
    target_fps: int,
    records: list[EpisodeRecord],
) -> None:
    metadata_dir = output_path / "meta" / "key_state"
    metadata_dir.mkdir(parents=True, exist_ok=True)
    metadata = {
        "encoding": "scalar_label_id",
        "phase_dim_index": PHASE_DIM_INDEX,
        "base_sm2sm_dim": BASE_STATE_DIM,
        "augmented_dim": AUGMENTED_DIM,
        "source_dataset_root": str(dataset_root),
        "annotation_relative_path": str(annotation_relative_path),
        "prompt_relative_path": str(prompt_relative_path),
        "fallback_annotation_relative_path": (
            str(fallback_annotation_relative_path) if fallback_annotation_relative_path is not None else None
        ),
        "target_fps": target_fps,
        "ordered_key_frame_specs": ORDERED_KEY_FRAME_SPECS,
        "phase_labels": {str(i): label for i, label in enumerate(PHASE_LABELS)},
        "episodes": [
            {
                "episode_index": idx,
                "source_path": str(record.path),
                "annotation_path": str(record.key_frame_path),
                "prompt": record.prompt,
                "total_frames": record.total_frames,
                "source_fps": record.source_fps,
                "phase_boundaries": record.phase_boundaries,
                "source_issue_ranges": record.issue_ranges,
            }
            for idx, record in enumerate(records)
        ],
    }
    with (metadata_dir / "phase_layout.json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)


def write_issue_metadata(
    output_path: Path,
    issue_annotation_relative_path: Path | None,
    issue_label: str,
    target_fps: int,
    records: list[EpisodeRecord],
    video_frame_counts: list[int],
) -> None:
    metadata_dir = output_path / "meta" / "key_state"
    metadata_dir.mkdir(parents=True, exist_ok=True)
    episodes = []
    for episode_index, (record, frame_count) in enumerate(
        zip(records, video_frame_counts, strict=True)
    ):
        episodes.append(
            {
                "episode_index": episode_index,
                "source_episode": record.path.name,
                "source_ranges": record.issue_ranges,
                "sample_ranges": target_issue_ranges(record, frame_count, target_fps),
            }
        )

    metadata = {
        "format_version": 1,
        "annotation_relative_path": (
            str(issue_annotation_relative_path)
            if issue_annotation_relative_path is not None
            else None
        ),
        "issue_label": issue_label,
        "target_fps": target_fps,
        "range_semantics": "half_open",
        "episodes": episodes,
    }
    with (metadata_dir / "excluded_sample_ranges.json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)


def cleanup_empty_images_dir(output_path: Path) -> None:
    img_dir = output_path / "images"
    if img_dir.is_dir():
        shutil.rmtree(img_dir)


def main(
    dataset_root: Path = DEFAULT_DATASET_ROOT,
    repo_name: str = DEFAULT_REPO_NAME,
    annotation_relative_path: Path = DEFAULT_ANNOTATION_RELATIVE_PATH,
    prompt_relative_path: Path = DEFAULT_PROMPT_RELATIVE_PATH,
    issue_annotation_relative_path: Path | None = DEFAULT_ISSUE_ANNOTATION_RELATIVE_PATH,
    issue_label: str = "5",
    fallback_annotation_relative_path: Path | None = None,
    *,
    push_to_hub: bool = False,
    private: bool = False,
    debug: bool = False,
    debug_episodes: int = 3,
    low_resolution: bool = True,
    num_workers: int = 10,
    target_fps: int = 20,
    video_codec: str = "h264",
    overwrite: bool = False,
) -> None:
    datasets.disable_progress_bars()
    if video_codec not in {"av1", "h264"}:
        raise ValueError(f"Unsupported video codec: {video_codec}. Expected 'av1' or 'h264'.")
    if target_fps <= 0:
        raise ValueError(f"target_fps must be positive, got {target_fps}")

    print(f"HF_LEROBOT_HOME: {HF_LEROBOT_HOME}")
    print(f"Dataset root: {dataset_root}")
    print(f"Annotation: {annotation_relative_path}")
    print(f"Prompt: {prompt_relative_path}")
    print(f"Issue annotation: {issue_annotation_relative_path}")
    print(f"Issue label: {issue_label}")
    if fallback_annotation_relative_path is not None:
        print(f"Fallback annotation: {fallback_annotation_relative_path}")
    print(f"Video codec: {video_codec}")
    print(f"Target fps: {target_fps}")
    print(f"Base SM2SM dim: {BASE_STATE_DIM}; phase dim index: {PHASE_DIM_INDEX}; total dim: {AUGMENTED_DIM}")

    output_path = HF_LEROBOT_HOME / repo_name
    if output_path.exists():
        if not overwrite:
            raise FileExistsError(f"{output_path} already exists. Pass --overwrite to replace it.")
        print(f"Removing existing dataset at {output_path}")
        shutil.rmtree(output_path)

    records, skipped_before_transcode = discover_episodes(
        dataset_root,
        annotation_relative_path,
        prompt_relative_path,
        issue_annotation_relative_path,
        issue_label,
        fallback_annotation_relative_path,
    )
    total_valid_records = len(records)
    if debug:
        records = records[:debug_episodes]
        print(f"Debug mode: only processing first {len(records)} episodes")
    if not records:
        raise RuntimeError(f"No valid episodes found under {dataset_root}")

    print(f"Found {total_valid_records} valid episodes")
    print(f"Processing {len(records)} episodes")
    if skipped_before_transcode:
        print(f"Skipped {len(skipped_before_transcode)} episodes before transcode")
        for path, reason in skipped_before_transcode[:20]:
            print(f"  - {path}: {reason}")
        if len(skipped_before_transcode) > 20:
            print(f"  ... {len(skipped_before_transcode) - 20} more")

    target_size = (320, 240) if low_resolution else (640, 480)
    shape = (target_size[1], target_size[0], 3)

    dataset = NoVideoIOLeRobotDataset.create(
        repo_id=repo_name,
        robot_type="ARX",
        fps=target_fps,
        root=output_path,
        features={
            "face_view": {
                "dtype": "video",
                "shape": shape,
                "names": ["height", "width", "channel"],
            },
            "left_wrist_view": {
                "dtype": "video",
                "shape": shape,
                "names": ["height", "width", "channel"],
            },
            "right_wrist_view": {
                "dtype": "video",
                "shape": shape,
                "names": ["height", "width", "channel"],
            },
            "state": {
                "dtype": "float32",
                "shape": (AUGMENTED_DIM,),
                "names": ["state"],
            },
            "actions": {
                "dtype": "float32",
                "shape": (AUGMENTED_DIM,),
                "names": ["actions"],
            },
        },
        image_writer_threads=0,
        image_writer_processes=0,
    )
    dataset._skip_all_media = True

    total_start = time.time()
    transcode_start = time.time()
    transcode_tasks = [
        (str(record.path), ep_idx, camera_name, video_filename, output_path, target_size, target_fps)
        for ep_idx, record in enumerate(records)
        for camera_name, video_filename in FILE_CAMERA_MAPPING.items()
    ]
    episode_frame_counts: dict[int, dict[str, int]] = {}
    episode_failures: dict[int, list[str]] = {}

    print("Transcoding videos...")
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = {
            executor.submit(transcode_single_video_with_codec, *task, video_codec): task[:3]
            for task in transcode_tasks
        }
        with tqdm.tqdm(total=len(transcode_tasks), desc="Transcoding videos") as pbar:
            for future in as_completed(futures):
                try:
                    ep_idx, camera_name, num_frames = future.result()
                    episode_frame_counts.setdefault(ep_idx, {})[camera_name] = num_frames
                except Exception as exc:
                    _ep_path, ep_idx, camera_name = futures[future]
                    error_summary = "\n".join(str(exc).splitlines()[-8:])
                    episode_failures.setdefault(ep_idx, []).append(
                        f"{camera_name} transcode failed: {error_summary}"
                    )
                pbar.update(1)

    transcode_seconds = time.time() - transcode_start
    print(f"Transcoding completed in {transcode_seconds:.2f}s")

    skipped_episode_indices = sorted(episode_failures)
    successful_episode_indices = [
        ep_idx for ep_idx in range(len(records))
        if ep_idx not in episode_failures
    ]

    for ep_idx in skipped_episode_indices:
        for camera_name in FILE_CAMERA_MAPPING:
            path = output_path / "videos" / "chunk-000" / camera_name / f"episode_{ep_idx:06d}.mp4"
            if path.exists():
                path.unlink()

    for new_idx, old_idx in enumerate(successful_episode_indices):
        if new_idx == old_idx:
            continue
        for camera_name in FILE_CAMERA_MAPPING:
            src = output_path / "videos" / "chunk-000" / camera_name / f"episode_{old_idx:06d}.mp4"
            dst = output_path / "videos" / "chunk-000" / camera_name / f"episode_{new_idx:06d}.mp4"
            if src.exists():
                if dst.exists():
                    dst.unlink()
                src.rename(dst)

    processed_records = [records[ep_idx] for ep_idx in successful_episode_indices]
    processed_video_frame_counts = [
        min(episode_frame_counts[ep_idx].values())
        for ep_idx in successful_episode_indices
    ]

    if skipped_episode_indices:
        print(f"Skipped {len(skipped_episode_indices)} episodes after transcode")
        for ep_idx in skipped_episode_indices[:20]:
            print(f"  [{ep_idx}] {records[ep_idx].path}")
            for reason in episode_failures[ep_idx]:
                print(f"      - {reason}")
        if len(skipped_episode_indices) > 20:
            print(f"  ... {len(skipped_episode_indices) - 20} more")

    if not processed_records:
        raise RuntimeError("No episodes remained after video transcoding")

    build_start = time.time()
    dummy_image = np.zeros(shape, dtype=np.uint8)
    print("Building dataset...")
    for record, video_frame_count in tqdm.tqdm(
        zip(processed_records, processed_video_frame_counts, strict=True),
        total=len(processed_records),
        desc="Building dataset",
    ):
        state_array, action_array = load_json_data_with_phase(
            record.path,
            record.phase_boundaries,
            target_frame_count=video_frame_count,
            target_fps=target_fps,
        )
        num_frames = len(state_array)
        dataset._video_frame_count = num_frames - 1

        for frame_idx in range(num_frames - 1):
            dataset.add_frame(
                {
                    "face_view": dummy_image,
                    "left_wrist_view": dummy_image,
                    "right_wrist_view": dummy_image,
                    "state": state_array[frame_idx],
                    "actions": action_array[frame_idx + 1],
                    "task": record.prompt,
                }
            )

        dataset.save_episode()

    build_seconds = time.time() - build_start
    print(f"Dataset building completed in {build_seconds:.2f}s")

    print("Finalizing video info...")
    dataset.finalize_video_info()
    cleanup_empty_images_dir(output_path)
    write_phase_metadata(
        output_path,
        dataset_root,
        annotation_relative_path,
        prompt_relative_path,
        fallback_annotation_relative_path,
        target_fps,
        processed_records,
    )
    write_issue_metadata(
        output_path,
        issue_annotation_relative_path,
        issue_label,
        target_fps,
        processed_records,
        processed_video_frame_counts,
    )

    total_seconds = time.time() - total_start
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  Valid episodes:          {total_valid_records}")
    print(f"  Episodes selected:       {len(records)}")
    print(f"  Episodes processed:      {len(processed_records)}")
    print(f"  Skipped before transcode:{len(skipped_before_transcode):>8}")
    print(f"  Skipped after transcode: {len(skipped_episode_indices):>7}")
    print(f"  Transcode:               {transcode_seconds:.2f}s")
    print(f"  Build:                   {build_seconds:.2f}s")
    print(f"  Total:                   {total_seconds:.2f}s")
    print(f"  Dataset saved at:        {output_path}")
    print("=" * 60)

    if push_to_hub:
        dataset.push_to_hub(private=private)


if __name__ == "__main__":
    tyro.cli(main)
