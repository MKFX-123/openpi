#!/usr/bin/env python3
"""Detect repeated lid-pressing intervals in human-annotated pour-tea episodes."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import tyro


DEFAULT_DATASET_ROOT = Path("/mnt/public3/datasets/x1pro/pour_tea_training")
DEFAULT_SUBTASK_ANNOTATION = Path("anno/subtask_human.json")
DEFAULT_OUTPUT_ANNOTATION = Path("anno/pour_tea_issues_auto.json")

_NUMBER = rb"-?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?"
_RIGHT_GRIPPER_PATTERN = re.compile(rb'"master_right_gripper"\s*:\s*(' + _NUMBER + rb")")
_FOLLOW_RIGHT_POSITION_PATTERN = re.compile(
    rb'"follow_right_position"\s*:\s*\[\s*('
    + _NUMBER
    + rb")\s*,\s*("
    + _NUMBER
    + rb")\s*,\s*("
    + _NUMBER
    + rb")\s*\]"
)
_TOTAL_PATTERN = re.compile(rb'"total"\s*:\s*(\d+)')
_FPS_PATTERN = re.compile(rb'"fps"\s*:\s*(' + _NUMBER + rb")")


@dataclass(frozen=True)
class Detection:
    episode: Path
    start_frame: int
    end_frame: int
    fps: float

    @property
    def num_frames(self) -> int:
        return self.end_frame - self.start_frame


def _episode_json_path(episode: Path) -> Path:
    return episode / f"{episode.name}.json"


def _load_robot_trajectory(path: Path) -> tuple[np.ndarray, np.ndarray, float]:
    raw = path.read_bytes()
    gripper = np.fromiter(
        (float(match.group(1)) for match in _RIGHT_GRIPPER_PATTERN.finditer(raw)),
        dtype=np.float32,
    )
    position = np.fromiter(
        (
            float(value)
            for match in _FOLLOW_RIGHT_POSITION_PATTERN.finditer(raw)
            for value in match.groups()
        ),
        dtype=np.float32,
    ).reshape(-1, 3)
    total_match = _TOTAL_PATTERN.search(raw[:4096])
    if total_match is not None:
        total = int(total_match.group(1))
        if len(gripper) != total or len(position) != total:
            raise ValueError(
                f"trajectory lengths do not match total {total}: "
                f"gripper={len(gripper)}, follow_right_position={len(position)}"
            )
    elif len(gripper) != len(position):
        raise ValueError(
            f"trajectory length mismatch: gripper={len(gripper)}, follow_right_position={len(position)}"
        )
    fps_match = _FPS_PATTERN.search(raw[:4096])
    fps = float(fps_match.group(1)) if fps_match is not None else 30.0
    if fps <= 0:
        raise ValueError(f"invalid fps: {fps}")
    return gripper, position, fps


def _stable_runs(mask: np.ndarray, stable_frames: int) -> list[tuple[int, int, bool]]:
    changes = np.flatnonzero(mask[1:] != mask[:-1]) + 1
    runs: list[tuple[int, int, bool]] = []
    start = 0
    for end in [*changes.tolist(), len(mask)]:
        if end - start >= stable_frames:
            runs.append((start, end, bool(mask[start])))
        start = end
    return runs


def find_lid_release(
    gripper: np.ndarray,
    position: np.ndarray,
    phase_start: int,
    phase_end: int,
    *,
    open_threshold: float,
    stable_frames: int,
    max_grasp_height: float,
) -> int | None:
    """Find the opening after a closed-gripper run that starts at table height."""
    phase_open = gripper[phase_start : phase_end + 1] > open_threshold
    runs = _stable_runs(phase_open, stable_frames)
    observed_open = False
    closed_start: int | None = None
    for start, _, is_open in runs:
        frame = phase_start + start
        if is_open:
            observed_open = True
            if closed_start is None:
                continue
            grasp_height = float(position[closed_start, 2])
            if grasp_height <= max_grasp_height:
                return frame
            closed_start = None
        elif observed_open:
            closed_start = frame
    return None


def main(
    dataset_root: Path = DEFAULT_DATASET_ROOT,
    subtask_annotation: Path = DEFAULT_SUBTASK_ANNOTATION,
    output_annotation: Path = DEFAULT_OUTPUT_ANNOTATION,
    *,
    min_gap_frames: int = 200,
    open_threshold: float = 0.3,
    stable_frames: int = 5,
    max_grasp_height: float = 0.03,
    start_offset_frames: int = 50,
    issue_label: str = "5",
    dry_run: bool = False,
    overwrite: bool = False,
) -> None:
    if min_gap_frames < 1:
        raise ValueError(f"min_gap_frames must be positive, got {min_gap_frames}")
    if stable_frames < 1:
        raise ValueError(f"stable_frames must be positive, got {stable_frames}")
    if start_offset_frames < 0:
        raise ValueError(f"start_offset_frames must be non-negative, got {start_offset_frames}")

    detections: list[Detection] = []
    skipped: list[tuple[str, str]] = []
    annotation_paths = sorted(dataset_root.glob(f"*/{subtask_annotation}"))

    for annotation_path in annotation_paths:
        episode = annotation_path.parents[len(subtask_annotation.parts) - 1]
        try:
            annotation = json.loads(annotation_path.read_text(encoding="utf-8"))
            phase_3 = annotation.get("3")
            phase_4 = annotation.get("4")
            if not isinstance(phase_3, list) or not phase_3:
                raise ValueError('missing "3"[0]')
            if not isinstance(phase_4, list) or not phase_4:
                raise ValueError('missing "4"[0]')

            phase_start = int(round(float(phase_3[0])))
            phase_end = int(round(float(phase_4[0])))
            gripper, position, fps = _load_robot_trajectory(_episode_json_path(episode))
            if not 0 <= phase_start < phase_end < len(gripper):
                raise ValueError(
                    f"invalid phase range [{phase_start}, {phase_end}] for {len(gripper)} frames"
                )

            lid_release = find_lid_release(
                gripper,
                position,
                phase_start,
                phase_end,
                open_threshold=open_threshold,
                stable_frames=stable_frames,
                max_grasp_height=max_grasp_height,
            )
            if lid_release is None:
                skipped.append(
                    (
                        episode.name,
                        "no table-height grasp followed by a stable opening",
                    )
                )
                continue
            if phase_end - lid_release >= min_gap_frames:
                detections.append(
                    Detection(episode, lid_release + start_offset_frames, phase_end, fps)
                )
        except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
            skipped.append((episode.name, str(exc)))

    existing_outputs = sorted(dataset_root.glob(f"*/{output_annotation}"))
    if existing_outputs and not overwrite and not dry_run:
        raise FileExistsError(
            f"Found {len(existing_outputs)} existing {output_annotation} files; pass --overwrite to regenerate"
        )

    if not dry_run:
        output_payloads = {
            annotation_path.parents[len(subtask_annotation.parts) - 1] / output_annotation: {}
            for annotation_path in annotation_paths
        }
        if overwrite:
            output_payloads.update({path: {} for path in existing_outputs})
        for detection in detections:
            output_payloads[detection.episode / output_annotation] = {
                issue_label: [detection.start_frame, detection.end_frame]
            }

        for output_path, payload in output_payloads.items():
            output_path.parent.mkdir(parents=True, exist_ok=True)
            temp_path = output_path.with_suffix(output_path.suffix + ".tmp")
            temp_path.write_text(
                json.dumps(payload, indent=2) + "\n",
                encoding="utf-8",
            )
            temp_path.replace(output_path)

    total_frames = sum(detection.num_frames for detection in detections)
    total_seconds = sum(detection.num_frames / detection.fps for detection in detections)
    print(f"Human annotations: {len(annotation_paths)}")
    print(f"Detected issues: {len(detections)}")
    print(f"Skipped episodes: {len(skipped)}")
    print(f"Total issue frames: {total_frames}")
    if detections:
        print(f"Mean issue frames: {total_frames / len(detections):.2f}")
        print(f"Mean issue seconds: {total_seconds / len(detections):.2f}")
    print(f"Output annotation: {output_annotation}")
    print(f"Dry run: {dry_run}")
    if not dry_run:
        print(f"Empty output annotations: {len(output_payloads) - len(detections)}")
    if skipped:
        print("Skipped:")
        for episode, reason in skipped:
            print(f"  {episode}: {reason}")


if __name__ == "__main__":
    tyro.cli(main)
