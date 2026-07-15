#!/usr/bin/env python3
"""Generate frame-range subtask prompts from episode annotation key frames."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import itertools
import json
from pathlib import Path
import re
import sys
from typing import Any


@dataclass(frozen=True)
class SegmentSpec:
    segment_id: str
    prompt: str
    annotation_key: str
    annotation_index: int


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    with temporary.open("w", encoding="utf-8") as file:
        json.dump(value, file, ensure_ascii=False, indent=2)
        file.write("\n")
    temporary.replace(path)


def load_schema(path: Path) -> list[SegmentSpec]:
    schema = load_json(path)
    if not isinstance(schema, dict):
        raise ValueError("schema root must be a JSON object")
    if schema.get("annotation_unit") != "frame_index":
        raise ValueError("schema annotation_unit must be 'frame_index'")
    raw_segments = schema.get("segments")
    if not isinstance(raw_segments, list) or not raw_segments:
        raise ValueError("schema segments must be a non-empty list")

    specs: list[SegmentSpec] = []
    seen_ids: set[str] = set()
    for position, raw in enumerate(raw_segments):
        if not isinstance(raw, dict):
            raise ValueError(f"schema segment {position} must be an object")
        segment_id = raw.get("id")
        prompt = raw.get("prompt")
        start = raw.get("start")
        if not isinstance(segment_id, str) or not re.fullmatch(r"[A-Za-z0-9_-]+", segment_id):
            raise ValueError(f"invalid segment id at position {position}: {segment_id!r}")
        if segment_id in seen_ids:
            raise ValueError(f"duplicate segment id: {segment_id}")
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError(f"empty prompt for segment {segment_id}")
        if not isinstance(start, dict):
            raise ValueError(f"missing start mapping for segment {segment_id}")
        key = start.get("key")
        index = start.get("index")
        if not isinstance(key, str) or not isinstance(index, int) or index < 0:
            raise ValueError(f"invalid start mapping for segment {segment_id}: {start!r}")
        specs.append(SegmentSpec(segment_id, prompt.strip(), key, index))
        seen_ids.add(segment_id)
    return specs


def find_episode_json(episode_dir: Path) -> Path:
    canonical = episode_dir / f"{episode_dir.name}.json"
    if canonical.is_file():
        return canonical
    candidates = sorted(episode_dir.glob("*.json"))
    if len(candidates) != 1:
        raise ValueError(
            f"expected {canonical.name} or exactly one root-level JSON; found {len(candidates)}"
        )
    return candidates[0]


def read_episode_metadata(path: Path) -> tuple[int, float]:
    with path.open("r", encoding="utf-8") as file:
        header = file.read(8192)
    total_match = re.search(r'"total"\s*:\s*(\d+)', header)
    fps_match = re.search(r'"fps"\s*:\s*([0-9]+(?:\.[0-9]+)?)', header)
    if total_match and fps_match:
        total = int(total_match.group(1))
        fps = float(fps_match.group(1))
    else:
        payload = load_json(path)
        data = payload.get("data")
        if not isinstance(data, list):
            raise ValueError(f"episode JSON has no data list: {path}")
        total = int(payload.get("total", len(data)))
        fps = float(payload.get("fps", 30.0))
    if total <= 0 or fps <= 0:
        raise ValueError(f"invalid total/fps: total={total}, fps={fps}")
    return total, fps


def read_start_frames(annotation: Any, specs: list[SegmentSpec], total: int) -> list[int]:
    if not isinstance(annotation, dict):
        raise ValueError("annotation root must be an object")
    starts: list[int] = []
    for spec in specs:
        values = annotation.get(spec.annotation_key)
        if not isinstance(values, list) or spec.annotation_index >= len(values):
            raise ValueError(
                f"missing {spec.annotation_key}[{spec.annotation_index}] for {spec.segment_id}"
            )
        value = values[spec.annotation_index]
        if isinstance(value, bool) or not isinstance(value, int | float):
            raise ValueError(
                f"non-numeric {spec.annotation_key}[{spec.annotation_index}]={value!r}"
            )
        frame = round(float(value))
        if frame < 0 or frame >= total:
            raise ValueError(
                f"{spec.annotation_key}[{spec.annotation_index}]={frame} outside [0, {total})"
            )
        starts.append(frame)
    if any(left >= right for left, right in itertools.pairwise(starts)):
        raise ValueError(f"segment starts are not strictly increasing: {starts}")
    return starts


def build_output(
    annotation_rel_path: Path,
    total: int,
    fps: float,
    starts: list[int],
    specs: list[SegmentSpec],
) -> dict[str, Any]:
    ends = [*starts[1:], total]
    segments = []
    for spec, start, end in zip(specs, starts, ends, strict=True):
        segments.append(
            {
                "id": spec.segment_id,
                "start_frame": start,
                "end_frame": end,
                "prompt": spec.prompt,
            }
        )
    return {
        "version": 1,
        "annotation_unit": "frame_index",
        "interval_convention": "[start_frame, end_frame)",
        "source_annotation": annotation_rel_path.as_posix(),
        "fps": fps,
        "total_frames": total,
        "annotated_start_frame": starts[0],
        "annotated_end_frame": total,
        "segments": segments,
    }


def safe_relative_path(value: Path, option: str) -> Path:
    if value.is_absolute() or ".." in value.parts:
        raise ValueError(f"{option} must be a safe relative path")
    return value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--input-anno-rel-path", type=Path, required=True)
    parser.add_argument("--schema", type=Path, required=True)
    parser.add_argument(
        "--output-anno-rel-path",
        type=Path,
        default=Path("anno/subtask_prompt.json"),
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--strict", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    input_dir = args.input_dir.resolve()
    if not input_dir.is_dir():
        raise FileNotFoundError(f"input directory does not exist: {input_dir}")
    annotation_rel_path = safe_relative_path(args.input_anno_rel_path, "--input-anno-rel-path")
    output_rel_path = safe_relative_path(args.output_anno_rel_path, "--output-anno-rel-path")
    if annotation_rel_path == output_rel_path:
        raise ValueError("input and output annotation paths must differ")
    specs = load_schema(args.schema.resolve())

    discovered = 0
    written = 0
    unchanged = 0
    invalid: list[tuple[str, str]] = []
    conflicts: list[Path] = []
    pending: list[tuple[Path, dict[str, Any]]] = []

    for episode_dir in sorted(path for path in input_dir.iterdir() if path.is_dir()):
        annotation_path = episode_dir / annotation_rel_path
        if not annotation_path.is_file():
            continue
        discovered += 1
        try:
            total, fps = read_episode_metadata(find_episode_json(episode_dir))
            starts = read_start_frames(load_json(annotation_path), specs, total)
            output = build_output(annotation_rel_path, total, fps, starts, specs)
            output_path = episode_dir / output_rel_path
            if output_path.is_file():
                try:
                    current = load_json(output_path)
                except (OSError, json.JSONDecodeError):
                    current = None
                if current == output:
                    unchanged += 1
                    continue
                if not args.overwrite:
                    conflicts.append(output_path)
                    continue
            pending.append((output_path, output))
        except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
            invalid.append((episode_dir.name, str(exc)))

    print(f"Annotation files discovered: {discovered}")
    print(f"Valid complete annotations: {len(pending) + unchanged + len(conflicts)}")
    print(f"Invalid/incomplete annotations: {len(invalid)}")
    print(f"Already up to date: {unchanged}")
    print(f"Files to write: {len(pending)}")
    if invalid:
        for episode, reason in invalid:
            print(f"SKIP {episode}: {reason}", file=sys.stderr)
    if conflicts:
        for path in conflicts[:20]:
            print(f"CONFLICT {path}", file=sys.stderr)
        if len(conflicts) > 20:
            print(f"... and {len(conflicts) - 20} more conflicts", file=sys.stderr)
        print("Use --overwrite to replace conflicting output files.", file=sys.stderr)
        return 3
    if args.strict and invalid:
        return 2
    if args.dry_run:
        return 0

    for output_path, output in pending:
        write_json(output_path, output)
        written += 1
    print(f"Written: {written}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
