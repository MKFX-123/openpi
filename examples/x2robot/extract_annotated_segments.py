"""Extract frame-aligned X2Robot episodes from paired frame annotations."""

import concurrent.futures
import dataclasses
import json
import os
from pathlib import Path
import shutil
import subprocess

import tqdm
import tyro

VIDEO_FILES = ("faceImg.mp4", "leftImg.mp4", "rightImg.mp4")


@dataclasses.dataclass(frozen=True)
class Segment:
    source_episode: Path
    segment_index: int
    start_frame: int
    end_frame: int

    @property
    def output_name(self) -> str:
        return f"{self.source_episode.name}__pick_and_place_pen_{self.segment_index:02d}"

    @property
    def length(self) -> int:
        return self.end_frame - self.start_frame


def _run(command: list[str]) -> str:
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        raise RuntimeError(f"Command failed: {' '.join(command)}\n{result.stderr}")
    return result.stdout.strip()


def _video_frame_count(path: Path) -> int:
    output = _run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-count_packets",
            "-show_entries",
            "stream=nb_read_packets",
            "-of",
            "csv=p=0",
            str(path),
        ]
    )
    return int(output)


def _trim_video(source: Path, output: Path, start_frame: int, end_frame: int) -> None:
    _run(
        [
            "ffmpeg",
            "-y",
            "-nostdin",
            "-v",
            "error",
            "-i",
            str(source),
            "-vf",
            f"trim=start_frame={start_frame}:end_frame={end_frame},setpts=PTS-STARTPTS",
            "-an",
            "-c:v",
            "libx264",
            "-preset",
            "fast",
            "-crf",
            "18",
            "-pix_fmt",
            "yuv420p",
            "-fps_mode",
            "passthrough",
            str(output),
        ]
    )


def _extract_segment(
    segment: Segment, output_root: Path, annotation_relative_path: Path, annotation_key: str
) -> dict:
    source_json = segment.source_episode / f"{segment.source_episode.name}.json"
    payload = json.loads(source_json.read_text(encoding="utf-8"))
    source_frames = payload["data"]
    if not 0 <= segment.start_frame < segment.end_frame <= len(source_frames):
        raise ValueError(
            f"Invalid range [{segment.start_frame}, {segment.end_frame}) for "
            f"{segment.source_episode.name} with {len(source_frames)} frames"
        )

    output_dir = output_root / segment.output_name
    temp_dir = output_root / f".{segment.output_name}.tmp-{os.getpid()}"
    if output_dir.exists() or temp_dir.exists():
        raise FileExistsError(f"Output episode already exists: {output_dir}")

    temp_dir.mkdir(parents=True)
    try:
        for filename in VIDEO_FILES:
            _trim_video(
                segment.source_episode / filename,
                temp_dir / filename,
                segment.start_frame,
                segment.end_frame,
            )
            actual_frames = _video_frame_count(temp_dir / filename)
            if actual_frames != segment.length:
                raise ValueError(
                    f"{segment.output_name}/{filename} has {actual_frames} frames, expected {segment.length}"
                )

        clipped_payload = dict(payload)
        clipped_payload["name"] = segment.output_name
        clipped_payload["total"] = segment.length
        clipped_payload["data"] = source_frames[segment.start_frame : segment.end_frame]
        (temp_dir / f"{segment.output_name}.json").write_text(
            json.dumps(clipped_payload, ensure_ascii=False, separators=(",", ":")),
            encoding="utf-8",
        )

        annotation_path = temp_dir / annotation_relative_path
        annotation_path.parent.mkdir(parents=True)
        annotation_path.write_text(
            json.dumps({annotation_key: [0, segment.length]}, indent=2),
            encoding="utf-8",
        )
        (annotation_path.parent / "source_segment.json").write_text(
            json.dumps(
                {
                    "source_episode": segment.source_episode.name,
                    "source_start_frame": segment.start_frame,
                    "source_end_frame": segment.end_frame,
                    "end_frame_exclusive": True,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        temp_dir.rename(output_dir)
    except Exception:
        shutil.rmtree(temp_dir, ignore_errors=True)
        raise

    return {
        "episode": segment.output_name,
        "source_episode": segment.source_episode.name,
        "segment_index": segment.segment_index,
        "start_frame": segment.start_frame,
        "end_frame": segment.end_frame,
        "frames": segment.length,
    }


def _discover_segments(
    dataset_root: Path,
    annotation_relative_path: Path,
    annotation_key: str,
) -> tuple[list[Segment], int]:
    segments = []
    empty_annotations = 0
    for annotation_path in sorted(dataset_root.glob(f"*/{annotation_relative_path}")):
        episode = annotation_path.parents[len(annotation_relative_path.parts) - 1]
        annotation = json.loads(annotation_path.read_text(encoding="utf-8"))
        boundaries = annotation.get(annotation_key, [])
        if not isinstance(boundaries, list):
            raise ValueError(f"{annotation_path} does not contain a list at key {annotation_key!r}")
        if len(boundaries) % 2:
            raise ValueError(f"{annotation_path} contains an odd number of boundaries: {boundaries}")
        if not boundaries:
            empty_annotations += 1
            continue
        for index in range(0, len(boundaries), 2):
            start_frame = int(boundaries[index])
            end_frame = int(boundaries[index + 1])
            if start_frame >= end_frame:
                raise ValueError(f"Invalid range [{start_frame}, {end_frame}) in {annotation_path}")
            segments.append(Segment(episode, index // 2, start_frame, end_frame))
    return segments, empty_annotations


def main(
    dataset_root: Path,
    output_root: Path,
    annotation_relative_path: Path = Path("anno/pick_and_place_pen.json"),
    annotation_key: str = "0",
    num_workers: int = 8,
    overwrite: bool = False,  # noqa: FBT001, FBT002
) -> None:
    if num_workers < 1:
        raise ValueError(f"num_workers must be positive, got {num_workers}")
    segments, empty_annotations = _discover_segments(
        dataset_root,
        annotation_relative_path,
        annotation_key,
    )
    if not segments:
        raise ValueError(f"No annotated segments found under {dataset_root}")

    if output_root.exists():
        if not overwrite:
            raise FileExistsError(f"{output_root} already exists; pass --overwrite to replace it")
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True)

    results = []
    failures = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = {
            executor.submit(
                _extract_segment, segment, output_root, annotation_relative_path, annotation_key
            ): segment
            for segment in segments
        }
        for future in tqdm.tqdm(
            concurrent.futures.as_completed(futures),
            total=len(futures),
            desc="Extracting segments",
        ):
            segment = futures[future]
            try:
                results.append(future.result())
            except Exception as exc:
                failures.append((segment, exc))

    results.sort(key=lambda item: item["episode"])
    with (output_root / "segments.jsonl").open("w", encoding="utf-8") as f:
        for result in results:
            f.write(json.dumps(result) + "\n")

    print(f"Annotations with no segments: {empty_annotations}")
    print(f"Segments extracted: {len(results)}/{len(segments)}")
    print(f"Frames extracted: {sum(item['frames'] for item in results)}")
    print(f"Output: {output_root}")
    if failures:
        for segment, exc in failures:
            print(f"FAILED {segment.output_name}: {exc}")
        raise RuntimeError(f"Failed to extract {len(failures)} segments")


if __name__ == "__main__":
    tyro.cli(main)
