#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Convert UMI (umi-v260729) pick_and_place episodes to a LeRobot dataset.

Input (produced by anno-platform/server/scripts/sync_robot_bags.py +
mcap_to_training.py with parse_version umi-v260729):

  {stage1_dir}/{episode}/
    ├── {episode}.json          # per-frame: follow_{left,right}_position(3)
    │                           #   + follow_{left,right}_rotation(euler xyz,3)
    │                           #   + follow_{left,right}_gripper(1) + timestamp
    ├── faceImg.mp4
    ├── leftImg.mp4
    └── rightImg.mp4

Output: a LeRobot dataset (repo_id) with:

  face_view / left_wrist_view / right_wrist_view   (video, ffmpeg transcoded)
  follow_left_pos(3) / follow_left_rotvec(3) / follow_left_gripper(1)   # abs map frame
  follow_right_pos(3) / follow_right_rotvec(3) / follow_right_gripper(1)
  left_action(7) / right_action(7)               # NEXT-frame abs pose+gripper
  actions(14)                                    # concat(left_action, right_action)
  demo_start_pose_left(6) / demo_start_pose_right(6)   # first-frame pos+rotvec
  task (str)

Conventions:
  - rotation stored as ROTVEC (euler->rotvec conversion here), matching umi_policy.
  - actions stored as ABSOLUTE next-frame pose (pos+rotvec+gripper). The relative
    conversion (inv(cur)@target -> rot6d) is done in umi_policy.UmiInputs, NOT here.
  - last frame is dropped (action = frame i+1), so writable frames = total - 1.

Engineering (copied from convert_x2robot_data_to_lerobot_v5.py):
  - Phase 1: parallel ffmpeg transcode of source MP4 -> target codec.
  - Phase 2: metadata-only dataset building with dummy video stats (skip image I/O).
  - Failed episodes are skipped and remaining episodes are renumbered to stay contiguous.
"""

import argparse
import glob
import json
import os
import shutil
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import tqdm
from scipy.spatial.transform import Rotation as R
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset, HF_LEROBOT_HOME

os.environ["SVT_LOG"] = "0"

FILE_CAMERA_MAPPING = {
    "face_view": "faceImg.mp4",
    "left_wrist_view": "leftImg.mp4",
    "right_wrist_view": "rightImg.mp4",
}


# ------------------------------------------------------------------
# Pose helpers
# ------------------------------------------------------------------
def euler_to_rotvec(euler_xyz: np.ndarray) -> np.ndarray:
    """Euler intrinsic xyz -> rotation vector."""
    return R.from_euler("xyz", np.asarray(euler_xyz, dtype=np.float64)).as_rotvec().astype(np.float32)


def load_episode_json(path: Path):
    """Load umi-v260729 episode json.

    Returns arrays:
      pos_l(N,3), rotvec_l(N,3), grip_l(N,), pos_r, rotvec_r, grip_r, fps.
    """
    meta = json.loads(path.read_text(encoding="utf-8"))
    data = meta["data"]
    if len(data) < 2:
        raise ValueError(f"{path} has <2 frames")
    recs = data
    pos_l = np.array([f["follow_left_position"] for f in recs], dtype=np.float32)
    eul_l = np.array([f["follow_left_rotation"] for f in recs], dtype=np.float32)
    grip_l = np.array([f["follow_left_gripper"] for f in recs], dtype=np.float32).reshape(-1, 1)
    pos_r = np.array([f["follow_right_position"] for f in recs], dtype=np.float32)
    eul_r = np.array([f["follow_right_rotation"] for f in recs], dtype=np.float32)
    grip_r = np.array([f["follow_right_gripper"] for f in recs], dtype=np.float32).reshape(-1, 1)
    rotvec_l = euler_to_rotvec(eul_l)
    rotvec_r = euler_to_rotvec(eul_r)
    fps = float(meta.get("fps", 30.0))
    return pos_l, rotvec_l, grip_l, pos_r, rotvec_r, grip_r, fps, len(recs)


def find_episodes(stage1_dir: Path, keep_list: Path | None = None) -> list[Path]:
    """Find episode directories that contain a {name}.json + 3 mp4.

    If keep_list is given, only episode dirs whose name (without trailing /)
    is listed in that file (one per line) are returned.
    """
    keep = None
    if keep_list is not None:
        keep = {ln.strip().rstrip("/") for ln in keep_list.read_text().splitlines() if ln.strip()}
    eps = []
    for d in sorted(stage1_dir.iterdir()):
        if not d.is_dir():
            continue
        if keep is not None and d.name not in keep:
            continue
        j = d / f"{d.name}.json"
        if not j.is_file():
            continue
        if not all((d / v).is_file() for v in FILE_CAMERA_MAPPING.values()):
            continue
        eps.append(d)
    return eps


# ------------------------------------------------------------------
# ffmpeg transcode (copied from v5)
# ------------------------------------------------------------------
def transcode_video_ffmpeg(
    input_path: Path,
    output_path: Path,
    target_size: tuple[int, int] | None,
    fps: int,
    vcodec: str,
    pix_fmt: str = "yuv420p",
    g: int = 2,
    crf: int = 23,
) -> int:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    vf = [] if target_size is None else [f"scale={target_size[0]}:{target_size[1]}"]
    cmd = ["ffmpeg", "-y", "-nostdin", "-i", str(input_path)]
    if vf:
        cmd += ["-vf", ",".join(vf)]
    cmd += ["-c:v", vcodec, "-pix_fmt", pix_fmt, "-r", str(fps), "-g", str(g), "-crf", str(crf), str(output_path)]
    result = subprocess.run(cmd, capture_output=True, text=True, env={**os.environ, "SVT_LOG": "0"})
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg transcode failed for {input_path}: {result.stderr[-1000:]}")

    probe_cmd = [
        "ffprobe", "-v", "error", "-select_streams", "v:0", "-count_packets",
        "-show_entries", "stream=nb_read_packets", "-of", "csv=p=0", str(output_path),
    ]
    probe = subprocess.run(probe_cmd, capture_output=True, text=True)
    if probe.returncode != 0 or not probe.stdout.strip():
        raise RuntimeError(f"ffprobe failed for {output_path}: {probe.stderr}")
    num_frames = int(probe.stdout.strip())

    check = subprocess.run(
        ["ffmpeg", "-v", "error", "-xerror", "-nostdin", "-i", str(output_path), "-f", "null", "-"],
        capture_output=True, text=True,
    )
    if check.returncode != 0 or check.stderr.strip():
        raise RuntimeError(f"ffmpeg decode check failed for {output_path}: {check.stderr[-1000:]}")
    return num_frames


def transcode_single_video(
    episode_dir: Path, episode_index: int, camera_name: str, video_filename: str,
    output_root: Path, target_size, fps: int, video_codec: str,
) -> tuple[int, str, int]:
    video_path = episode_dir / video_filename
    output_path = output_root / "videos" / "chunk-000" / camera_name / f"episode_{episode_index:06d}.mp4"
    if video_codec == "av1":
        vcodec, crf = "libsvtav1", 30
    elif video_codec == "h264":
        vcodec, crf = "libx264", 23
    else:
        raise ValueError(f"Unsupported video codec: {video_codec}")
    num_frames = transcode_video_ffmpeg(
        video_path, output_path, target_size, fps, vcodec=vcodec, pix_fmt="yuv420p", g=2, crf=crf,
    )
    return episode_index, camera_name, num_frames


# ------------------------------------------------------------------
# Dummy video stats (copied from v5)
# ------------------------------------------------------------------
def get_dummy_video_stats(num_frames: int) -> dict:
    return {
        "min": np.array([[[0.0]], [[0.0]], [[0.0]]]),
        "max": np.array([[[1.0]], [[1.0]], [[1.0]]]),
        "mean": np.array([[[0.4]], [[0.4]], [[0.4]]]),
        "std": np.array([[[0.25]], [[0.25]], [[0.25]]]),
        "count": np.array([num_frames]),
    }


def compute_episode_stats_with_dummy_video(episode_buffer: dict, features: dict, video_frame_count: int) -> dict:
    from lerobot.common.datasets.compute_stats import get_feature_stats
    ep_stats = {}
    for key, data in episode_buffer.items():
        if key not in features:
            continue
        if features[key]["dtype"] == "string":
            continue
        elif features[key]["dtype"] in ["image", "video"]:
            ep_stats[key] = get_dummy_video_stats(video_frame_count)
        else:
            axes_to_reduce = 0
            keepdims = data.ndim == 1
            ep_stats[key] = get_feature_stats(data, axis=axes_to_reduce, keepdims=keepdims)
    return ep_stats


class NoVideoIOLeRobotDataset(LeRobotDataset):
    """LeRobotDataset that skips all video/image I/O in save_episode (copied from v5)."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._skip_all_media = False
        self._video_frame_count = 0

    def _save_image(self, image, fpath: Path) -> None:
        if self._skip_all_media:
            return
        super()._save_image(image, fpath)

    def save_episode(self, episode_data: dict | None = None) -> None:
        if not self._skip_all_media:
            super().save_episode(episode_data)
            return
        if not episode_data:
            episode_buffer = self.episode_buffer

        from lerobot.common.datasets.lerobot_dataset import (
            validate_episode_buffer, write_info, write_episode, write_episode_stats, aggregate_stats,
        )
        validate_episode_buffer(episode_buffer, self.meta.total_episodes, self.features)

        episode_length = episode_buffer.pop("size")
        tasks = episode_buffer.pop("task")
        episode_tasks = list(set(tasks))
        episode_index = episode_buffer["episode_index"]

        episode_buffer["index"] = np.arange(self.meta.total_frames, self.meta.total_frames + episode_length)
        episode_buffer["episode_index"] = np.full((episode_length,), episode_index)

        for task in episode_tasks:
            if self.meta.get_task_index(task) is None:
                self.meta.add_task(task)
        episode_buffer["task_index"] = np.array([self.meta.get_task_index(task) for task in tasks])

        for key, ft in self.features.items():
            if key in ["index", "episode_index", "task_index"] or ft["dtype"] in ["image", "video"]:
                continue
            episode_buffer[key] = np.stack(episode_buffer[key])

        self._wait_image_writer()
        self._save_episode_table(episode_buffer, episode_index)

        ep_stats = compute_episode_stats_with_dummy_video(episode_buffer, self.features, self._video_frame_count)

        self.meta.info["total_episodes"] += 1
        self.meta.info["total_frames"] += episode_length
        chunk = self.meta.get_episode_chunk(episode_index)
        if chunk >= self.meta.total_chunks:
            self.meta.info["total_chunks"] += 1
        self.meta.info["splits"] = {"train": f"0:{self.meta.info['total_episodes']}"}
        self.meta.info["total_videos"] += len(self.meta.video_keys)
        write_info(self.meta.info, self.meta.root)

        episode_dict = {"episode_index": episode_index, "tasks": episode_tasks, "length": episode_length}
        self.meta.episodes[episode_index] = episode_dict
        write_episode(episode_dict, self.meta.root)

        self.meta.episodes_stats[episode_index] = ep_stats
        self.meta.stats = aggregate_stats([self.meta.stats, ep_stats]) if self.meta.stats else ep_stats
        write_episode_stats(episode_index, ep_stats, self.meta.root)

        if not episode_data:
            self.episode_buffer = self.create_episode_buffer()

    def finalize_video_info(self) -> None:
        self.meta.update_video_info()
        from lerobot.common.datasets.lerobot_dataset import write_info
        write_info(self.meta.info, self.meta.root)

    @classmethod
    def create(cls, **kwargs) -> "NoVideoIOLeRobotDataset":
        parent_obj = LeRobotDataset.create(**kwargs)
        obj = cls.__new__(cls)
        obj.__dict__.update(parent_obj.__dict__)
        obj._skip_all_media = False
        obj._video_frame_count = 0
        return obj


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description="Convert UMI umi-v260729 episodes to LeRobot.")
    p.add_argument("--stage1-dir", required=True, help="dir of umi-v260729 episodes (the training dir)")
    p.add_argument("--repo", required=True, help="LeRobot repo_id (relative to HF_LEROBOT_HOME)")
    p.add_argument("--task", default="pick and place")
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--video-codec", choices=["h264", "av1"], default="h264")
    p.add_argument("--low-resolution", action="store_true", help="320x240; otherwise keep source resolution")
    p.add_argument("--num-workers", type=int, default=10)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--debug", action="store_true")
    p.add_argument("--debug-episodes", type=int, default=3)
    p.add_argument("--keep-list", type=Path, default=None,
                   help="file with episode dir names to keep (one per line); skips the rest")
    return p.parse_args()


def main():
    args = parse_args()
    stage1_dir = Path(args.stage1_dir).expanduser().resolve()
    if not stage1_dir.is_dir():
        raise FileNotFoundError(stage1_dir)

    episode_dirs = find_episodes(stage1_dir, keep_list=args.keep_list)
    if args.debug:
        episode_dirs = episode_dirs[: args.debug_episodes]
    print(f"[INFO] Found {len(episode_dirs)} episodes under {stage1_dir}")
    if not episode_dirs:
        raise SystemExit("No episodes found.")

    target_size = (320, 240) if args.low_resolution else None

    output_path = HF_LEROBOT_HOME / args.repo
    if output_path.exists():
        if not args.overwrite:
            raise FileExistsError(f"{output_path} already exists. Pass --overwrite to replace it.")
        print(f"[INFO] Removing existing dataset at {output_path}")
        shutil.rmtree(output_path)

    # probe first episode to get image shapes
    first = episode_dirs[0]
    import imageio.v3 as iio
    Hl, Wl = np.asarray(iio.imread(str(first / "leftImg.mp4"), index=0), dtype=np.uint8).shape[:2]
    Hr, Wr = np.asarray(iio.imread(str(first / "rightImg.mp4"), index=0), dtype=np.uint8).shape[:2]
    Ht, Wt = np.asarray(iio.imread(str(first / "faceImg.mp4"), index=0), dtype=np.uint8).shape[:2]
    if target_size is not None:
        (Hl, Wl), (Hr, Wr), (Ht, Wt) = (target_size[1], target_size[0]), (target_size[1], target_size[0]), (target_size[1], target_size[0])

    features = {
        "face_view": {"dtype": "video", "shape": (Ht, Wt, 3), "names": ["h", "w", "c"]},
        "left_wrist_view": {"dtype": "video", "shape": (Hl, Wl, 3), "names": ["h", "w", "c"]},
        "right_wrist_view": {"dtype": "video", "shape": (Hr, Wr, 3), "names": ["h", "w", "c"]},
        "follow_left_pos": {"dtype": "float32", "shape": (3,), "names": ["x", "y", "z"]},
        "follow_left_rotvec": {"dtype": "float32", "shape": (3,), "names": ["rx", "ry", "rz"]},
        "follow_left_gripper": {"dtype": "float32", "shape": (1,), "names": ["g"]},
        "follow_right_pos": {"dtype": "float32", "shape": (3,), "names": ["x", "y", "z"]},
        "follow_right_rotvec": {"dtype": "float32", "shape": (3,), "names": ["rx", "ry", "rz"]},
        "follow_right_gripper": {"dtype": "float32", "shape": (1,), "names": ["g"]},
        "demo_start_pose_left": {"dtype": "float32", "shape": (6,), "names": ["x", "y", "z", "rx", "ry", "rz"]},
        "demo_start_pose_right": {"dtype": "float32", "shape": (6,), "names": ["x", "y", "z", "rx", "ry", "rz"]},
        "left_action": {"dtype": "float32", "shape": (7,), "names": ["x", "y", "z", "rx", "ry", "rz", "g"]},
        "right_action": {"dtype": "float32", "shape": (7,), "names": ["x", "y", "z", "rx", "ry", "rz", "g"]},
        "actions": {"dtype": "float32", "shape": (14,), "names": [
            "lx", "ly", "lz", "lrx", "lry", "lrz", "lg",
            "rx", "ry", "rz", "rrx", "rry", "rrz", "rg",
        ]},
    }

    dataset = NoVideoIOLeRobotDataset.create(
        repo_id=args.repo,
        robot_type="X1PRO_DUAL",
        fps=args.fps,
        features=features,
        image_writer_threads=0,
        image_writer_processes=0,
    )
    dataset._skip_all_media = True

    # ---------- Phase 1: parallel transcode ----------
    print(f"\n{'='*60}\nPHASE 1: parallel ffmpeg transcoding\n{'='*60}")
    t0 = time.time()
    tasks = []
    for ep_idx, ep in enumerate(episode_dirs):
        for cam, fn in FILE_CAMERA_MAPPING.items():
            tasks.append((ep, ep_idx, cam, fn, output_path, target_size, args.fps, args.video_codec))

    episode_frame_counts: dict[int, dict[str, int]] = {}
    failures: dict[int, list[str]] = {}
    with ThreadPoolExecutor(max_workers=args.num_workers) as ex:
        futs = {ex.submit(transcode_single_video, *t): t[:3] for t in tasks}
        with tqdm.tqdm(total=len(tasks), desc="transcode") as pbar:
            for fut in as_completed(futs):
                try:
                    ep_idx, cam, n = fut.result()
                    episode_frame_counts.setdefault(ep_idx, {})[cam] = n
                except Exception as e:
                    ep_dir, ep_idx, cam = futs[fut]
                    failures.setdefault(ep_idx, []).append(f"{cam} transcode failed: {str(e)[-300:]}")
                    print(f"\n[skip] ep{ep_idx} {cam}: {str(e)[-300:]}")
                pbar.update(1)
    print(f"[INFO] transcode done in {time.time()-t0:.1f}s")

    skipped = sorted(failures)
    ok_indices = [i for i in range(len(episode_dirs)) if i not in failures]
    # cleanup failed videos + renumber
    for i in skipped:
        for cam in FILE_CAMERA_MAPPING:
            p = output_path / "videos" / "chunk-000" / cam / f"episode_{i:06d}.mp4"
            if p.exists():
                p.unlink()
    for new, old in enumerate(ok_indices):
        if new == old:
            continue
        for cam in FILE_CAMERA_MAPPING:
            src = output_path / "videos" / "chunk-000" / cam / f"episode_{old:06d}.mp4"
            dst = output_path / "videos" / "chunk-000" / cam / f"episode_{new:06d}.mp4"
            if src.exists():
                if dst.exists():
                    dst.unlink()
                src.rename(dst)
    processed_eps = [episode_dirs[i] for i in ok_indices]
    processed_counts = [min(episode_frame_counts[i].values()) for i in ok_indices]
    if skipped:
        print(f"[INFO] skipped {len(skipped)} episodes: {skipped}")
    if not processed_eps:
        raise RuntimeError("No valid episodes left after transcoding.")

    # ---------- Phase 2: build dataset (metadata only) ----------
    print(f"\n{'='*60}\nPHASE 2: building dataset (dummy video stats)\n{'='*60}")
    t0 = time.time()
    def _dummy(h, w):
        return np.zeros((h, w, 3), dtype=np.uint8)
    dummy_face = _dummy(Ht, Wt)
    dummy_left = _dummy(Hl, Wl)
    dummy_right = _dummy(Hr, Wr)

    import datasets as hfdatasets
    hfdatasets.disable_progress_bars()

    total_eps = total_frames = 0
    for ep_dir, vfc in tqdm.tqdm(list(zip(processed_eps, processed_counts, strict=True)), desc="build"):
        ep_json = ep_dir / f"{ep_dir.name}.json"
        try:
            pos_l, rvec_l, grip_l, pos_r, rvec_r, grip_r, _, T = load_episode_json(ep_json)
        except Exception as e:
            print(f"[skip] {ep_dir.name}: {e}")
            continue

        # demo start pose (pos + rotvec) from first frame
        demo_l = np.concatenate([pos_l[0], rvec_l[0]], axis=0).astype(np.float32)
        demo_r = np.concatenate([pos_r[0], rvec_r[0]], axis=0).astype(np.float32)

        n = min(T - 1, vfc)  # drop last frame (action = i+1)
        dataset._video_frame_count = n
        for i in range(n):
            j = i + 1  # next-frame absolute target
            left_action = np.concatenate([pos_l[j], rvec_l[j], grip_l[j]], axis=0).astype(np.float32)
            right_action = np.concatenate([pos_r[j], rvec_r[j], grip_r[j]], axis=0).astype(np.float32)
            actions = np.concatenate([left_action, right_action], axis=0).astype(np.float32)
            dataset.add_frame({
                "face_view": dummy_face,
                "left_wrist_view": dummy_left,
                "right_wrist_view": dummy_right,
                "follow_left_pos": pos_l[i],
                "follow_left_rotvec": rvec_l[i],
                "follow_left_gripper": grip_l[i],
                "follow_right_pos": pos_r[i],
                "follow_right_rotvec": rvec_r[i],
                "follow_right_gripper": grip_r[i],
                "demo_start_pose_left": demo_l,
                "demo_start_pose_right": demo_r,
                "left_action": left_action,
                "right_action": right_action,
                "actions": actions,
                "task": args.task,
            })
        dataset.save_episode()
        total_eps += 1
        total_frames += n
    print(f"[INFO] build done in {time.time()-t0:.1f}s: {total_eps} eps, {total_frames} frames")

    # ---------- Phase 3: finalize ----------
    print("\n[INFO] finalize video info ...")
    dataset.finalize_video_info()
    img_dir = output_path / "images"
    if img_dir.is_dir():
        shutil.rmtree(img_dir)

    print(f"\nDONE -> {output_path}")
    print(f"Episodes: {total_eps}  Frames: {total_frames}")


if __name__ == "__main__":
    main()
