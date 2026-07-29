#!/usr/bin/env python3
"""
离线计算每帧的自适应加速因子（DTW + rel 模式）。

对每条 episode 的每一帧:
  1. 提取双臂末端位置轨迹（master 或 follow, 6D: 左3D + 右3D）
  2. 对每个 scale_factor in [1.0, 1.1, ..., max_factor]:
     - 取参考片段 (chunk_size) 和扩展片段 (L = chunk_size * factor)
     - 用 DTW 计算两段之间的距离
  3. 每帧最优因子:
     - rel 模式: distance < dynamic_threshold 的最大 factor
       dynamic_threshold = dist[first] + ratio * (dist[last] - dist[first])
  4. 中值滤波平滑
  5. 写入 episode 目录: factor/c{chunk_size}_{ratio}/adaptive_factor.json

示例:
  python compute_factors.py \
    --data_root /mnt/public3/datasets/x1pro/table_clean_sop_0720_v2 \
    --chunk_size 20 \
    --ratio 0.4 \
    --action_source master
"""

import argparse
import json
import os
import sys
import numpy as np
from multiprocessing import Pool, cpu_count

from tqdm import tqdm
from dtaidistance import dtw_ndim

from utils import extract_action_chunk, compute_mse_distance, apply_filter, save_factors_json


def extract_trajectory(data, action_source):
    action_fields = [
        f"{action_source}_left_position",
        f"{action_source}_right_position",
    ]

    positions_all = []
    for frame in data["data"]:
        pos = []
        for field in action_fields:
            pos.extend(frame[field])
        positions_all.append(pos)

    return np.array(positions_all, dtype=np.float32), data["fps"], action_fields


def compute_frame_distance(trajectory, frame_idx, chunk_size, scale_factor, dtw_window):
    total = trajectory.shape[0]
    available = total - frame_idx

    if available <= chunk_size:
        return float("inf")

    L = min(int(chunk_size * scale_factor), available)
    if L <= chunk_size:
        return 0.0

    reference_chunk = extract_action_chunk(trajectory, frame_idx, chunk_size, pad_last_frame=True)
    extended_chunk = extract_action_chunk(trajectory, frame_idx, L, pad_last_frame=True)

    return dtw_ndim.distance(reference_chunk, extended_chunk, window=dtw_window)


def _worker_compute(args_tuple):
    (
        episode_dir,
        chunk_size,
        scale_factors,
        ratio,
        dtw_window,
        filter_window,
        frame_stride,
        action_source,
    ) = args_tuple

    json_path = os.path.join(episode_dir, os.path.basename(episode_dir) + ".json")

    with open(json_path) as f:
        raw_data = json.load(f)

    trajectory, fps, action_fields = extract_trajectory(raw_data, action_source)
    episode_len = trajectory.shape[0]

    frame_indices_full = np.arange(episode_len)
    actual_stride = max(1, min(frame_stride, episode_len // 10 + 1))

    indices_to_compute = frame_indices_full if actual_stride == 1 else frame_indices_full[::actual_stride]
    n_compute = len(indices_to_compute)
    optimal_factors = np.ones(n_compute, dtype=np.float64)

    for si, t in enumerate(indices_to_compute):
        dists = []
        for sf in scale_factors:
            dists.append(compute_frame_distance(trajectory, t, chunk_size, sf, dtw_window))
        dists = np.array(dists)
        if np.isinf(dists).all():
            optimal_factors[si] = 1.0
            continue
        dynamic_thresh = dists[0] + ratio * (dists[-1] - dists[0])
        best_factor = 1.0
        for i, (sf, d) in enumerate(zip(scale_factors, dists)):
            if d <= dynamic_thresh:
                best_factor = sf
            else:
                break
        optimal_factors[si] = best_factor

    if actual_stride != 1:
        optimal_factors = np.interp(frame_indices_full, indices_to_compute, optimal_factors)

    filtered_factors = apply_filter(optimal_factors, filter_window)

    out_path = save_factors_json(
        episode_dir, filtered_factors, chunk_size, ratio, fps,
        action_source, action_fields,
    )

    return os.path.basename(episode_dir), episode_len, float(np.mean(filtered_factors)), out_path


def main():
    parser = argparse.ArgumentParser(description="Compute per-frame adaptive acceleration factors (DTW + rel)")
    parser.add_argument("--data_root", type=str, required=True,
                        help="Root directory of raw episode data")
    parser.add_argument("--chunk_size", type=int, required=True,
                        help="Action chunk length (same as model action_horizon)")
    parser.add_argument("--ratio", type=float, required=True,
                        help="Relative ratio for dynamic threshold (e.g. 0.4)")
    parser.add_argument("--action_source", type=str, choices=["master", "follow"], default="master",
                        help="Which arm to use for trajectory: master or follow")
    parser.add_argument("--max_factor", type=float, default=5.0,
                        help="Maximum scale factor to search")
    parser.add_argument("--scale_step", type=float, default=0.1,
                        help="Scale factor step size")
    parser.add_argument("--filter_window", type=int, default=9,
                        help="Median filter window size")
    parser.add_argument("--dtw_window", type=int, default=5,
                        help="DTW Sakoe-Chiba window (0=None)")
    parser.add_argument("--frame_stride", type=int, default=1,
                        help="Analyze every N frames for speed (factors will be interpolated back)")
    parser.add_argument("--num_workers", type=int, default=0,
                        help="Number of parallel workers (0=auto detect)")
    parser.add_argument("--episodes", type=int, nargs="+", default=None,
                        help="Specific episode indices to process (0-based, in sorted order)")

    args = parser.parse_args()

    if not os.path.isdir(args.data_root):
        raise RuntimeError(f"Data root not found: {args.data_root}")

    ep_dirs = sorted([
        os.path.join(args.data_root, d)
        for d in os.listdir(args.data_root)
        if os.path.isdir(os.path.join(args.data_root, d)) and not d.startswith(".")
    ])

    if args.episodes is not None:
        ep_dirs = [ep_dirs[i] for i in args.episodes if i < len(ep_dirs)]

    for ep_dir in ep_dirs:
        json_path = os.path.join(ep_dir, os.path.basename(ep_dir) + ".json")
        if not os.path.exists(json_path):
            print(f"Warning: No JSON found for {os.path.basename(ep_dir)}, skipping.")
            ep_dirs.remove(ep_dir)

    scale_factors = np.arange(1.0, args.max_factor + args.scale_step / 2, args.scale_step)
    scale_factors = [round(float(sf), 2) for sf in scale_factors]

    n_workers = args.num_workers if args.num_workers > 0 else min(cpu_count(), len(ep_dirs))
    n_workers = max(1, n_workers)

    dtw_window = args.dtw_window if args.dtw_window > 0 else None

    ratio_str = str(args.ratio).replace(".", "_")
    dir_name = f"c{args.chunk_size}_{ratio_str}"

    print(f"Data root: {args.data_root}")
    print(f"Action source: {args.action_source}")
    print(f"Chunk size: {args.chunk_size}")
    print(f"Ratio: {args.ratio}")
    print(f"Factor output: factor/{dir_name}/adaptive_factor.json")
    print(f"Scale factors: {scale_factors}")
    print(f"DTW window: {dtw_window}")
    print(f"Episodes: {len(ep_dirs)}")
    print(f"Workers: {n_workers} (CPU cores: {cpu_count()})")
    print()

    worker_args = [
        (
            ep_dir,
            args.chunk_size,
            scale_factors,
            args.ratio,
            dtw_window,
            args.filter_window,
            args.frame_stride,
            args.action_source,
        )
        for ep_dir in ep_dirs
    ]

    pool = Pool(processes=n_workers)
    results = []
    try:
        results = list(tqdm(
            pool.imap_unordered(_worker_compute, worker_args),
            total=len(worker_args),
            desc="Episodes",
            smoothing=0.05,
        ))
    except KeyboardInterrupt:
        print("\nInterrupted, terminating workers...")
        pool.terminate()
        pool.join()
        sys.exit(1)
    finally:
        pool.terminate()
        pool.join()

    results.sort(key=lambda x: x[0])

    print(f"\n{'='*70}")
    print(f"{'Episode':>40} {'Frames':>8} {'Mean':>10}")
    print(f"{'-'*70}")

    all_means = []
    for ep_name, ep_len, mean_factor, out_path in results:
        all_means.append(mean_factor)
        print(f"{ep_name:>40} {ep_len:>8} {mean_factor:>10.3f}")

    print(f"{'-'*70}")
    print(f"{'ALL':>40} {'':>8} {np.mean(all_means):>10.3f}")
    print(f"{'='*70}")
    print(f"\nWrote {len(results)} factor JSON files under: {args.data_root}")


if __name__ == "__main__":
    main()
