import json
import numpy as np
from pathlib import Path


def extract_action_chunk(actions, start_idx, length, pad_last_frame=True):
    total_frames = len(actions)
    actual_end = min(start_idx + length, total_frames)
    chunk = actions[start_idx:actual_end].astype(np.float32)

    if len(chunk) < length and pad_last_frame:
        pad_len = length - len(chunk)
        last_frame = actions[-1:].astype(np.float32)
        chunk = np.concatenate([chunk, np.repeat(last_frame, pad_len, axis=0)], axis=0)

    return chunk


def resample_linear(actions, original_length, target_length):
    if original_length == target_length:
        return actions.copy()
    original_indices = np.linspace(0, original_length - 1, original_length)
    target_indices = np.linspace(0, original_length - 1, target_length)
    resampled = np.zeros((target_length, actions.shape[1]), dtype=np.float32)
    for dim in range(actions.shape[1]):
        resampled[:, dim] = np.interp(target_indices, original_indices, actions[:, dim])
    return resampled


def compute_mse_distance(reference_chunk, extended_chunk, chunk_size):
    from scipy.interpolate import interp1d
    L = len(extended_chunk)
    if L < 2:
        repeated = np.repeat(extended_chunk, chunk_size, axis=0)[:chunk_size]
        return np.mean((reference_chunk - repeated) ** 2)
    x_old = np.linspace(0, 1, L)
    x_new = np.linspace(0, 1, chunk_size)
    resampled = np.array([
        interp1d(x_old, extended_chunk[:, d], kind="linear")(x_new)
        for d in range(extended_chunk.shape[1])
    ]).T
    return np.mean((reference_chunk - resampled) ** 2)


def apply_filter(optimal_factors, window_size):
    from scipy.signal import medfilt
    n = len(optimal_factors)
    actual_window = min(window_size, n)
    if actual_window % 2 == 0:
        actual_window += 1
    return medfilt(optimal_factors, kernel_size=actual_window)


def load_factors_json(episode_dir, action_horizon, ratio):
    ratio_str = str(ratio).replace(".", "_")
    dir_name = f"c{action_horizon}_{ratio_str}"
    fpath = Path(episode_dir) / "factor" / dir_name / "adaptive_factor.json"
    if fpath.exists():
        with open(fpath) as f:
            return json.load(f)["factors"]
    return None


def save_factors_json(episode_dir, factors, action_horizon, ratio, fps, action_source, action_fields):
    ratio_str = str(ratio).replace(".", "_")
    dir_name = f"c{action_horizon}_{ratio_str}"
    out_dir = Path(episode_dir) / "factor" / dir_name
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "adaptive_factor.json"
    with open(out_path, "w") as f:
        json.dump({
            "factors": [round(float(x), 2) for x in np.asarray(factors).tolist()],
            "fps": fps,
            "params": {
                "chunk_size": action_horizon,
                "mode": "dtw",
                "threshold_mode": "rel",
                "ratio": ratio,
                "action_source": action_source,
                "action_fields": action_fields,
            }
        }, f)
    return str(out_path)
