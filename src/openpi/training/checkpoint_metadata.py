from __future__ import annotations

import dataclasses
import enum
import json
import logging
from pathlib import Path
from typing import Any

from etils import epath
import tyro
from wandb.sdk.lib.config_util import dict_from_config_file

from openpi.training import config as _config

SCHEMA_VERSION = 1
TRAIN_CONFIG_FILENAME = "train_config.yaml"
DATASETS_FILENAME = "datasets.json"


def save(directory: epath.Path | str, train_config: _config.TrainConfig) -> None:
    """Save the resolved train config and inference-relevant dataset metadata."""
    directory = epath.Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    (directory / TRAIN_CONFIG_FILENAME).write_text(tyro.extras.to_yaml(train_config))
    (directory / DATASETS_FILENAME).write_text(
        json.dumps(_collect_datasets_metadata(train_config), indent=2, ensure_ascii=False)
    )


def load_train_config(checkpoint_dir: epath.Path | str) -> _config.TrainConfig:
    path = epath.Path(checkpoint_dir) / "metadata" / TRAIN_CONFIG_FILENAME
    if not path.exists():
        raise FileNotFoundError(
            f"Checkpoint config not found at {path}. Backfill metadata for legacy checkpoints first."
        )
    return tyro.extras.from_yaml(_config.TrainConfig, path.read_text())


def load_datasets(checkpoint_dir: epath.Path | str) -> dict[str, Any]:
    path = epath.Path(checkpoint_dir) / "metadata" / DATASETS_FILENAME
    if not path.exists():
        raise FileNotFoundError(
            f"Checkpoint dataset metadata not found at {path}. Backfill metadata for legacy checkpoints first."
        )
    payload = json.loads(path.read_text())
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"Unsupported checkpoint metadata schema: {payload.get('schema_version')}")
    return payload


def restore_train_config_from_wandb(
    wandb_run_dir: Path | str,
    *,
    base_config_name: str | None = None,
) -> _config.TrainConfig:
    """Reconstruct a typed TrainConfig using W&B's resolved config values."""
    config_path = Path(wandb_run_dir) / "files" / "config.yaml"
    if not config_path.is_file():
        raise FileNotFoundError(f"W&B config not found: {config_path}")

    payload = dict_from_config_file(str(config_path), must_exist=True)
    if payload is None:
        raise ValueError(f"Failed to read W&B config: {config_path}")
    values = {key: value for key, value in payload.items() if key != "_wandb"}
    recorded_name = values.get("name")
    template_name = base_config_name or recorded_name
    if not template_name:
        raise ValueError("W&B config does not contain a config name; pass --base-config-name")

    try:
        template = _config.get_config(template_name)
    except ValueError as exc:
        if base_config_name is None:
            raise ValueError(
                f"Recorded config {recorded_name!r} is no longer registered; pass --base-config-name"
            ) from exc
        raise

    restored = _restore_dataclass(template, values)
    logging.info(
        "Restored config from W&B: recorded_name=%s, template=%s, repo_id=%s",
        recorded_name,
        template_name,
        getattr(restored.data, "repo_id", None),
    )
    return restored


def _collect_datasets_metadata(train_config: _config.TrainConfig) -> dict[str, Any]:
    repo_ids_value = getattr(train_config.data, "repo_id", None)
    repo_ids = [] if not repo_ids_value else [item.strip() for item in repo_ids_value.split(",") if item.strip()]
    datasets = []
    for repo_id in repo_ids:
        if repo_id == "fake":
            datasets.append({"repo_id": repo_id})
            continue
        try:
            from lerobot.common.datasets.lerobot_dataset import LeRobotDatasetMetadata

            metadata = LeRobotDatasetMetadata(repo_id)
            entry: dict[str, Any] = {
                "repo_id": repo_id,
                "fps": metadata.fps,
                "total_episodes": metadata.total_episodes,
                "features": metadata.info.get("features", {}),
            }
            phase_layout_path = metadata.root / "meta" / "key_state" / "phase_layout.json"
            if phase_layout_path.is_file():
                entry["key_state"] = json.loads(phase_layout_path.read_text(encoding="utf-8"))
            datasets.append(entry)
        except Exception as exc:
            raise RuntimeError(f"Failed to snapshot metadata for LeRobot dataset {repo_id!r}") from exc
    return {"schema_version": SCHEMA_VERSION, "datasets": datasets}


def _restore_dataclass(instance: Any, values: dict[str, Any]) -> Any:
    updates = {}
    for field in dataclasses.fields(instance):
        if field.name not in values:
            continue
        current = getattr(instance, field.name)
        restored, supported = _restore_value(current, values[field.name])
        if supported:
            updates[field.name] = restored
    return dataclasses.replace(instance, **updates)


def _restore_value(current: Any, value: Any) -> tuple[Any, bool]:
    if dataclasses.is_dataclass(current) and isinstance(value, dict):
        return _restore_dataclass(current, value), True
    if isinstance(current, enum.Enum):
        return type(current)(value), True
    if current is None:
        return value, value is None or isinstance(value, str | int | float | bool)
    if isinstance(current, bool) and isinstance(value, bool):
        return value, True
    if isinstance(current, str) and isinstance(value, str):
        return value, True
    if isinstance(current, int) and not isinstance(current, bool) and isinstance(value, int):
        return value, True
    if isinstance(current, float) and isinstance(value, int | float):
        return float(value), True
    if isinstance(current, dict) and isinstance(value, dict):
        return value, True
    if isinstance(current, tuple) and isinstance(value, tuple | list) and all(
        isinstance(item, str | int | float | bool | None) for item in value
    ):
        return tuple(value), True
    return current, False
