#!/usr/bin/env python3
"""Write pour-tea task prompts based on the episode recording date."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path
import re

import tyro

DEFAULT_DATASET_ROOT = Path("/mnt/public/datasets/x1pro/pour_tea_training")
DEFAULT_PROMPT_RELATIVE_PATH = Path("anno/prompt.txt")
DEFAULT_STANDARD_PROMPT = "Use the right arm to pour tea into three cups."
DEFAULT_OBSERVATION_PROMPT = "Move the left arm to observe while the right arm pours tea into three cups."
DEFAULT_OBSERVATION_START_DATE = "2026-07-13"

EPISODE_DATE_PATTERN = re.compile(r"@(\d{4})_(\d{2})_(\d{2})_\d{2}_\d{2}_\d{2}")


@dataclass(frozen=True)
class EpisodeClassification:
    path: Path
    recorded_date: date
    uses_left_observation_arm: bool


def get_episode_json_path(episode_path: Path) -> Path:
    return episode_path / f"{episode_path.name}.json"


def get_episode_dirs(dataset_root: Path) -> list[Path]:
    return sorted(
        path
        for path in dataset_root.iterdir()
        if path.is_dir() and get_episode_json_path(path).is_file()
    )


def get_recorded_date(episode_path: Path) -> date:
    match = EPISODE_DATE_PATTERN.search(episode_path.name)
    if match is None:
        raise ValueError(f"cannot parse recording date from episode name: {episode_path.name}")
    return date(*(int(value) for value in match.groups()))


def classify_episode(episode_path: Path, observation_start_date: date) -> EpisodeClassification:
    recorded_date = get_recorded_date(episode_path)
    return EpisodeClassification(
        path=episode_path,
        recorded_date=recorded_date,
        uses_left_observation_arm=recorded_date >= observation_start_date,
    )


def main(
    dataset_root: Path = DEFAULT_DATASET_ROOT,
    prompt_relative_path: Path = DEFAULT_PROMPT_RELATIVE_PATH,
    standard_prompt: str = DEFAULT_STANDARD_PROMPT,
    observation_prompt: str = DEFAULT_OBSERVATION_PROMPT,
    observation_start_date: str = DEFAULT_OBSERVATION_START_DATE,
    *,
    dry_run: bool = False,
    overwrite: bool = False,
) -> None:
    if not standard_prompt.strip() or not observation_prompt.strip():
        raise ValueError("prompts must not be empty")
    if standard_prompt.strip() == observation_prompt.strip():
        raise ValueError("standard and observation prompts must differ")
    try:
        cutoff_date = date.fromisoformat(observation_start_date)
    except ValueError as exc:
        raise ValueError(f"observation_start_date must use YYYY-MM-DD format: {observation_start_date}") from exc

    episode_paths = get_episode_dirs(dataset_root)
    if not episode_paths:
        raise RuntimeError(f"No episodes found under {dataset_root}")

    classifications = [classify_episode(path, cutoff_date) for path in episode_paths]
    standard = [item for item in classifications if not item.uses_left_observation_arm]
    observation = [item for item in classifications if item.uses_left_observation_arm]
    print(f"Episodes: {len(classifications)}")
    print(f"Standard episodes before {cutoff_date.isoformat()}: {len(standard)}")
    print(f"Left-observation episodes from {cutoff_date.isoformat()}: {len(observation)}")
    print(f"Standard prompt: {standard_prompt.strip()}")
    print(f"Observation prompt: {observation_prompt.strip()}")

    if dry_run:
        return

    conflicts: list[Path] = []
    for item in classifications:
        prompt_path = item.path / prompt_relative_path
        expected_prompt = observation_prompt if item.uses_left_observation_arm else standard_prompt
        if prompt_path.is_file() and prompt_path.read_text(encoding="utf-8").strip() != expected_prompt.strip():
            conflicts.append(prompt_path)
    if conflicts and not overwrite:
        examples = "\n".join(f"  - {path}" for path in conflicts[:20])
        raise FileExistsError(
            f"Found {len(conflicts)} prompt files with different content. Pass --overwrite to replace them:\n{examples}"
        )

    written = 0
    unchanged = 0
    for item in classifications:
        prompt_path = item.path / prompt_relative_path
        prompt = observation_prompt if item.uses_left_observation_arm else standard_prompt
        content = f"{prompt.strip()}\n"
        if prompt_path.is_file() and prompt_path.read_text(encoding="utf-8") == content:
            unchanged += 1
            continue
        prompt_path.parent.mkdir(parents=True, exist_ok=True)
        prompt_path.write_text(content, encoding="utf-8")
        written += 1

    print(f"Prompt files written: {written}")
    print(f"Prompt files unchanged: {unchanged}")


if __name__ == "__main__":
    tyro.cli(main)
