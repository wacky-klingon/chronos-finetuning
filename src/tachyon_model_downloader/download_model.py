from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml
from huggingface_hub import snapshot_download

from .schemas import DownloadState, ModelDownloadConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download a Hugging Face base model using YAML config."
    )
    parser.add_argument(
        "--config",
        required=True,
        help="Path to YAML config file.",
    )
    return parser.parse_args()


def load_yaml_config(config_path: Path) -> ModelDownloadConfig:
    if not config_path.exists():
        raise FileNotFoundError(f"Config file does not exist: {config_path}")

    with config_path.open("r", encoding="utf-8") as file_handle:
        raw: dict[str, Any] = yaml.safe_load(file_handle) or {}

    return ModelDownloadConfig(**raw)


def _config_json_path(output_dir: Path) -> Path:
    return output_dir / "config.json"


def _state_path(output_dir: Path) -> Path:
    return output_dir / "download_state.json"


def _write_state(
    output_dir: Path,
    repo_id: str,
    revision: str | None,
    force_download_used: bool,
) -> None:
    state = DownloadState(
        repo_id=repo_id,
        revision=revision,
        output_dir=str(output_dir),
        downloaded_at_utc=datetime.now(timezone.utc).isoformat(),
        force_download_used=force_download_used,
    )
    state_file = _state_path(output_dir)
    with state_file.open("w", encoding="utf-8") as file_handle:
        json.dump(state.model_dump(), file_handle, indent=2)


def _should_skip_download(output_dir: Path, force_download: bool) -> bool:
    return _config_json_path(output_dir).exists() and not force_download


def download_from_hf(config: ModelDownloadConfig, project_root: Path) -> Path:
    output_dir = (project_root / config.download.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if _should_skip_download(output_dir=output_dir, force_download=config.download.force_download):
        _write_state(
            output_dir=output_dir,
            repo_id=config.model.repo_id,
            revision=config.model.revision,
            force_download_used=config.download.force_download,
        )
        return output_dir

    snapshot_download(
        repo_id=config.model.repo_id,
        revision=config.model.revision,
        local_dir=str(output_dir),
        allow_patterns=config.model.allow_patterns,
        ignore_patterns=config.model.ignore_patterns,
        local_dir_use_symlinks=config.download.local_dir_use_symlinks,
        force_download=config.download.force_download,
    )

    if not _config_json_path(output_dir).exists():
        raise RuntimeError("Download completed but config.json was not found in output directory.")

    _write_state(
        output_dir=output_dir,
        repo_id=config.model.repo_id,
        revision=config.model.revision,
        force_download_used=config.download.force_download,
    )
    return output_dir


def main() -> None:
    args = parse_args()
    config_path = Path(args.config).resolve()
    config = load_yaml_config(config_path=config_path)
    project_root = config_path.parent.parent
    downloaded_to = download_from_hf(config=config, project_root=project_root)
    print(f"Model ready at: {downloaded_to}")


if __name__ == "__main__":
    main()
