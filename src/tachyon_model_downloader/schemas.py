from __future__ import annotations

from pydantic import BaseModel, Field


class ModelConfig(BaseModel):
    repo_id: str = Field(min_length=1)
    revision: str | None = None
    allow_patterns: list[str] | None = None
    ignore_patterns: list[str] | None = None


class DownloadConfig(BaseModel):
    output_dir: str = Field(min_length=1)
    force_download: bool = False
    local_dir_use_symlinks: bool = False


class ModelDownloadConfig(BaseModel):
    model: ModelConfig
    download: DownloadConfig


class DownloadState(BaseModel):
    repo_id: str
    revision: str | None = None
    output_dir: str
    downloaded_at_utc: str
    force_download_used: bool
