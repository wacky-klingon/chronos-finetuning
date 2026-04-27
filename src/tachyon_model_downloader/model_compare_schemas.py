"""Pydantic config models for the model comparison pipeline."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from .dataset_schema import SchemaSourceConfig


class QueryDataConfig(BaseModel):
    history_parquet: str = Field(min_length=1)
    targets_parquet: str = Field(min_length=1)


class SamplingConfig(BaseModel):
    temperature: float = Field(gt=0.0)
    top_p: float = Field(gt=0.0, le=1.0)
    num_samples: int = Field(gt=0)
    seed: int
    prediction_length: int = Field(gt=0)
    lower_quantile: float = Field(gt=0.0, lt=1.0, default=0.1)
    median_quantile: float = Field(gt=0.0, lt=1.0, default=0.5)
    upper_quantile: float = Field(gt=0.0, lt=1.0, default=0.9)

    @model_validator(mode="after")
    def validate_quantile_ordering(self) -> SamplingConfig:
        if not (self.lower_quantile < self.median_quantile < self.upper_quantile):
            raise ValueError(
                "Quantiles must satisfy lower_quantile < median_quantile < upper_quantile."
            )
        return self


class ModelPathsConfig(BaseModel):
    base_model_dir: str = Field(min_length=1)
    finetuned_model_dir: str = Field(min_length=1)
    device: str = Field(min_length=1, default="cpu")


class OutputConfig(BaseModel):
    output_root: str = Field(min_length=1)
    run_label: str = Field(min_length=1, default="default")


class CodexHookConfig(BaseModel):
    api_base: str = Field(min_length=1, default="https://api.openai.com/v1")
    model: str = Field(min_length=1, default="gpt-4o-mini")
    api_key_env: str = Field(min_length=1, default="OPENAI_API_KEY")
    max_output_tokens: int = Field(gt=0, default=2000)
    request_timeout_seconds: int = Field(gt=0, default=60)


class AnalysisHooksConfig(BaseModel):
    enabled: bool = False
    provider: Literal["codex", "none"] = "none"
    codex: CodexHookConfig | None = None

    @model_validator(mode="after")
    def validate_codex_block_when_enabled(self) -> AnalysisHooksConfig:
        if self.enabled and self.provider == "codex" and self.codex is None:
            raise ValueError(
                "analysis_hooks.codex must be set when enabled=true and provider='codex'."
            )
        return self


class ModelCompareConfig(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    query_data: QueryDataConfig
    sampling: SamplingConfig
    models: ModelPathsConfig
    output: OutputConfig
    schema_source: SchemaSourceConfig = Field(alias="schema")
    analysis_hooks: AnalysisHooksConfig = Field(default_factory=AnalysisHooksConfig)


def resolve_path(project_root: Path, relative_or_abs: str) -> Path:
    candidate = Path(relative_or_abs)
    if candidate.is_absolute():
        return candidate
    return (project_root / candidate).resolve()


def load_model_compare_config(config_path: Path) -> ModelCompareConfig:
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with config_path.open("r", encoding="utf-8") as file_handle:
        raw: dict[str, Any] = yaml.safe_load(file_handle) or {}
    return ModelCompareConfig(**raw)
