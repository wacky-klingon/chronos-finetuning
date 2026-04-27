from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field, model_validator


class QueryDataConfig(BaseModel):
    history_csv: str = Field(min_length=1)
    targets_csv: str = Field(min_length=1)
    item_id_column: str = Field(min_length=1)
    timestamp_column: str = Field(min_length=1)
    target_column: str = Field(min_length=1)


class SamplingConfig(BaseModel):
    temperature: float = Field(gt=0.0)
    top_p: float = Field(gt=0.0, le=1.0)
    num_samples: int = Field(gt=0)
    seed: int
    lower_quantile_column: str = Field(min_length=1, default="0.1")
    median_quantile_column: str = Field(min_length=1, default="0.5")
    upper_quantile_column: str = Field(min_length=1, default="0.9")


class ModelPathsConfig(BaseModel):
    base_predictor_dir: str = Field(min_length=1)
    finetuned_predictor_dir: str = Field(min_length=1)


class OutputConfig(BaseModel):
    output_root: str = Field(min_length=1)
    run_label: str = Field(min_length=1, default="default")


class ModelCompareConfig(BaseModel):
    query_data: QueryDataConfig
    sampling: SamplingConfig
    models: ModelPathsConfig
    output: OutputConfig

    @model_validator(mode="after")
    def validate_quantile_columns(self) -> ModelCompareConfig:
        quantile_columns = {
            self.sampling.lower_quantile_column,
            self.sampling.median_quantile_column,
            self.sampling.upper_quantile_column,
        }
        if len(quantile_columns) != 3:
            raise ValueError("Quantile column names must be unique.")
        return self


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
