"""Shared parquet loading utilities used by inference and fine-tuning paths."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .dataset_schema import DatasetRoles


def load_parquet_dataframe(parquet_path: Path) -> pd.DataFrame:
    """Load a parquet file into a pandas DataFrame.

    Raises:
        FileNotFoundError: if ``parquet_path`` does not exist.
        ValueError: if the path does not have a ``.parquet`` extension.
    """
    if not parquet_path.exists():
        raise FileNotFoundError(f"Parquet file not found: {parquet_path}")
    if parquet_path.suffix.lower() != ".parquet":
        raise ValueError(f"Only .parquet files are supported in this pipeline. Got: {parquet_path}")
    return pd.read_parquet(parquet_path)


def validate_required_columns(frame: pd.DataFrame, roles: DatasetRoles, source_label: str) -> None:
    """Ensure ``frame`` contains the required role columns."""
    missing = roles.required_columns().difference(frame.columns)
    if missing:
        raise ValueError(f"{source_label} parquet missing required role columns: {sorted(missing)}")


def coerce_timestamp_column(frame: pd.DataFrame, roles: DatasetRoles) -> pd.DataFrame:
    """Coerce the timestamp column to pandas ``datetime64[ns]``; raise on failure."""
    out = frame.copy()
    out[roles.timestamp_column] = pd.to_datetime(out[roles.timestamp_column], errors="raise")
    return out


def load_dataset_with_roles(
    parquet_path: Path, roles: DatasetRoles, source_label: str
) -> pd.DataFrame:
    """Load a parquet file, validate role columns exist, and coerce timestamps."""
    frame = load_parquet_dataframe(parquet_path=parquet_path)
    validate_required_columns(frame=frame, roles=roles, source_label=source_label)
    return coerce_timestamp_column(frame=frame, roles=roles)
