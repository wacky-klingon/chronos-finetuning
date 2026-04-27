"""Dataset role schema modeling and resolution.

Roles describe how the columns in a parquet file map to the time-series concepts
required by Chronos workflows: target, time index, item id, and covariate roles.

Resolution priority order (deterministic, idempotent):

1. Embedded parquet schema metadata under the key ``chronos_roles``.
2. JSON schema file pointed to by ``schema.json_schema_path``.
3. Inline ``schema.role_hints`` provided in the config.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq
from pydantic import BaseModel, Field, model_validator

PARQUET_ROLES_METADATA_KEY = b"chronos_roles"


class DatasetRoles(BaseModel):
    """Role assignment for time-series columns within a parquet dataset."""

    item_id_column: str = Field(min_length=1)
    timestamp_column: str = Field(min_length=1)
    target_column: str = Field(min_length=1)
    known_covariate_columns: list[str] = Field(default_factory=list)
    past_covariate_columns: list[str] = Field(default_factory=list)
    static_covariate_columns: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_unique_columns(self) -> DatasetRoles:
        all_columns: list[str] = [
            self.item_id_column,
            self.timestamp_column,
            self.target_column,
        ]
        all_columns.extend(self.known_covariate_columns)
        all_columns.extend(self.past_covariate_columns)
        all_columns.extend(self.static_covariate_columns)
        if len(all_columns) != len(set(all_columns)):
            raise ValueError("Dataset role columns must be unique across all roles.")
        return self

    def required_columns(self) -> set[str]:
        return {self.item_id_column, self.timestamp_column, self.target_column}

    def all_columns(self) -> set[str]:
        cols: set[str] = self.required_columns()
        cols.update(self.known_covariate_columns)
        cols.update(self.past_covariate_columns)
        cols.update(self.static_covariate_columns)
        return cols


class JsonSchemaRoleHints(BaseModel):
    """JSON file shape used for fallback role hint loading."""

    roles: DatasetRoles


class SchemaSourceConfig(BaseModel):
    """Configuration controlling how dataset roles are resolved."""

    use_parquet_metadata: bool = True
    json_schema_path: str | None = None
    role_hints: DatasetRoles | None = None

    @model_validator(mode="after")
    def validate_at_least_one_source(self) -> SchemaSourceConfig:
        if (
            not self.use_parquet_metadata
            and self.json_schema_path is None
            and self.role_hints is None
        ):
            raise ValueError(
                "schema source must provide at least one of: use_parquet_metadata=true, "
                "json_schema_path, or role_hints."
            )
        return self


def read_parquet_roles_from_metadata(parquet_path: Path) -> DatasetRoles | None:
    """Return DatasetRoles from parquet schema metadata, or None if absent."""
    if not parquet_path.exists():
        raise FileNotFoundError(f"Parquet file not found: {parquet_path}")
    schema = pq.read_schema(str(parquet_path))
    metadata = schema.metadata
    if metadata is None:
        return None
    raw_value = metadata.get(PARQUET_ROLES_METADATA_KEY)
    if raw_value is None:
        return None
    try:
        decoded = json.loads(raw_value.decode("utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise ValueError(
            f"Parquet metadata key {PARQUET_ROLES_METADATA_KEY!r} is not valid UTF-8 JSON."
        ) from exc
    if not isinstance(decoded, dict):
        raise ValueError(
            f"Parquet metadata key {PARQUET_ROLES_METADATA_KEY!r} must decode to a JSON object."
        )
    return DatasetRoles(**decoded)


def load_roles_from_json_schema(json_schema_path: Path) -> DatasetRoles:
    """Load DatasetRoles from a JSON schema file."""
    if not json_schema_path.exists():
        raise FileNotFoundError(f"Role JSON schema not found: {json_schema_path}")
    with json_schema_path.open("r", encoding="utf-8") as file_handle:
        raw: dict[str, Any] = json.load(file_handle)
    return JsonSchemaRoleHints(**raw).roles


def resolve_dataset_roles(
    parquet_path: Path,
    schema_source: SchemaSourceConfig,
    project_root: Path,
) -> DatasetRoles:
    """Resolve dataset roles using the configured priority order."""
    if schema_source.use_parquet_metadata:
        roles_from_metadata = read_parquet_roles_from_metadata(parquet_path=parquet_path)
        if roles_from_metadata is not None:
            return roles_from_metadata
    if schema_source.json_schema_path is not None:
        json_path = Path(schema_source.json_schema_path)
        if not json_path.is_absolute():
            json_path = (project_root / json_path).resolve()
        return load_roles_from_json_schema(json_schema_path=json_path)
    if schema_source.role_hints is not None:
        return schema_source.role_hints
    raise ValueError(
        f"Unable to resolve dataset roles for parquet file: {parquet_path}. "
        "No role metadata embedded in parquet, and no schema.json_schema_path or "
        "schema.role_hints fallback was configured."
    )
