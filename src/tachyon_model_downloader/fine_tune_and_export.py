"""Fine-tune a Chronos model with AutoGluon and export safetensors-only artifacts.

Inputs are parquet files; dataset roles are resolved using the shared schema
source. The exported model directory is validated to comply with the
safetensors-only offline mode policy.
"""

from __future__ import annotations

import argparse
import json
import shutil
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter, sleep
from typing import Any

import pandas as pd
import yaml
from autogluon.timeseries import TimeSeriesDataFrame, TimeSeriesPredictor
from pydantic import BaseModel, ConfigDict, Field

from .dataset_schema import DatasetRoles, SchemaSourceConfig, resolve_dataset_roles
from .model_validation import (
    FORBIDDEN_ARTIFACT_SUFFIXES,
    REQUIRED_MODEL_FILES,
    validate_safetensors_only_model_dir,
)
from .parquet_loader import load_dataset_with_roles


class DataConfig(BaseModel):
    train_parquet: str = Field(min_length=1)
    val_parquet: str = Field(min_length=1)


class TrainingConfig(BaseModel):
    prediction_length: int = Field(gt=0)
    eval_metric: str = Field(min_length=1)
    presets: str = Field(min_length=1)
    chronos_model_path: str = Field(min_length=1)
    context_length: int = Field(gt=0)
    batch_size: int = Field(gt=0)
    max_epochs: int = Field(gt=0)
    device: str = Field(min_length=1)
    fine_tune: bool = True
    fine_tune_lr: float = Field(gt=0.0)
    fine_tune_steps: int = Field(gt=0)
    fine_tune_batch_size: int = Field(gt=0)


class PathConfig(BaseModel):
    predictor_dir: str = Field(min_length=1)
    export_dir: str = Field(min_length=1)


class FineTuneConfig(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    data: DataConfig
    training: TrainingConfig
    paths: PathConfig
    schema_source: SchemaSourceConfig = Field(alias="schema")


class ExportState(BaseModel):
    created_at_utc: str
    predictor_dir: str
    export_dir: str
    prediction_length: int
    eval_metric: str
    presets: str
    chronos_model_path: str
    context_length: int
    batch_size: int
    max_epochs: int
    device: str
    fine_tune: bool
    fine_tune_lr: float
    fine_tune_steps: int
    fine_tune_batch_size: int
    export_success: bool
    export_message: str
    export_safetensors_compliant: bool


class MetricsReport(BaseModel):
    created_at_utc: str
    total_time_seconds: float
    data_loading_seconds: float
    training_seconds: float
    export_seconds: float
    train_rows: int
    val_rows: int
    prediction_length: int
    predictor_file_count: int
    predictor_total_bytes: int
    export_file_count: int
    export_total_bytes: int
    exported_model_safetensors_exists: bool
    export_success: bool
    export_message: str
    export_safetensors_compliant: bool


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fine-tune a Chronos model with AutoGluon and export safetensors artifacts."
    )
    parser.add_argument(
        "--config",
        required=False,
        default=None,
        help="Path to fine_tune YAML config (default: config/fine_tune.yaml).",
    )
    return parser.parse_args()


def load_config(config_path: Path) -> FineTuneConfig:
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with config_path.open("r", encoding="utf-8") as file_handle:
        raw: dict[str, Any] = yaml.safe_load(file_handle) or {}
    return FineTuneConfig(**raw)


def resolve_path(project_root: Path, relative_or_abs: str) -> Path:
    candidate = Path(relative_or_abs)
    if candidate.is_absolute():
        return candidate
    return (project_root / candidate).resolve()


def load_training_data(
    config: FineTuneConfig, project_root: Path, roles: DatasetRoles
) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_path = resolve_path(project_root, config.data.train_parquet)
    val_path = resolve_path(project_root, config.data.val_parquet)
    train_df = load_dataset_with_roles(
        parquet_path=train_path, roles=roles, source_label="training"
    )
    val_df = load_dataset_with_roles(parquet_path=val_path, roles=roles, source_label="validation")
    return train_df, val_df


def to_timeseries_dataframe(roles: DatasetRoles, frame: pd.DataFrame) -> TimeSeriesDataFrame:
    subset = frame[[roles.item_id_column, roles.timestamp_column, roles.target_column]].copy()
    return TimeSeriesDataFrame.from_data_frame(
        subset,
        id_column=roles.item_id_column,
        timestamp_column=roles.timestamp_column,
    )


def resolve_chronos_model_path(config: FineTuneConfig, project_root: Path) -> Path:
    model_path = resolve_path(project_root, config.training.chronos_model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Chronos model path not found: {model_path}")
    if not model_path.is_dir():
        raise ValueError(f"Chronos model path is not a directory: {model_path}")
    config_json = model_path / "config.json"
    if not config_json.exists():
        raise FileNotFoundError(
            f"Chronos model directory missing required config.json: {config_json}"
        )
    return model_path


def _extract_by_path(root: Any, attribute_path: tuple[str, ...]) -> Any | None:
    current = root
    for part in attribute_path:
        if current is None or not hasattr(current, part):
            return None
        current = getattr(current, part)
    return current


def _iter_objects_for_export(root: Any) -> list[Any]:
    queue: deque[Any] = deque([root])
    visited: set[int] = set()
    discovered: list[Any] = []

    while queue:
        current = queue.popleft()
        current_id = id(current)
        if current_id in visited:
            continue
        visited.add(current_id)
        discovered.append(current)

        if isinstance(current, dict):
            queue.extend(current.values())
            continue
        if isinstance(current, list | tuple | set):
            queue.extend(current)
            continue

        for attr_name in (
            "model",
            "_model",
            "backbone",
            "network",
            "module",
            "pipeline",
            "tokenizer",
            "_tokenizer",
            "processor",
            "_processor",
            "trainer",
            "_trainer",
            "learner",
            "_learner",
            "models",
            "_models",
            "model_best",
            "_model_best",
        ):
            if hasattr(current, attr_name):
                queue.append(getattr(current, attr_name))

    return discovered


def _is_save_pretrained_object(candidate: Any) -> bool:
    return hasattr(candidate, "save_pretrained") and callable(candidate.save_pretrained)


def _find_exportable_model(root: Any) -> Any | None:
    search_roots: list[Any] = [root]
    for path in (
        ("_learner",),
        ("_learner", "trainer"),
        ("_learner", "trainer", "model"),
        ("_learner", "trainer", "models"),
        ("_learner", "model"),
        ("_learner", "model", "model"),
        ("_learner", "model", "backbone"),
    ):
        extracted = _extract_by_path(root, path)
        if extracted is not None:
            search_roots.append(extracted)

    for candidate_root in search_roots:
        for candidate in _iter_objects_for_export(candidate_root):
            if _is_save_pretrained_object(candidate):
                return candidate

    return None


def _find_exportable_tokenizer(root: Any) -> Any | None:
    for candidate in _iter_objects_for_export(root):
        for attr_name in ("tokenizer", "_tokenizer", "processor", "_processor"):
            if hasattr(candidate, attr_name):
                item = getattr(candidate, attr_name)
                if _is_save_pretrained_object(item):
                    return item
        if _is_save_pretrained_object(candidate) and (
            "tokenizer" in type(candidate).__name__.lower()
            or "processor" in type(candidate).__name__.lower()
        ):
            return candidate
    return None


def _purge_forbidden_artifacts(export_dir: Path) -> list[Path]:
    """Remove any pickle-based artifacts produced during export."""
    purged: list[Path] = []
    for path in export_dir.rglob("*"):
        if path.is_file() and path.suffix.lower() in FORBIDDEN_ARTIFACT_SUFFIXES:
            purged.append(path)
            path.unlink()
    return purged


def _find_finetuned_checkpoint_dir(predictor_dir: Path) -> Path | None:
    models_dir = predictor_dir / "models"
    if not models_dir.exists() or not models_dir.is_dir():
        return None
    candidates: list[Path] = []
    for candidate in models_dir.rglob("fine-tuned-ckpt"):
        if not candidate.is_dir():
            continue
        config_path = candidate / "config.json"
        safetensors_path = candidate / "model.safetensors"
        if config_path.exists() and safetensors_path.exists():
            candidates.append(candidate)
    if not candidates:
        return None
    candidates.sort(key=lambda path: path.stat().st_mtime, reverse=True)
    return candidates[0]


def _copy_checkpoint_fallback(
    predictor_dir: Path, export_dir: Path
) -> tuple[bool, str, str | None]:
    checkpoint_dir = _find_finetuned_checkpoint_dir(predictor_dir=predictor_dir)
    if checkpoint_dir is None:
        return (
            False,
            "Unable to locate an exportable model object and no fine-tuned checkpoint "
            "directory with config.json + model.safetensors was found under predictor models.",
            None,
        )

    source_config = checkpoint_dir / "config.json"
    source_safetensors = checkpoint_dir / "model.safetensors"
    target_config = export_dir / "config.json"
    target_safetensors = export_dir / "model.safetensors"

    # Buffer cross-filesystem handle visibility (WSL to Windows mount transitions).
    sleep(0.5)
    shutil.copy2(source_config, target_config)
    shutil.copy2(source_safetensors, target_safetensors)
    return (
        True,
        f"Copied fine-tuned checkpoint artifacts from {checkpoint_dir} after 500ms delay.",
        checkpoint_dir.name,
    )


def export_finetuned_safetensors(predictor_dir: Path, export_dir: Path) -> dict[str, Any]:
    export_dir.mkdir(parents=True, exist_ok=True)
    predictor = TimeSeriesPredictor.load(str(predictor_dir))

    export_result: dict[str, Any] = {
        "success": False,
        "message": "",
        "exported_model_class": None,
        "safetensors_compliant": False,
        "purged_pickle_artifacts": [],
    }

    model_obj = _find_exportable_model(predictor)
    if model_obj is None:
        fallback_success, fallback_message, fallback_source_label = _copy_checkpoint_fallback(
            predictor_dir=predictor_dir, export_dir=export_dir
        )
        export_result["success"] = fallback_success
        export_result["message"] = fallback_message
        if fallback_source_label is not None:
            export_result["exported_model_class"] = fallback_source_label
    else:
        try:
            model_obj.save_pretrained(str(export_dir), safe_serialization=True)
            tokenizer_or_processor = _find_exportable_tokenizer(predictor)
            if tokenizer_or_processor is not None:
                tokenizer_or_processor.save_pretrained(str(export_dir))
            export_result["success"] = True
            export_result["message"] = "Exported model and tokenizer/processor artifacts."
            export_result["exported_model_class"] = type(model_obj).__name__
        except Exception as export_exc:
            fallback_success, fallback_message, fallback_source_label = _copy_checkpoint_fallback(
                predictor_dir=predictor_dir, export_dir=export_dir
            )
            export_result["success"] = fallback_success
            if fallback_success:
                export_result["message"] = (
                    "save_pretrained failed; fallback copy succeeded. "
                    f"save_pretrained error: {export_exc}. {fallback_message}"
                )
                if fallback_source_label is not None:
                    export_result["exported_model_class"] = fallback_source_label
            else:
                export_result["message"] = (
                    f"Export failed while calling save_pretrained: {export_exc}. "
                    f"Fallback copy also failed: {fallback_message}"
                )

    if export_result["success"]:
        purged = _purge_forbidden_artifacts(export_dir=export_dir)
        export_result["purged_pickle_artifacts"] = [str(path) for path in purged]
        try:
            validate_safetensors_only_model_dir(model_dir=export_dir, role_label="exported")
            export_result["safetensors_compliant"] = True
        except (FileNotFoundError, NotADirectoryError, ValueError) as validation_exc:
            export_result["safetensors_compliant"] = False
            export_result["message"] = (
                f"Export wrote files but failed safetensors-only validation: {validation_exc}"
            )

    export_manifest = {
        "exported_at_utc": datetime.now(timezone.utc).isoformat(),
        "predictor_dir": str(predictor_dir),
        "export_dir": str(export_dir),
        "export_success": export_result["success"],
        "export_message": export_result["message"],
        "exported_model_class": export_result["exported_model_class"],
        "safetensors_compliant": export_result["safetensors_compliant"],
        "required_files": list(REQUIRED_MODEL_FILES),
        "forbidden_suffixes": list(FORBIDDEN_ARTIFACT_SUFFIXES),
        "purged_pickle_artifacts": export_result["purged_pickle_artifacts"],
    }
    manifest_path = export_dir / "export_manifest.json"
    with manifest_path.open("w", encoding="utf-8") as file_handle:
        json.dump(export_manifest, file_handle, indent=2)
    return export_result


def compute_directory_metrics(directory: Path) -> tuple[int, int]:
    files = [path for path in directory.rglob("*") if path.is_file()]
    total_bytes = sum(path.stat().st_size for path in files)
    return len(files), total_bytes


def write_metrics_report(
    report_path: Path,
    predictor_dir: Path,
    export_dir: Path,
    train_rows: int,
    val_rows: int,
    prediction_length: int,
    total_time_seconds: float,
    data_loading_seconds: float,
    training_seconds: float,
    export_seconds: float,
    export_success: bool,
    export_message: str,
    export_safetensors_compliant: bool,
) -> None:
    predictor_file_count, predictor_total_bytes = compute_directory_metrics(predictor_dir)
    export_file_count, export_total_bytes = compute_directory_metrics(export_dir)
    model_safetensors_path = export_dir / "model.safetensors"

    metrics = MetricsReport(
        created_at_utc=datetime.now(timezone.utc).isoformat(),
        total_time_seconds=round(total_time_seconds, 3),
        data_loading_seconds=round(data_loading_seconds, 3),
        training_seconds=round(training_seconds, 3),
        export_seconds=round(export_seconds, 3),
        train_rows=train_rows,
        val_rows=val_rows,
        prediction_length=prediction_length,
        predictor_file_count=predictor_file_count,
        predictor_total_bytes=predictor_total_bytes,
        export_file_count=export_file_count,
        export_total_bytes=export_total_bytes,
        exported_model_safetensors_exists=model_safetensors_path.exists(),
        export_success=export_success,
        export_message=export_message,
        export_safetensors_compliant=export_safetensors_compliant,
    )

    with report_path.open("w", encoding="utf-8") as file_handle:
        json.dump(metrics.model_dump(), file_handle, indent=2)


def write_export_state(
    config: FineTuneConfig,
    predictor_dir: Path,
    export_dir: Path,
    export_success: bool,
    export_message: str,
    export_safetensors_compliant: bool,
) -> None:
    state = ExportState(
        created_at_utc=datetime.now(timezone.utc).isoformat(),
        predictor_dir=str(predictor_dir),
        export_dir=str(export_dir),
        prediction_length=config.training.prediction_length,
        eval_metric=config.training.eval_metric,
        presets=config.training.presets,
        chronos_model_path=config.training.chronos_model_path,
        context_length=config.training.context_length,
        batch_size=config.training.batch_size,
        max_epochs=config.training.max_epochs,
        device=config.training.device,
        fine_tune=config.training.fine_tune,
        fine_tune_lr=config.training.fine_tune_lr,
        fine_tune_steps=config.training.fine_tune_steps,
        fine_tune_batch_size=config.training.fine_tune_batch_size,
        export_success=export_success,
        export_message=export_message,
        export_safetensors_compliant=export_safetensors_compliant,
    )
    state_path = predictor_dir / "training_state.json"
    with state_path.open("w", encoding="utf-8") as file_handle:
        json.dump(state.model_dump(), file_handle, indent=2)


def fine_tune_and_export(config: FineTuneConfig, project_root: Path) -> tuple[Path, Path]:
    run_start = perf_counter()
    predictor_dir = resolve_path(project_root, config.paths.predictor_dir)
    export_dir = resolve_path(project_root, config.paths.export_dir)
    predictor_dir.mkdir(parents=True, exist_ok=True)
    export_dir.mkdir(parents=True, exist_ok=True)
    chronos_model_path = resolve_chronos_model_path(config=config, project_root=project_root)

    train_path = resolve_path(project_root, config.data.train_parquet)
    train_roles = resolve_dataset_roles(
        parquet_path=train_path,
        schema_source=config.schema_source,
        project_root=project_root,
    )

    data_start = perf_counter()
    train_df, val_df = load_training_data(
        config=config, project_root=project_root, roles=train_roles
    )
    train_ts = to_timeseries_dataframe(roles=train_roles, frame=train_df)
    val_ts = to_timeseries_dataframe(roles=train_roles, frame=val_df)
    data_seconds = perf_counter() - data_start

    predictor = TimeSeriesPredictor(
        prediction_length=config.training.prediction_length,
        target=train_roles.target_column,
        eval_metric=config.training.eval_metric,
        path=str(predictor_dir),
    )

    model_hyperparameters: dict[str, Any] = {
        "Chronos": {
            "model_path": str(chronos_model_path),
            "context_length": config.training.context_length,
            "batch_size": config.training.batch_size,
            "max_epochs": config.training.max_epochs,
            "device": config.training.device,
            "fine_tune": config.training.fine_tune,
            "fine_tune_lr": config.training.fine_tune_lr,
            "fine_tune_steps": config.training.fine_tune_steps,
            "fine_tune_batch_size": config.training.fine_tune_batch_size,
        }
    }

    train_start = perf_counter()
    predictor.fit(
        train_data=train_ts,
        tuning_data=val_ts,
        presets=config.training.presets,
        hyperparameters=model_hyperparameters,
    )
    training_seconds = perf_counter() - train_start

    export_start = perf_counter()
    export_result = export_finetuned_safetensors(predictor_dir=predictor_dir, export_dir=export_dir)
    export_seconds = perf_counter() - export_start
    write_export_state(
        config=config,
        predictor_dir=predictor_dir,
        export_dir=export_dir,
        export_success=bool(export_result["success"]),
        export_message=str(export_result["message"]),
        export_safetensors_compliant=bool(export_result["safetensors_compliant"]),
    )

    total_seconds = perf_counter() - run_start
    metrics_report_path = predictor_dir / "metrics_report.json"
    write_metrics_report(
        report_path=metrics_report_path,
        predictor_dir=predictor_dir,
        export_dir=export_dir,
        train_rows=len(train_df),
        val_rows=len(val_df),
        prediction_length=config.training.prediction_length,
        total_time_seconds=total_seconds,
        data_loading_seconds=data_seconds,
        training_seconds=training_seconds,
        export_seconds=export_seconds,
        export_success=bool(export_result["success"]),
        export_message=str(export_result["message"]),
        export_safetensors_compliant=bool(export_result["safetensors_compliant"]),
    )
    return predictor_dir, export_dir


def main() -> None:
    args = parse_args()
    project_root = Path(__file__).resolve().parents[2]
    if args.config is not None:
        config_path = Path(args.config).resolve()
    else:
        config_path = project_root / "config" / "fine_tune.yaml"
    config = load_config(config_path=config_path)
    predictor_dir, export_dir = fine_tune_and_export(config=config, project_root=project_root)
    print(f"Predictor bundle: {predictor_dir}")
    print(f"Standalone export target: {export_dir}")
    print(f"Export metadata: {export_dir / 'export_manifest.json'}")
    print(f"Training metadata: {predictor_dir / 'training_state.json'}")


if __name__ == "__main__":
    main()
