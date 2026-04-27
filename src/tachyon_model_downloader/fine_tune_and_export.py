from __future__ import annotations

import json
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Any

import pandas as pd
import yaml
from autogluon.timeseries import TimeSeriesDataFrame, TimeSeriesPredictor
from pydantic import BaseModel, Field


class DataConfig(BaseModel):
    train_csv: str = Field(min_length=1)
    val_csv: str = Field(min_length=1)
    item_id_column: str = Field(min_length=1)
    timestamp_column: str = Field(min_length=1)
    target_column: str = Field(min_length=1)


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
    data: DataConfig
    training: TrainingConfig
    paths: PathConfig


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
    config: FineTuneConfig, project_root: Path
) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_csv = resolve_path(project_root, config.data.train_csv)
    val_csv = resolve_path(project_root, config.data.val_csv)

    if not train_csv.exists():
        raise FileNotFoundError(f"Training CSV not found: {train_csv}")
    if not val_csv.exists():
        raise FileNotFoundError(f"Validation CSV not found: {val_csv}")

    train_df = pd.read_csv(train_csv)
    val_df = pd.read_csv(val_csv)

    required_columns = {
        config.data.item_id_column,
        config.data.timestamp_column,
        config.data.target_column,
    }
    missing_train = required_columns.difference(train_df.columns)
    missing_val = required_columns.difference(val_df.columns)
    if missing_train:
        raise ValueError(f"Training CSV missing required columns: {sorted(missing_train)}")
    if missing_val:
        raise ValueError(f"Validation CSV missing required columns: {sorted(missing_val)}")

    train_df[config.data.timestamp_column] = pd.to_datetime(
        train_df[config.data.timestamp_column], errors="raise"
    )
    val_df[config.data.timestamp_column] = pd.to_datetime(
        val_df[config.data.timestamp_column], errors="raise"
    )

    return train_df, val_df


def to_timeseries_dataframe(config: FineTuneConfig, frame: pd.DataFrame) -> TimeSeriesDataFrame:
    subset = frame[
        [
            config.data.item_id_column,
            config.data.timestamp_column,
            config.data.target_column,
        ]
    ].copy()

    return TimeSeriesDataFrame.from_data_frame(
        subset,
        id_column=config.data.item_id_column,
        timestamp_column=config.data.timestamp_column,
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

        # Traverse common object containers first.
        if isinstance(current, dict):
            queue.extend(current.values())
            continue
        if isinstance(current, (list, tuple, set)):
            queue.extend(current)
            continue

        # Traverse likely internal model attributes.
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


def export_finetuned_safetensors(predictor_dir: Path, export_dir: Path) -> dict[str, Any]:
    export_dir.mkdir(parents=True, exist_ok=True)
    predictor = TimeSeriesPredictor.load(str(predictor_dir))

    export_result: dict[str, Any] = {
        "success": False,
        "message": "",
        "exported_model_class": None,
    }

    model_obj = _find_exportable_model(predictor)
    if model_obj is None:
        export_result["message"] = (
            "Unable to locate an exportable Hugging Face model object from AutoGluon "
            "predictor internals. This AutoGluon/Chronos version may not expose a "
            "direct save_pretrained-compatible module."
        )
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
            export_result["message"] = (
                f"Export failed while calling save_pretrained: {export_exc}"
            )

    export_manifest = {
        "exported_at_utc": datetime.now(timezone.utc).isoformat(),
        "predictor_dir": str(predictor_dir),
        "export_dir": str(export_dir),
        "export_success": export_result["success"],
        "export_message": export_result["message"],
        "exported_model_class": export_result["exported_model_class"],
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
    )

    with report_path.open("w", encoding="utf-8") as file_handle:
        json.dump(metrics.model_dump(), file_handle, indent=2)


def write_export_state(
    config: FineTuneConfig,
    predictor_dir: Path,
    export_dir: Path,
    export_success: bool,
    export_message: str,
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

    data_start = perf_counter()
    train_df, val_df = load_training_data(config=config, project_root=project_root)
    train_ts = to_timeseries_dataframe(config=config, frame=train_df)
    val_ts = to_timeseries_dataframe(config=config, frame=val_df)
    data_seconds = perf_counter() - data_start

    predictor = TimeSeriesPredictor(
        prediction_length=config.training.prediction_length,
        target=config.data.target_column,
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
    export_result = export_finetuned_safetensors(
        predictor_dir=predictor_dir, export_dir=export_dir
    )
    export_seconds = perf_counter() - export_start
    write_export_state(
        config=config,
        predictor_dir=predictor_dir,
        export_dir=export_dir,
        export_success=bool(export_result["success"]),
        export_message=str(export_result["message"]),
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
    )
    return predictor_dir, export_dir


def main() -> None:
    project_root = Path(__file__).resolve().parents[2]
    config_path = project_root / "config" / "fine_tune.yaml"
    config = load_config(config_path=config_path)
    predictor_dir, export_dir = fine_tune_and_export(config=config, project_root=project_root)
    print(f"Predictor bundle: {predictor_dir}")
    print(f"Standalone export target: {export_dir}")
    print(f"Export metadata: {export_dir / 'export_manifest.json'}")
    print(f"Training metadata: {predictor_dir / 'training_state.json'}")


if __name__ == "__main__":
    main()
