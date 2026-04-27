from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import pandas as pd
from pydantic import BaseModel

from .model_compare_schemas import ModelCompareConfig, load_model_compare_config, resolve_path

NORMAL_Z_Q10_TO_Q90 = 2.5631031311
MIN_STD = 1e-6


class QueryInferenceRecord(BaseModel):
    query_id: str
    model_role: str
    model_id: str
    item_id: str
    matched_points: int
    average_logprob: float
    average_abs_error: float
    latency_seconds: float
    temperature: float
    top_p: float
    num_samples: int
    seed: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run probabilistic inference for one model role and emit JSONL records."
    )
    parser.add_argument(
        "--config",
        required=True,
        help="Path to model_compare YAML config.",
    )
    parser.add_argument(
        "--model-role",
        required=True,
        choices=["base", "finetuned"],
        help="Model role to run.",
    )
    parser.add_argument(
        "--output-file",
        required=True,
        help="Target JSONL output file path.",
    )
    return parser.parse_args()


def _normal_logprob(observed: float, mean: float, std_dev: float) -> float:
    variance = std_dev * std_dev
    return -0.5 * math.log(2.0 * math.pi * variance) - ((observed - mean) ** 2) / (2.0 * variance)


def _select_model_dir(config: ModelCompareConfig, project_root: Path, model_role: str) -> Path:
    if model_role == "base":
        model_dir = resolve_path(project_root, config.models.base_predictor_dir)
    else:
        model_dir = resolve_path(project_root, config.models.finetuned_predictor_dir)
    if not model_dir.exists():
        raise FileNotFoundError(f"Model directory not found: {model_dir}")
    if not model_dir.is_dir():
        raise ValueError(f"Model path is not a directory: {model_dir}")
    required_files = [model_dir / "config.json", model_dir / "model.safetensors"]
    missing_files = [str(path) for path in required_files if not path.exists()]
    if missing_files:
        raise FileNotFoundError(
            "Model directory is missing required offline files: " + ", ".join(missing_files)
        )
    return model_dir


def _load_chronos_pipeline(model_dir: Path) -> Any:
    try:
        from chronos import BaseChronosPipeline
    except ImportError as exc:
        raise RuntimeError(
            "Chronos inference library is required for offline safetensors inference. "
            "Install it with: poetry add chronos-forecasting"
        ) from exc
    return BaseChronosPipeline.from_pretrained(str(model_dir), device_map="cpu")


def _to_forecast_array(forecast_output: Any, quantile_count_hint: int | None = None) -> np.ndarray:
    raw = forecast_output
    if hasattr(raw, "detach"):
        raw = raw.detach()
    if hasattr(raw, "cpu"):
        raw = raw.cpu()
    if hasattr(raw, "numpy"):
        raw = raw.numpy()
    forecast = np.asarray(raw, dtype=float)
    if forecast.ndim == 3:
        forecast = forecast[0]
    if forecast.ndim != 2:
        raise ValueError(
            f"Unexpected forecast output shape: {forecast.shape}. Expected [quantiles, prediction_length]."
        )
    if quantile_count_hint is not None and forecast.shape[0] != quantile_count_hint:
        if forecast.shape[1] == quantile_count_hint:
            forecast = forecast.T
    return forecast


def _parse_quantile(value: str) -> float:
    try:
        return float(value)
    except ValueError as exc:
        raise ValueError(
            f"Quantile column value '{value}' is not numeric. Expected values like '0.1', '0.5', '0.9'."
        ) from exc


def _resolve_quantiles(
    config: ModelCompareConfig, pipeline: Any, forecast: np.ndarray
) -> list[float]:
    if hasattr(pipeline, "quantiles"):
        quantiles_attr = pipeline.quantiles
        quantiles = [float(value) for value in quantiles_attr]
    else:
        quantiles = [
            _parse_quantile(config.sampling.lower_quantile_column),
            _parse_quantile(config.sampling.median_quantile_column),
            _parse_quantile(config.sampling.upper_quantile_column),
        ]
    if len(quantiles) != forecast.shape[0]:
        raise ValueError(
            f"Quantile count mismatch. Pipeline returned {forecast.shape[0]} quantile rows, "
            f"but quantile metadata has {len(quantiles)} values."
        )
    return quantiles


def _find_quantile_index(quantiles: list[float], target: float) -> int:
    for index, value in enumerate(quantiles):
        if abs(value - target) < 1e-9:
            return index
    raise ValueError(
        f"Requested quantile {target} is not present in model output quantiles: {quantiles}"
    )


def _load_query_frames(
    config: ModelCompareConfig, project_root: Path
) -> tuple[pd.DataFrame, pd.DataFrame]:
    history_csv = resolve_path(project_root, config.query_data.history_csv)
    targets_csv = resolve_path(project_root, config.query_data.targets_csv)
    if not history_csv.exists():
        raise FileNotFoundError(f"History CSV not found: {history_csv}")
    if not targets_csv.exists():
        raise FileNotFoundError(f"Targets CSV not found: {targets_csv}")

    history_df = pd.read_csv(history_csv)
    targets_df = pd.read_csv(targets_csv)
    item_id_column = config.query_data.item_id_column
    timestamp_column = config.query_data.timestamp_column
    target_column = config.query_data.target_column
    required_columns = {item_id_column, timestamp_column, target_column}
    missing_history = required_columns.difference(history_df.columns)
    missing_targets = required_columns.difference(targets_df.columns)
    if missing_history:
        raise ValueError(f"History CSV missing required columns: {sorted(missing_history)}")
    if missing_targets:
        raise ValueError(f"Targets CSV missing required columns: {sorted(missing_targets)}")

    history_df[timestamp_column] = pd.to_datetime(history_df[timestamp_column], errors="raise")
    targets_df[timestamp_column] = pd.to_datetime(targets_df[timestamp_column], errors="raise")
    return history_df, targets_df


def run_inference(
    config: ModelCompareConfig,
    project_root: Path,
    model_role: str,
    output_file: Path,
) -> int:
    model_dir = _select_model_dir(config=config, project_root=project_root, model_role=model_role)
    history_df, targets_df = _load_query_frames(config=config, project_root=project_root)
    item_id_column = config.query_data.item_id_column
    timestamp_column = config.query_data.timestamp_column
    target_column = config.query_data.target_column

    pipeline = _load_chronos_pipeline(model_dir=model_dir)
    unique_items = sorted(targets_df[item_id_column].astype(str).unique().tolist())
    output_file.parent.mkdir(parents=True, exist_ok=True)
    records_written = 0

    with output_file.open("w", encoding="utf-8") as file_handle:
        for item_id in unique_items:
            item_history = (
                history_df[history_df[item_id_column].astype(str) == item_id]
                .copy()
                .sort_values(by=timestamp_column)
            )
            item_targets = (
                targets_df[targets_df[item_id_column].astype(str) == item_id]
                .copy()
                .sort_values(by=timestamp_column)
            )
            if item_history.empty or item_targets.empty:
                continue

            context_values = item_history[target_column].astype(float).tolist()
            if not context_values:
                continue
            prediction_length = int(len(item_targets))
            if prediction_length <= 0:
                continue

            predict_start = perf_counter()
            forecast_output = pipeline.predict(
                context=context_values,
                prediction_length=prediction_length,
            )
            latency_seconds = perf_counter() - predict_start
            forecast = _to_forecast_array(forecast_output=forecast_output)
            quantiles = _resolve_quantiles(config=config, pipeline=pipeline, forecast=forecast)
            lower_q = _parse_quantile(config.sampling.lower_quantile_column)
            median_q = _parse_quantile(config.sampling.median_quantile_column)
            upper_q = _parse_quantile(config.sampling.upper_quantile_column)
            lower_idx = _find_quantile_index(quantiles=quantiles, target=lower_q)
            median_idx = _find_quantile_index(quantiles=quantiles, target=median_q)
            upper_idx = _find_quantile_index(quantiles=quantiles, target=upper_q)

            matched_points = min(len(item_targets), forecast.shape[1])
            if matched_points <= 0:
                continue
            actual_values = item_targets[target_column].astype(float).to_numpy()[:matched_points]
            predicted_center = forecast[median_idx, :matched_points]
            lower_values = forecast[lower_idx, :matched_points]
            upper_values = forecast[upper_idx, :matched_points]
            estimated_std = np.maximum(
                np.abs(upper_values - lower_values) / NORMAL_Z_Q10_TO_Q90, MIN_STD
            )
            logprob_values = [
                _normal_logprob(observed=float(actual), mean=float(mean), std_dev=float(std_dev))
                for actual, mean, std_dev in zip(
                    actual_values, predicted_center, estimated_std, strict=False
                )
            ]
            abs_error_values = np.abs(actual_values - predicted_center)

            record = QueryInferenceRecord(
                query_id=item_id,
                model_role=model_role,
                model_id=str(model_dir),
                item_id=item_id,
                matched_points=int(matched_points),
                average_logprob=float(sum(logprob_values) / len(logprob_values)),
                average_abs_error=float(float(np.mean(abs_error_values))),
                latency_seconds=float(latency_seconds),
                temperature=config.sampling.temperature,
                top_p=config.sampling.top_p,
                num_samples=config.sampling.num_samples,
                seed=config.sampling.seed,
            )
            file_handle.write(json.dumps(record.model_dump(), ensure_ascii=True) + "\n")
            records_written += 1
    return records_written


def main() -> None:
    args = parse_args()
    config_path = Path(args.config).resolve()
    project_root = config_path.parent.parent
    config = load_model_compare_config(config_path=config_path)
    output_file = Path(args.output_file).resolve()
    records_written = run_inference(
        config=config,
        project_root=project_root,
        model_role=str(args.model_role),
        output_file=output_file,
    )
    print(f"Wrote {records_written} inference records to: {output_file}")


if __name__ == "__main__":
    main()
