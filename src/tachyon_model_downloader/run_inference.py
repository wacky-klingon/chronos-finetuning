"""Run probabilistic inference for a single model role and emit JSONL records.

Inference is performed directly against safetensors offline model directories
via ``BaseChronosPipeline``. AutoGluon predictor pickle files are not loaded.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from time import perf_counter
from typing import Any

import pandas as pd
from pydantic import BaseModel

from .chronos_pipeline_loader import load_chronos_pipeline
from .dataset_schema import DatasetRoles, resolve_dataset_roles
from .model_compare_schemas import (
    ModelCompareConfig,
    SamplingConfig,
    load_model_compare_config,
    resolve_path,
)
from .parquet_loader import load_dataset_with_roles

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
    average_interval_width: float
    interval_coverage: float
    pinball_loss_lower: float
    pinball_loss_median: float
    pinball_loss_upper: float
    latency_seconds: float
    lower_quantile: float
    median_quantile: float
    upper_quantile: float
    temperature: float
    top_p: float
    num_samples: int
    seed: int
    prediction_length: int


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


def select_model_dir(config: ModelCompareConfig, project_root: Path, model_role: str) -> Path:
    if model_role == "base":
        return resolve_path(project_root, config.models.base_model_dir)
    return resolve_path(project_root, config.models.finetuned_model_dir)


def _normal_logprob(observed: float, mean: float, std_dev: float) -> float:
    variance = std_dev * std_dev
    return -0.5 * math.log(2.0 * math.pi * variance) - ((observed - mean) ** 2) / (2.0 * variance)


def _mean_pinball_loss(
    observed: pd.Series, predicted_quantile: pd.Series, quantile: float
) -> float:
    diff = observed.astype(float) - predicted_quantile.astype(float)
    positive = (diff >= 0).astype(float) * quantile * diff
    negative = (diff < 0).astype(float) * (quantile - 1.0) * diff
    return float((positive + negative).mean())


def _build_predictions_for_item(
    pipeline: Any,
    history_values: list[float],
    target_timestamps: pd.Series,
    item_id_value: str,
    roles: DatasetRoles,
    sampling: SamplingConfig,
) -> tuple[pd.DataFrame, float]:
    import torch

    horizon_length = min(sampling.prediction_length, len(target_timestamps))
    if horizon_length <= 0:
        return pd.DataFrame(), 0.0
    context_tensor = torch.tensor(history_values, dtype=torch.float32).unsqueeze(0)
    quantile_levels = [
        sampling.lower_quantile,
        sampling.median_quantile,
        sampling.upper_quantile,
    ]
    predict_start = perf_counter()
    quantiles_tensor, mean_tensor = pipeline.predict_quantiles(
        inputs=context_tensor,
        prediction_length=horizon_length,
        quantile_levels=quantile_levels,
    )
    latency_seconds = perf_counter() - predict_start
    horizon_timestamps = target_timestamps.iloc[:horizon_length].reset_index(drop=True)
    predictions_df = pd.DataFrame(
        {
            roles.item_id_column: [item_id_value] * horizon_length,
            roles.timestamp_column: horizon_timestamps,
            "lower_quantile": quantiles_tensor[0, :, 0].detach().cpu().numpy(),
            "median_quantile": quantiles_tensor[0, :, 1].detach().cpu().numpy(),
            "upper_quantile": quantiles_tensor[0, :, 2].detach().cpu().numpy(),
            "mean": mean_tensor[0, :].detach().cpu().numpy(),
        }
    )
    return predictions_df, latency_seconds


def _compute_record_for_item(
    item_id_value: str,
    item_history: pd.DataFrame,
    item_targets: pd.DataFrame,
    pipeline: Any,
    roles: DatasetRoles,
    sampling: SamplingConfig,
    model_role: str,
    model_id: str,
) -> QueryInferenceRecord | None:
    history_sorted = item_history.sort_values(roles.timestamp_column)
    targets_sorted = item_targets.sort_values(roles.timestamp_column).reset_index(drop=True)
    history_values = history_sorted[roles.target_column].astype(float).tolist()
    if not history_values:
        return None
    predictions_df, latency_seconds = _build_predictions_for_item(
        pipeline=pipeline,
        history_values=history_values,
        target_timestamps=targets_sorted[roles.timestamp_column],
        item_id_value=item_id_value,
        roles=roles,
        sampling=sampling,
    )
    if predictions_df.empty:
        return None
    merged = targets_sorted.merge(
        predictions_df,
        how="inner",
        on=[roles.item_id_column, roles.timestamp_column],
        suffixes=("_actual", "_predicted"),
    )
    if merged.empty:
        return None

    target_values = merged[roles.target_column].astype(float)
    mean_values = merged["mean"].astype(float)
    interval_width_series = (
        merged["upper_quantile"].astype(float) - merged["lower_quantile"].astype(float)
    ).abs()
    estimated_std = (interval_width_series / NORMAL_Z_Q10_TO_Q90).clip(lower=MIN_STD)
    abs_error_series = (target_values - mean_values).abs()
    logprob_values = [
        _normal_logprob(
            observed=float(target_values.iloc[i]),
            mean=float(mean_values.iloc[i]),
            std_dev=float(estimated_std.iloc[i]),
        )
        for i in range(len(merged))
    ]
    coverage_mask = (target_values >= merged["lower_quantile"].astype(float)) & (
        target_values <= merged["upper_quantile"].astype(float)
    )
    coverage = float(coverage_mask.mean())
    pinball_lower = _mean_pinball_loss(
        observed=target_values,
        predicted_quantile=merged["lower_quantile"],
        quantile=sampling.lower_quantile,
    )
    pinball_median = _mean_pinball_loss(
        observed=target_values,
        predicted_quantile=merged["median_quantile"],
        quantile=sampling.median_quantile,
    )
    pinball_upper = _mean_pinball_loss(
        observed=target_values,
        predicted_quantile=merged["upper_quantile"],
        quantile=sampling.upper_quantile,
    )
    return QueryInferenceRecord(
        query_id=str(item_id_value),
        model_role=model_role,
        model_id=model_id,
        item_id=str(item_id_value),
        matched_points=int(len(merged)),
        average_logprob=float(sum(logprob_values) / len(logprob_values)),
        average_abs_error=float(abs_error_series.mean()),
        average_interval_width=float(interval_width_series.mean()),
        interval_coverage=coverage,
        pinball_loss_lower=pinball_lower,
        pinball_loss_median=pinball_median,
        pinball_loss_upper=pinball_upper,
        latency_seconds=float(latency_seconds),
        lower_quantile=sampling.lower_quantile,
        median_quantile=sampling.median_quantile,
        upper_quantile=sampling.upper_quantile,
        temperature=sampling.temperature,
        top_p=sampling.top_p,
        num_samples=sampling.num_samples,
        seed=sampling.seed,
        prediction_length=sampling.prediction_length,
    )


def run_inference(
    config: ModelCompareConfig,
    project_root: Path,
    model_role: str,
    output_file: Path,
) -> int:
    role_label = f"{model_role} model"
    model_dir = select_model_dir(config=config, project_root=project_root, model_role=model_role)
    history_path = resolve_path(project_root, config.query_data.history_parquet)
    targets_path = resolve_path(project_root, config.query_data.targets_parquet)
    history_roles = resolve_dataset_roles(
        parquet_path=history_path,
        schema_source=config.schema_source,
        project_root=project_root,
    )
    targets_roles = resolve_dataset_roles(
        parquet_path=targets_path,
        schema_source=config.schema_source,
        project_root=project_root,
    )
    if history_roles != targets_roles:
        raise ValueError(
            "History and targets parquet files resolved to different role mappings. "
            "Both files must declare identical roles."
        )
    roles = history_roles
    history_df = load_dataset_with_roles(
        parquet_path=history_path, roles=roles, source_label="history"
    )
    targets_df = load_dataset_with_roles(
        parquet_path=targets_path, roles=roles, source_label="targets"
    )
    pipeline = load_chronos_pipeline(
        model_dir=model_dir,
        device=config.models.device,
        role_label=role_label,
    )
    unique_items = sorted(targets_df[roles.item_id_column].astype(str).unique().tolist())
    output_file.parent.mkdir(parents=True, exist_ok=True)
    records_written = 0
    with output_file.open("w", encoding="utf-8") as file_handle:
        for item_id_value in unique_items:
            item_history = history_df[
                history_df[roles.item_id_column].astype(str) == item_id_value
            ].copy()
            item_targets = targets_df[
                targets_df[roles.item_id_column].astype(str) == item_id_value
            ].copy()
            if item_history.empty or item_targets.empty:
                continue
            record = _compute_record_for_item(
                item_id_value=item_id_value,
                item_history=item_history,
                item_targets=item_targets,
                pipeline=pipeline,
                roles=roles,
                sampling=config.sampling,
                model_role=model_role,
                model_id=str(model_dir),
            )
            if record is None:
                continue
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
