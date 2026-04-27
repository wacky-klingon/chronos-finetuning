"""Compare base and fine-tuned probabilistic inference records.

Outputs raw metric comparisons only. No winner labels, pass/fail flags, or
threshold-based judgments are produced. Interpretation is delegated to
downstream analysis (for example via :mod:`analysis_hooks`).
"""

from __future__ import annotations

import argparse
import json
import statistics
from datetime import datetime, timezone
from pathlib import Path

from pydantic import BaseModel, Field


class InferenceRecord(BaseModel):
    query_id: str = Field(min_length=1)
    model_role: str = Field(min_length=1)
    model_id: str = Field(min_length=1)
    item_id: str = Field(min_length=1)
    matched_points: int = Field(ge=0)
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


class ComparisonRow(BaseModel):
    query_id: str
    base_average_logprob: float
    finetuned_average_logprob: float
    logprob_delta: float
    base_average_abs_error: float
    finetuned_average_abs_error: float
    abs_error_delta: float
    base_average_interval_width: float
    finetuned_average_interval_width: float
    interval_width_delta: float
    base_interval_coverage: float
    finetuned_interval_coverage: float
    interval_coverage_delta: float
    base_pinball_loss_lower: float
    finetuned_pinball_loss_lower: float
    pinball_loss_lower_delta: float
    base_pinball_loss_median: float
    finetuned_pinball_loss_median: float
    pinball_loss_median_delta: float
    base_pinball_loss_upper: float
    finetuned_pinball_loss_upper: float
    pinball_loss_upper_delta: float
    base_latency_seconds: float
    finetuned_latency_seconds: float
    latency_delta_seconds: float


class AggregateMetrics(BaseModel):
    base_mean_logprob: float
    finetuned_mean_logprob: float
    mean_logprob_delta: float
    median_logprob_delta: float
    stdev_logprob_delta: float
    base_mean_abs_error: float
    finetuned_mean_abs_error: float
    mean_abs_error_delta: float
    base_mean_interval_width: float
    finetuned_mean_interval_width: float
    mean_interval_width_delta: float
    base_mean_interval_coverage: float
    finetuned_mean_interval_coverage: float
    mean_interval_coverage_delta: float
    base_mean_pinball_loss_median: float
    finetuned_mean_pinball_loss_median: float
    mean_pinball_loss_median_delta: float
    base_mean_latency_seconds: float
    finetuned_mean_latency_seconds: float
    mean_latency_delta_seconds: float


class ComparisonSummary(BaseModel):
    generated_at_utc: str
    total_queries_compared: int
    aggregates: AggregateMetrics
    rows: list[ComparisonRow]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare base and fine-tuned inference records.")
    parser.add_argument(
        "--base-file", required=True, help="Path to base model JSONL inference output."
    )
    parser.add_argument(
        "--finetuned-file",
        required=True,
        help="Path to fine-tuned model JSONL inference output.",
    )
    parser.add_argument("--output-json", required=True, help="Path to comparison JSON output.")
    parser.add_argument(
        "--output-markdown", required=True, help="Path to comparison markdown output."
    )
    return parser.parse_args()


def _read_jsonl(path: Path) -> list[InferenceRecord]:
    if not path.exists():
        raise FileNotFoundError(f"Inference output does not exist: {path}")
    records: list[InferenceRecord] = []
    with path.open("r", encoding="utf-8") as file_handle:
        for raw_line in file_handle:
            line = raw_line.strip()
            if not line:
                continue
            records.append(InferenceRecord(**json.loads(line)))
    return records


def _records_by_query(records: list[InferenceRecord]) -> dict[str, InferenceRecord]:
    by_query: dict[str, InferenceRecord] = {}
    for record in records:
        by_query[record.query_id] = record
    return by_query


def _build_comparison_row(
    base_record: InferenceRecord, finetuned_record: InferenceRecord
) -> ComparisonRow:
    return ComparisonRow(
        query_id=base_record.query_id,
        base_average_logprob=base_record.average_logprob,
        finetuned_average_logprob=finetuned_record.average_logprob,
        logprob_delta=finetuned_record.average_logprob - base_record.average_logprob,
        base_average_abs_error=base_record.average_abs_error,
        finetuned_average_abs_error=finetuned_record.average_abs_error,
        abs_error_delta=finetuned_record.average_abs_error - base_record.average_abs_error,
        base_average_interval_width=base_record.average_interval_width,
        finetuned_average_interval_width=finetuned_record.average_interval_width,
        interval_width_delta=(
            finetuned_record.average_interval_width - base_record.average_interval_width
        ),
        base_interval_coverage=base_record.interval_coverage,
        finetuned_interval_coverage=finetuned_record.interval_coverage,
        interval_coverage_delta=(
            finetuned_record.interval_coverage - base_record.interval_coverage
        ),
        base_pinball_loss_lower=base_record.pinball_loss_lower,
        finetuned_pinball_loss_lower=finetuned_record.pinball_loss_lower,
        pinball_loss_lower_delta=(
            finetuned_record.pinball_loss_lower - base_record.pinball_loss_lower
        ),
        base_pinball_loss_median=base_record.pinball_loss_median,
        finetuned_pinball_loss_median=finetuned_record.pinball_loss_median,
        pinball_loss_median_delta=(
            finetuned_record.pinball_loss_median - base_record.pinball_loss_median
        ),
        base_pinball_loss_upper=base_record.pinball_loss_upper,
        finetuned_pinball_loss_upper=finetuned_record.pinball_loss_upper,
        pinball_loss_upper_delta=(
            finetuned_record.pinball_loss_upper - base_record.pinball_loss_upper
        ),
        base_latency_seconds=base_record.latency_seconds,
        finetuned_latency_seconds=finetuned_record.latency_seconds,
        latency_delta_seconds=(finetuned_record.latency_seconds - base_record.latency_seconds),
    )


def _safe_stdev(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    return float(statistics.pstdev(values))


def _aggregate(rows: list[ComparisonRow]) -> AggregateMetrics:
    base_logprobs = [row.base_average_logprob for row in rows]
    ft_logprobs = [row.finetuned_average_logprob for row in rows]
    logprob_deltas = [row.logprob_delta for row in rows]
    base_abs = [row.base_average_abs_error for row in rows]
    ft_abs = [row.finetuned_average_abs_error for row in rows]
    base_iw = [row.base_average_interval_width for row in rows]
    ft_iw = [row.finetuned_average_interval_width for row in rows]
    base_cov = [row.base_interval_coverage for row in rows]
    ft_cov = [row.finetuned_interval_coverage for row in rows]
    base_pin = [row.base_pinball_loss_median for row in rows]
    ft_pin = [row.finetuned_pinball_loss_median for row in rows]
    base_lat = [row.base_latency_seconds for row in rows]
    ft_lat = [row.finetuned_latency_seconds for row in rows]
    return AggregateMetrics(
        base_mean_logprob=float(statistics.fmean(base_logprobs)),
        finetuned_mean_logprob=float(statistics.fmean(ft_logprobs)),
        mean_logprob_delta=float(statistics.fmean(logprob_deltas)),
        median_logprob_delta=float(statistics.median(logprob_deltas)),
        stdev_logprob_delta=_safe_stdev(logprob_deltas),
        base_mean_abs_error=float(statistics.fmean(base_abs)),
        finetuned_mean_abs_error=float(statistics.fmean(ft_abs)),
        mean_abs_error_delta=float(statistics.fmean(ft_abs)) - float(statistics.fmean(base_abs)),
        base_mean_interval_width=float(statistics.fmean(base_iw)),
        finetuned_mean_interval_width=float(statistics.fmean(ft_iw)),
        mean_interval_width_delta=(
            float(statistics.fmean(ft_iw)) - float(statistics.fmean(base_iw))
        ),
        base_mean_interval_coverage=float(statistics.fmean(base_cov)),
        finetuned_mean_interval_coverage=float(statistics.fmean(ft_cov)),
        mean_interval_coverage_delta=(
            float(statistics.fmean(ft_cov)) - float(statistics.fmean(base_cov))
        ),
        base_mean_pinball_loss_median=float(statistics.fmean(base_pin)),
        finetuned_mean_pinball_loss_median=float(statistics.fmean(ft_pin)),
        mean_pinball_loss_median_delta=(
            float(statistics.fmean(ft_pin)) - float(statistics.fmean(base_pin))
        ),
        base_mean_latency_seconds=float(statistics.fmean(base_lat)),
        finetuned_mean_latency_seconds=float(statistics.fmean(ft_lat)),
        mean_latency_delta_seconds=(
            float(statistics.fmean(ft_lat)) - float(statistics.fmean(base_lat))
        ),
    )


def compare_records(
    base_records: list[InferenceRecord],
    finetuned_records: list[InferenceRecord],
) -> ComparisonSummary:
    base_by_query = _records_by_query(base_records)
    finetuned_by_query = _records_by_query(finetuned_records)
    query_ids = sorted(set(base_by_query).intersection(finetuned_by_query))
    if not query_ids:
        raise ValueError("No shared query_id values found between base and fine-tuned records.")
    rows = [
        _build_comparison_row(
            base_record=base_by_query[query_id],
            finetuned_record=finetuned_by_query[query_id],
        )
        for query_id in query_ids
    ]
    return ComparisonSummary(
        generated_at_utc=datetime.now(timezone.utc).isoformat(),
        total_queries_compared=len(rows),
        aggregates=_aggregate(rows),
        rows=rows,
    )


def _write_json(path: Path, summary: ComparisonSummary) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file_handle:
        json.dump(summary.model_dump(), file_handle, indent=2)


def _write_markdown(path: Path, summary: ComparisonSummary) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    aggregates = summary.aggregates
    lines: list[str] = [
        "# Base vs Fine-Tuned Raw Metric Comparison",
        "",
        f"- Generated at UTC: {summary.generated_at_utc}",
        f"- Total queries compared: {summary.total_queries_compared}",
        "",
        "## Aggregate metrics",
        "",
        "| metric | base | finetuned | delta (finetuned - base) |",
        "| --- | ---: | ---: | ---: |",
        f"| mean logprob | {aggregates.base_mean_logprob:.6f} | "
        f"{aggregates.finetuned_mean_logprob:.6f} | {aggregates.mean_logprob_delta:.6f} |",
        f"| median logprob delta | - | - | {aggregates.median_logprob_delta:.6f} |",
        f"| stdev logprob delta | - | - | {aggregates.stdev_logprob_delta:.6f} |",
        f"| mean abs error | {aggregates.base_mean_abs_error:.6f} | "
        f"{aggregates.finetuned_mean_abs_error:.6f} | {aggregates.mean_abs_error_delta:.6f} |",
        f"| mean interval width | {aggregates.base_mean_interval_width:.6f} | "
        f"{aggregates.finetuned_mean_interval_width:.6f} | "
        f"{aggregates.mean_interval_width_delta:.6f} |",
        f"| mean interval coverage | {aggregates.base_mean_interval_coverage:.6f} | "
        f"{aggregates.finetuned_mean_interval_coverage:.6f} | "
        f"{aggregates.mean_interval_coverage_delta:.6f} |",
        f"| mean pinball loss (median) | {aggregates.base_mean_pinball_loss_median:.6f} | "
        f"{aggregates.finetuned_mean_pinball_loss_median:.6f} | "
        f"{aggregates.mean_pinball_loss_median_delta:.6f} |",
        f"| mean latency seconds | {aggregates.base_mean_latency_seconds:.6f} | "
        f"{aggregates.finetuned_mean_latency_seconds:.6f} | "
        f"{aggregates.mean_latency_delta_seconds:.6f} |",
        "",
        "## Per-query metrics",
        "",
        "| query_id | base_logprob | finetuned_logprob | logprob_delta | "
        "abs_error_delta | interval_coverage_delta | pinball_median_delta | "
        "latency_delta |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in summary.rows:
        lines.append(
            f"| {row.query_id} | {row.base_average_logprob:.6f} | "
            f"{row.finetuned_average_logprob:.6f} | {row.logprob_delta:.6f} | "
            f"{row.abs_error_delta:.6f} | {row.interval_coverage_delta:.6f} | "
            f"{row.pinball_loss_median_delta:.6f} | {row.latency_delta_seconds:.6f} |"
        )
    with path.open("w", encoding="utf-8") as file_handle:
        file_handle.write("\n".join(lines) + "\n")


def main() -> None:
    args = parse_args()
    base_file = Path(args.base_file).resolve()
    finetuned_file = Path(args.finetuned_file).resolve()
    output_json = Path(args.output_json).resolve()
    output_markdown = Path(args.output_markdown).resolve()
    base_records = _read_jsonl(base_file)
    finetuned_records = _read_jsonl(finetuned_file)
    summary = compare_records(base_records=base_records, finetuned_records=finetuned_records)
    _write_json(output_json, summary)
    _write_markdown(output_markdown, summary)
    print(f"Comparison JSON: {output_json}")
    print(f"Comparison markdown: {output_markdown}")


if __name__ == "__main__":
    main()
