from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from pydantic import BaseModel, Field


class InferenceRecord(BaseModel):
    query_id: str = Field(min_length=1)
    average_logprob: float
    average_abs_error: float
    latency_seconds: float
    model_role: str = Field(min_length=1)
    model_id: str = Field(min_length=1)
    matched_points: int = Field(ge=0)
    temperature: float
    top_p: float
    num_samples: int
    seed: int


class ComparisonRow(BaseModel):
    query_id: str
    base_average_logprob: float
    finetuned_average_logprob: float
    logprob_delta: float
    base_average_abs_error: float
    finetuned_average_abs_error: float
    abs_error_delta: float
    base_latency_seconds: float
    finetuned_latency_seconds: float
    latency_delta_seconds: float
    winner: str


class ComparisonSummary(BaseModel):
    generated_at_utc: str
    total_queries_compared: int
    mean_logprob_delta: float
    mean_abs_error_delta: float
    mean_latency_delta_seconds: float
    finetuned_wins: int
    base_wins: int
    ties: int
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


def compare_records(
    base_records: list[InferenceRecord], finetuned_records: list[InferenceRecord]
) -> ComparisonSummary:
    base_by_query = _records_by_query(base_records)
    finetuned_by_query = _records_by_query(finetuned_records)
    query_ids = sorted(set(base_by_query).intersection(finetuned_by_query))
    if not query_ids:
        raise ValueError("No shared query_id values found between base and fine-tuned records.")

    rows: list[ComparisonRow] = []
    finetuned_wins = 0
    base_wins = 0
    ties = 0

    for query_id in query_ids:
        base_record = base_by_query[query_id]
        finetuned_record = finetuned_by_query[query_id]
        logprob_delta = finetuned_record.average_logprob - base_record.average_logprob
        abs_error_delta = finetuned_record.average_abs_error - base_record.average_abs_error
        latency_delta = finetuned_record.latency_seconds - base_record.latency_seconds
        winner = "tie"
        if logprob_delta > 0:
            winner = "finetuned"
            finetuned_wins += 1
        elif logprob_delta < 0:
            winner = "base"
            base_wins += 1
        else:
            ties += 1
        row = ComparisonRow(
            query_id=query_id,
            base_average_logprob=base_record.average_logprob,
            finetuned_average_logprob=finetuned_record.average_logprob,
            logprob_delta=logprob_delta,
            base_average_abs_error=base_record.average_abs_error,
            finetuned_average_abs_error=finetuned_record.average_abs_error,
            abs_error_delta=abs_error_delta,
            base_latency_seconds=base_record.latency_seconds,
            finetuned_latency_seconds=finetuned_record.latency_seconds,
            latency_delta_seconds=latency_delta,
            winner=winner,
        )
        rows.append(row)

    mean_logprob_delta = sum(row.logprob_delta for row in rows) / len(rows)
    mean_abs_error_delta = sum(row.abs_error_delta for row in rows) / len(rows)
    mean_latency_delta = sum(row.latency_delta_seconds for row in rows) / len(rows)
    return ComparisonSummary(
        generated_at_utc=datetime.now(timezone.utc).isoformat(),
        total_queries_compared=len(rows),
        mean_logprob_delta=mean_logprob_delta,
        mean_abs_error_delta=mean_abs_error_delta,
        mean_latency_delta_seconds=mean_latency_delta,
        finetuned_wins=finetuned_wins,
        base_wins=base_wins,
        ties=ties,
        rows=rows,
    )


def _write_json(path: Path, summary: ComparisonSummary) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file_handle:
        json.dump(summary.model_dump(), file_handle, indent=2)


def _write_markdown(path: Path, summary: ComparisonSummary) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Base vs Fine-Tuned Comparison",
        "",
        f"- Generated at UTC: {summary.generated_at_utc}",
        f"- Total queries compared: {summary.total_queries_compared}",
        f"- Mean logprob delta (fine_tuned - base): {summary.mean_logprob_delta:.6f}",
        f"- Mean abs error delta (fine_tuned - base): {summary.mean_abs_error_delta:.6f}",
        f"- Mean latency delta seconds (fine_tuned - base): {summary.mean_latency_delta_seconds:.6f}",
        f"- Win counts: finetuned={summary.finetuned_wins}, base={summary.base_wins}, ties={summary.ties}",
        "",
        "## Per-query",
        "",
        "| query_id | base_logprob | finetuned_logprob | delta | winner |",
        "| --- | ---: | ---: | ---: | --- |",
    ]
    for row in summary.rows:
        lines.append(
            f"| {row.query_id} | {row.base_average_logprob:.6f} | "
            f"{row.finetuned_average_logprob:.6f} | {row.logprob_delta:.6f} | {row.winner} |"
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
