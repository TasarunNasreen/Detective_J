from __future__ import annotations

import argparse
import math
import os
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
from openai import OpenAIError

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from multimodal_evidence_review.config import Settings  # noqa: E402
from multimodal_evidence_review.factory import build_service  # noqa: E402
from multimodal_evidence_review.infrastructure.csv_repository import CsvRepository  # noqa: E402
from multimodal_evidence_review.logging_config import configure_logging  # noqa: E402
from multimodal_evidence_review.validation.output import result_to_row, write_output  # noqa: E402


EXPECTED_ALIASES = {
    "classification": (
        "expected_classification",
        "ground_truth_classification",
        "expected_label",
        "label",
        "classification",
    ),
    "severity": ("expected_severity", "ground_truth_severity"),
    "issue_type": ("expected_issue_type", "ground_truth_issue_type"),
    "object_part": ("expected_object_part", "ground_truth_object_part"),
    "evidence_standard_met": (
        "expected_evidence_standard_met",
        "ground_truth_evidence_standard_met",
    ),
}


def _column(frame: pd.DataFrame, aliases: tuple[str, ...]) -> str | None:
    columns = {str(column).strip().lower(): str(column) for column in frame.columns}
    return next((columns[alias] for alias in aliases if alias in columns), None)


def _norm(value: Any) -> str:
    return " ".join(str(value).strip().lower().replace("_", " ").split())


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, math.ceil(percentile * len(ordered)) - 1))
    return ordered[index]


def _cost(input_tokens: int, output_tokens: int) -> tuple[float | None, str]:
    input_rate = os.getenv("OPENAI_INPUT_COST_PER_1M_TOKENS", "").strip()
    output_rate = os.getenv("OPENAI_OUTPUT_COST_PER_1M_TOKENS", "").strip()
    if not input_rate or not output_rate:
        return None, (
            "Not calculated. Set OPENAI_INPUT_COST_PER_1M_TOKENS and "
            "OPENAI_OUTPUT_COST_PER_1M_TOKENS to current model prices."
        )
    value = input_tokens / 1_000_000 * float(input_rate)
    value += output_tokens / 1_000_000 * float(output_rate)
    return value, "Calculated from measured API usage and configured per-million-token rates."


def evaluate(
    settings: Settings,
    *,
    report_path: Path,
    results_path: Path,
    limit: int | None = None,
) -> Path:
    repository = CsvRepository()
    claims, frame = repository.load_claims(settings.sample_claims_path)
    if limit is not None:
        claims = claims[:limit]
        frame = frame.iloc[:limit].copy()

    service = build_service(settings, repository)
    started = time.perf_counter()
    results = service.review_all(claims)
    runtime = time.perf_counter() - started
    write_output(results, results_path)

    actual_rows = {result.claim_id: result_to_row(result) for result in results}
    claim_id_col = _column(frame, ("claim_id", "id", "case_id"))
    field_scores: dict[str, tuple[int, int]] = {}
    for field, aliases in EXPECTED_ALIASES.items():
        expected_col = _column(frame, aliases)
        if expected_col is None or claim_id_col is None:
            continue
        correct = 0
        evaluated = 0
        for _, source_row in frame.iterrows():
            claim_id = str(source_row[claim_id_col]).strip()
            expected = str(source_row[expected_col]).strip()
            if not expected or claim_id not in actual_rows:
                continue
            evaluated += 1
            correct += _norm(expected) == _norm(actual_rows[claim_id][field])
        if evaluated:
            field_scores[field] = (correct, evaluated)

    classification_score = field_scores.get("classification")
    latencies = [
        result.metrics.latency_seconds
        for result in results
        if result.metrics.latency_seconds > 0
    ]
    input_tokens = sum(result.metrics.input_tokens for result in results)
    output_tokens = sum(result.metrics.output_tokens for result in results)
    total_tokens = sum(result.metrics.total_tokens for result in results)
    estimated_cost, cost_note = _cost(input_tokens, output_tokens)

    accuracy_text = (
        f"{classification_score[0] / classification_score[1]:.2%} "
        f"({classification_score[0]}/{classification_score[1]})"
        if classification_score
        else "Not available: no expected classification column was found."
    )
    field_lines = [
        f"| {field} | {correct / count:.2%} | {correct}/{count} |"
        for field, (correct, count) in field_scores.items()
    ] or ["| n/a | n/a | No ground-truth columns found |"]
    cost_text = f"${estimated_cost:.6f}" if estimated_cost is not None else "Not available"

    report = f"""# Detective_J Evaluation Report

Generated: {datetime.now(timezone.utc).isoformat()}

## Summary

- Sample rows evaluated: {len(results)}
- Classification accuracy: {accuracy_text}
- Total runtime: {runtime:.3f} seconds
- Mean end-to-end time per row: {(runtime / len(results) if results else 0):.3f} seconds
- Mean API latency: {(statistics.fmean(latencies) if latencies else 0):.3f} seconds
- P50 API latency: {_percentile(latencies, 0.50):.3f} seconds
- P95 API latency: {_percentile(latencies, 0.95):.3f} seconds
- Measured input tokens: {input_tokens}
- Measured output tokens: {output_tokens}
- Measured total tokens: {total_tokens}
- Estimated API cost: {cost_text}
- Cost method: {cost_note}
- Model: `{settings.model}`

## Accuracy by field

| Field | Accuracy | Correct / evaluated |
|---|---:|---:|
{chr(10).join(field_lines)}

## Method

The evaluator runs the same image-first pipeline used for `claims.csv`, compares
outputs with any `expected_*`, `ground_truth_*`, or supported label columns in
`sample_claims.csv`, and records actual Responses API usage. User-history data is
used only to create risk flags and is never supplied to the vision model.

Detailed predictions are written to `{results_path.as_posix()}`.
"""
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report, encoding="utf-8")
    return report_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate sample_claims.csv")
    parser.add_argument("--dataset-dir", default="dataset")
    parser.add_argument("--report", default="evaluation/evaluation_report.md")
    parser.add_argument("--results", default="evaluation/evaluation_results.csv")
    parser.add_argument("--model", default=None)
    parser.add_argument("--limit", type=int, default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    settings = Settings.from_env(
        dataset_dir=args.dataset_dir,
        output_path=args.results,
        model=args.model,
    )
    configure_logging(settings.log_level)
    try:
        report = evaluate(
            settings,
            report_path=Path(args.report).resolve(),
            results_path=Path(args.results).resolve(),
            limit=args.limit,
        )
    except (FileNotFoundError, ValueError, RuntimeError, OpenAIError) as exc:
        print(f"Evaluation failed: {exc}", file=sys.stderr)
        return 1
    print(f"Wrote {report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
