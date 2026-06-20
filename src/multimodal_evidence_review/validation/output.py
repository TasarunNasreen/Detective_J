from __future__ import annotations

import csv
import json
from pathlib import Path

from ..domain.models import ReviewResult
from .risk import VALID_RISK_FLAGS


OUTPUT_COLUMNS = [
    "claim_id",
    "damage_claim",
    "classification",
    "issue_type",
    "object_part",
    "severity",
    "image_quality",
    "supporting_image_ids",
    "risk_flags",
    "evidence_standard_met",
    "justification",
]


def result_to_row(result: ReviewResult) -> dict[str, str]:
    invalid_flags = set(result.risk_flags) - VALID_RISK_FLAGS
    if invalid_flags:
        raise ValueError(f"Invalid risk flags: {sorted(invalid_flags)}")
    return {
        "claim_id": result.claim_id,
        "damage_claim": result.damage_claim,
        "classification": result.classification.value,
        "issue_type": result.issue_type,
        "object_part": result.object_part,
        "severity": result.severity.value,
        "image_quality": result.image_quality.value,
        "supporting_image_ids": json.dumps(result.supporting_image_ids, ensure_ascii=False),
        "risk_flags": json.dumps(result.risk_flags, ensure_ascii=False),
        "evidence_standard_met": str(result.evidence_standard_met).lower(),
        "justification": result.justification,
    }


def write_output(results: list[ReviewResult], path: Path) -> None:
    if len({result.claim_id for result in results}) != len(results):
        raise ValueError("Duplicate claim_id values would produce an invalid output.csv")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=OUTPUT_COLUMNS, extrasaction="raise")
        writer.writeheader()
        writer.writerows(result_to_row(result) for result in results)

