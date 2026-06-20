from __future__ import annotations

import re
from collections.abc import Iterable

from ..domain.models import ClaimRecord
from ..infrastructure.csv_repository import parse_list


VALID_RISK_FLAGS = {
    "adverse_claim_history",
    "duplicate_evidence_history",
    "high_claim_frequency",
    "new_account",
    "prior_fraud_indicator",
    "repeated_similar_claims",
}

_TRUE_VALUES = {"1", "true", "yes", "y", "flagged", "positive"}


def _truthy(value: str | None) -> bool:
    return str(value or "").strip().lower() in _TRUE_VALUES


def _first_number(rows: Iterable[dict[str, str]], names: set[str]) -> float | None:
    for row in rows:
        for name in names:
            if name in row and str(row[name]).strip():
                match = re.search(r"-?\d+(?:\.\d+)?", str(row[name]))
                if match:
                    return float(match.group())
    return None


class RiskFlagger:
    """Creates risk flags from history only; it never changes evidence fields."""

    def flags_for(
        self, claim: ClaimRecord, history_rows: list[dict[str, str]]
    ) -> list[str]:
        if not history_rows:
            return []
        flags: set[str] = set()

        for row in history_rows:
            for value in parse_list(row.get("risk_flags")):
                normalized = value.strip().lower().replace(" ", "_")
                if normalized in VALID_RISK_FLAGS:
                    flags.add(normalized)
            if any(
                _truthy(row.get(name))
                for name in ("fraud_flag", "suspected_fraud", "prior_fraud", "is_fraud")
            ):
                flags.add("prior_fraud_indicator")
            if any(
                _truthy(row.get(name))
                for name in ("duplicate_image", "duplicate_evidence", "reused_image")
            ):
                flags.add("duplicate_evidence_history")

        claim_count = _first_number(
            history_rows,
            {"claim_count", "prior_claim_count", "claims_last_12_months", "total_claims"},
        )
        if (claim_count is not None and claim_count >= 3) or len(history_rows) >= 3:
            flags.add("high_claim_frequency")

        account_age = _first_number(
            history_rows, {"account_age_days", "customer_age_days", "tenure_days"}
        )
        if account_age is not None and account_age < 30:
            flags.add("new_account")

        adverse = 0
        comparable = 0
        prior_texts: list[str] = []
        for row in history_rows:
            outcome = next(
                (
                    row[name].strip().lower()
                    for name in ("classification", "outcome", "status", "decision")
                    if row.get(name)
                ),
                "",
            )
            if outcome:
                comparable += 1
                if any(word in outcome for word in ("denied", "rejected", "contradicted", "fraud")):
                    adverse += 1
            prior_texts.extend(
                row[name].lower()
                for name in ("user_claim", "claim", "claim_text", "description")
                if row.get(name)
            )
        if comparable >= 2 and adverse / comparable >= 0.5:
            flags.add("adverse_claim_history")

        current_terms = set(re.findall(r"[a-z]{4,}", claim.user_claim.lower()))
        for prior_text in prior_texts:
            prior_terms = set(re.findall(r"[a-z]{4,}", prior_text))
            union = current_terms | prior_terms
            if union and len(current_terms & prior_terms) / len(union) >= 0.6:
                flags.add("repeated_similar_claims")
                break

        return sorted(flags & VALID_RISK_FLAGS)

