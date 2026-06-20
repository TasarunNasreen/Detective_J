from multimodal_evidence_review.domain.models import ClaimRecord
from multimodal_evidence_review.validation.risk import RiskFlagger


def test_history_creates_flags_without_touching_evidence() -> None:
    claim = ClaimRecord(claim_id="C1", user_id="U1", user_claim="broken front screen")
    rows = [
        {"fraud_flag": "true", "claim_count": "4", "account_age_days": "10"},
    ]
    assert RiskFlagger().flags_for(claim, rows) == [
        "high_claim_frequency",
        "new_account",
        "prior_fraud_indicator",
    ]

