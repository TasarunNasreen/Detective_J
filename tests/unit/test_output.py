import csv
from pathlib import Path

from multimodal_evidence_review.domain.enums import Classification, ImageQuality, Severity
from multimodal_evidence_review.domain.models import ReviewResult
from multimodal_evidence_review.validation.output import OUTPUT_COLUMNS, write_output


def test_output_has_exact_schema(tmp_path: Path) -> None:
    result = ReviewResult(
        claim_id="C1",
        damage_claim="dented door",
        classification=Classification.SUPPORTED,
        issue_type="dent",
        object_part="door",
        severity=Severity.MINOR,
        image_quality=ImageQuality.GOOD,
        supporting_image_ids=["a.jpg"],
        risk_flags=[],
        evidence_standard_met=True,
        justification="The dent is clearly visible.",
    )
    output = tmp_path / "output.csv"
    write_output([result], output)
    with output.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
    assert reader.fieldnames == OUTPUT_COLUMNS
    assert rows[0]["classification"] == "supported"
    assert rows[0]["evidence_standard_met"] == "true"

