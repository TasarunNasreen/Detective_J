from pathlib import Path

from multimodal_evidence_review.domain.enums import Classification, ImageQuality, Severity
from multimodal_evidence_review.domain.models import (
    EvidenceRequirement,
    LoadedImage,
    PerImageEvidence,
    VisionAnalysis,
)
from multimodal_evidence_review.validation.evidence import EvidenceValidator


def _analysis() -> VisionAnalysis:
    return VisionAnalysis(
        damage_claim="dent on front door",
        claimed_issue_type="dent",
        claimed_object_part="front door",
        issue_type="dent",
        object_part="front door",
        severity=Severity.MINOR,
        image_quality=ImageQuality.GOOD,
        classification=Classification.SUPPORTED,
        supporting_image_ids=["img-1"],
        relevant_image_ids=["img-1"],
        justification="A dent is visible on the front door.",
        images=[
            PerImageEvidence(
                image_id="img-1",
                issue_type="dent",
                object_part="front door",
                severity=Severity.MINOR,
                quality=ImageQuality.GOOD,
                viewpoint="front",
                relevant=True,
                damage_visible=True,
                supports_claim=True,
                contradicts_claim=False,
                observations="A shallow dent is visible.",
            )
        ],
    )


def test_minimum_evidence_is_enforced() -> None:
    validator = EvidenceValidator(
        [EvidenceRequirement(issue_type="dent", min_images=2, min_quality=ImageQuality.FAIR)]
    )
    image = LoadedImage(
        image_id="img-1",
        path=Path("img.jpg"),
        data_url="data:image/jpeg;base64,AA==",
        width=1200,
        height=900,
        blur_score=100,
        brightness=120,
        technical_quality=ImageQuality.GOOD,
    )
    result = validator.validate(_analysis(), [image])
    assert result.met is False
    assert "requires 2" in result.reasons[0]

