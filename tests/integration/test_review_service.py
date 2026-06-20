from pathlib import Path

from PIL import Image

from multimodal_evidence_review.application.review_service import ReviewService
from multimodal_evidence_review.domain.enums import Classification, ImageQuality, Severity
from multimodal_evidence_review.domain.models import (
    ApiMetrics,
    ClaimRecord,
    EvidenceRequirement,
    PerImageEvidence,
    VisionAnalysis,
)
from multimodal_evidence_review.infrastructure.image_repository import ImageRepository
from multimodal_evidence_review.validation.evidence import EvidenceValidator
from multimodal_evidence_review.validation.risk import RiskFlagger


class FakeAnalyzer:
    def analyze(self, claim, images, requirement_summary):
        image_id = images[0].image_id
        return (
            VisionAnalysis(
                damage_claim="dent on door",
                claimed_issue_type="dent",
                claimed_object_part="door",
                issue_type="dent",
                object_part="door",
                severity=Severity.MINOR,
                image_quality=ImageQuality.GOOD,
                classification=Classification.SUPPORTED,
                supporting_image_ids=[image_id],
                relevant_image_ids=[image_id],
                justification="The door dent is visible.",
                images=[
                    PerImageEvidence(
                        image_id=image_id,
                        issue_type="dent",
                        object_part="door",
                        severity=Severity.MINOR,
                        quality=ImageQuality.GOOD,
                        viewpoint="front",
                        relevant=True,
                        damage_visible=True,
                        supports_claim=True,
                        contradicts_claim=False,
                        observations="Visible dent.",
                    )
                ],
            ),
            ApiMetrics(total_tokens=100),
        )


def test_service_downgrades_when_minimum_evidence_is_not_met(tmp_path: Path) -> None:
    images_dir = tmp_path / "images"
    images_dir.mkdir()
    image_path = images_dir / "C1_1.jpg"
    Image.new("RGB", (1200, 900), color=(100, 120, 140)).save(image_path)

    service = ReviewService(
        image_repository=ImageRepository(images_dir),
        analyzer=FakeAnalyzer(),
        evidence_validator=EvidenceValidator(
            [EvidenceRequirement(issue_type="dent", min_images=2)]
        ),
        risk_flagger=RiskFlagger(),
        history_by_user={},
    )
    result = service.review(ClaimRecord(claim_id="C1", user_claim="There is a door dent"))
    assert result.classification == Classification.NOT_ENOUGH_INFORMATION
    assert result.evidence_standard_met is False

