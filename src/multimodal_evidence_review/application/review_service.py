from __future__ import annotations

import logging
import re

from ..domain.enums import Classification, ImageQuality, Severity
from ..domain.models import ApiMetrics, ClaimRecord, ReviewResult
from ..infrastructure.image_repository import ImageRepository
from ..infrastructure.openai_vision import VisionAnalyzer
from ..validation.evidence import EvidenceValidator, conservative_quality
from ..validation.risk import RiskFlagger


LOGGER = logging.getLogger(__name__)


def extract_claim_fallback(user_claim: str) -> str:
    text = " ".join(user_claim.split()).strip()
    text = re.sub(
        r"^(?:i\s+)?(?:am\s+)?(?:claiming|reporting|stating|saying)(?:\s+that)?\s+",
        "",
        text,
        flags=re.I,
    )
    return text[:500] or "unspecified damage claim"


class ReviewService:
    def __init__(
        self,
        *,
        image_repository: ImageRepository,
        analyzer: VisionAnalyzer,
        evidence_validator: EvidenceValidator,
        risk_flagger: RiskFlagger,
        history_by_user: dict[str, list[dict[str, str]]],
    ) -> None:
        self.image_repository = image_repository
        self.analyzer = analyzer
        self.evidence_validator = evidence_validator
        self.risk_flagger = risk_flagger
        self.history_by_user = history_by_user

    def review_all(self, claims: list[ClaimRecord]) -> list[ReviewResult]:
        return [self.review(claim) for claim in claims]

    def review(self, claim: ClaimRecord) -> ReviewResult:
        log_extra = {"claim_id": claim.claim_id}
        LOGGER.info("Starting review", extra=log_extra)
        images, missing_ids = self.image_repository.load_for_claim(claim)
        history = self.history_by_user.get(claim.user_id or "", [])
        risk_flags = self.risk_flagger.flags_for(claim, history)

        if not images:
            justification = "No readable local images were available, so the claim cannot be verified."
            if missing_ids:
                justification += f" Missing or unreadable image IDs: {', '.join(missing_ids)}."
            return ReviewResult(
                claim_id=claim.claim_id,
                damage_claim=extract_claim_fallback(claim.user_claim),
                classification=Classification.NOT_ENOUGH_INFORMATION,
                issue_type="unknown",
                object_part="unknown",
                severity=Severity.UNKNOWN,
                image_quality=ImageQuality.UNUSABLE,
                supporting_image_ids=[],
                risk_flags=risk_flags,
                evidence_standard_met=False,
                justification=justification,
            )

        analysis, metrics = self.analyzer.analyze(
            claim, images, self.evidence_validator.summarize_candidates()
        )
        allowed_ids = {image.image_id for image in images}
        analysis.supporting_image_ids = [
            image_id for image_id in analysis.supporting_image_ids if image_id in allowed_ids
        ]
        analysis.relevant_image_ids = [
            image_id for image_id in analysis.relevant_image_ids if image_id in allowed_ids
        ]
        analysis.images = [item for item in analysis.images if item.image_id in allowed_ids]

        validation = self.evidence_validator.validate(analysis, images)
        classification = analysis.classification
        justification = analysis.justification.strip()
        if classification != Classification.NOT_ENOUGH_INFORMATION and not validation.met:
            classification = Classification.NOT_ENOUGH_INFORMATION
            justification += " Minimum evidence was not met: " + "; ".join(validation.reasons) + "."
        elif not validation.met:
            justification += " Evidence standard not met: " + "; ".join(validation.reasons) + "."
        if missing_ids:
            justification += f" Missing or unreadable image IDs: {', '.join(missing_ids)}."

        severity = analysis.severity
        if classification == Classification.CONTRADICTED and not any(
            item.damage_visible for item in analysis.images if item.relevant
        ):
            severity = Severity.NONE

        result = ReviewResult(
            claim_id=claim.claim_id,
            damage_claim=analysis.damage_claim,
            classification=classification,
            issue_type=analysis.issue_type.strip().lower() or "unknown",
            object_part=analysis.object_part.strip().lower() or "unknown",
            severity=severity,
            image_quality=conservative_quality(analysis, images),
            supporting_image_ids=analysis.supporting_image_ids,
            risk_flags=risk_flags,
            evidence_standard_met=validation.met,
            justification=" ".join(justification.split())[:1200],
            metrics=metrics,
        )
        LOGGER.info(
            "Completed review classification=%s evidence_standard_met=%s",
            result.classification.value,
            result.evidence_standard_met,
            extra=log_extra,
        )
        return result
