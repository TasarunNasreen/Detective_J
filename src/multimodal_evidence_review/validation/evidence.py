from __future__ import annotations

from ..domain.enums import QUALITY_RANK, ImageQuality
from ..domain.models import (
    EvidenceRequirement,
    EvidenceValidation,
    LoadedImage,
    VisionAnalysis,
)


def _norm(value: str | None) -> str:
    return " ".join((value or "").strip().lower().replace("_", " ").split())


class EvidenceValidator:
    def __init__(self, requirements: list[EvidenceRequirement]) -> None:
        self.requirements = requirements

    def find_requirement(
        self, issue_type: str, object_part: str, severity: str
    ) -> EvidenceRequirement | None:
        issue = _norm(issue_type)
        part = _norm(object_part)
        severity_value = _norm(severity)
        matches: list[tuple[int, EvidenceRequirement]] = []
        for requirement in self.requirements:
            requirement_issue = _norm(requirement.issue_type)
            if requirement_issue not in {"*", "any", "default", issue}:
                continue
            score = 1 if requirement_issue == issue else 0
            if requirement.object_part:
                if _norm(requirement.object_part) != part:
                    continue
                score += 2
            if requirement.severity:
                if _norm(requirement.severity) != severity_value:
                    continue
                score += 4
            matches.append((score, requirement))
        return max(matches, key=lambda item: item[0])[1] if matches else None

    def summarize_candidates(self) -> str:
        lines = []
        for requirement in self.requirements:
            lines.append(
                f"issue_type={requirement.issue_type}; object_part="
                f"{requirement.object_part or 'any'}; severity={requirement.severity or 'any'}; "
                f"min_images={requirement.min_images}; min_quality="
                f"{requirement.min_quality.value}; required_views="
                f"{','.join(requirement.required_views) or 'none'}"
            )
        return "\n".join(lines)

    def validate(
        self,
        analysis: VisionAnalysis,
        loaded_images: list[LoadedImage],
    ) -> EvidenceValidation:
        requirement = self.find_requirement(
            analysis.issue_type or analysis.claimed_issue_type,
            analysis.object_part or analysis.claimed_object_part,
            analysis.severity.value,
        )
        if requirement is None:
            return EvidenceValidation(
                met=False,
                reasons=["no matching row in evidence_requirements.csv"],
                matched_requirement=None,
            )

        loaded_by_id = {image.image_id: image for image in loaded_images}
        observations_by_id = {item.image_id: item for item in analysis.images}
        relevant_ids = list(
            dict.fromkeys(
                analysis.relevant_image_ids
                + [item.image_id for item in analysis.images if item.relevant]
            )
        )
        qualifying_ids: list[str] = []
        viewpoints: set[str] = set()
        for image_id in relevant_ids:
            loaded = loaded_by_id.get(image_id)
            observed = observations_by_id.get(image_id)
            if loaded is None or observed is None:
                continue
            effective_rank = min(
                QUALITY_RANK[loaded.technical_quality], QUALITY_RANK[observed.quality]
            )
            if effective_rank >= QUALITY_RANK[requirement.min_quality]:
                qualifying_ids.append(image_id)
                viewpoints.add(_norm(observed.viewpoint))

        reasons: list[str] = []
        if len(qualifying_ids) < requirement.min_images:
            reasons.append(
                f"requires {requirement.min_images} relevant image(s) at "
                f"{requirement.min_quality.value} quality or better; found {len(qualifying_ids)}"
            )

        for required_view in requirement.required_views:
            normalized_view = _norm(required_view)
            if not any(
                normalized_view in actual_view or actual_view in normalized_view
                for actual_view in viewpoints
                if actual_view
            ):
                reasons.append(f"missing required view: {required_view}")

        return EvidenceValidation(
            met=not reasons,
            reasons=reasons,
            matched_requirement=requirement,
        )


def conservative_quality(
    analysis: VisionAnalysis, loaded_images: list[LoadedImage]
) -> ImageQuality:
    if not loaded_images:
        return ImageQuality.UNUSABLE
    relevant = set(analysis.relevant_image_ids)
    candidates = [
        image.technical_quality
        for image in loaded_images
        if not relevant or image.image_id in relevant
    ]
    model_quality = analysis.image_quality
    return min(candidates + [model_quality], key=lambda quality: QUALITY_RANK[quality])

