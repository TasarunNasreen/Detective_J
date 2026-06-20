from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

from .enums import Classification, ImageQuality, Severity


class PerImageEvidence(BaseModel):
    model_config = ConfigDict(extra="forbid")

    image_id: str
    issue_type: str = "unknown"
    object_part: str = "unknown"
    severity: Severity = Severity.UNKNOWN
    quality: ImageQuality
    viewpoint: str = "unknown"
    relevant: bool
    damage_visible: bool
    supports_claim: bool
    contradicts_claim: bool
    observations: str = Field(min_length=1, max_length=500)


class VisionAnalysis(BaseModel):
    """Strict structured output returned by the vision model."""

    model_config = ConfigDict(extra="forbid")

    damage_claim: str = Field(min_length=1, max_length=500)
    claimed_issue_type: str = "unknown"
    claimed_object_part: str = "unknown"
    issue_type: str = "unknown"
    object_part: str = "unknown"
    severity: Severity
    image_quality: ImageQuality
    classification: Classification
    supporting_image_ids: list[str] = Field(default_factory=list)
    relevant_image_ids: list[str] = Field(default_factory=list)
    justification: str = Field(min_length=1, max_length=1200)
    images: list[PerImageEvidence] = Field(default_factory=list)

    @field_validator("supporting_image_ids", "relevant_image_ids")
    @classmethod
    def unique_ids(cls, values: list[str]) -> list[str]:
        return list(dict.fromkeys(str(value).strip() for value in values if str(value).strip()))


@dataclass(slots=True)
class ClaimRecord:
    claim_id: str
    user_claim: str
    user_id: str | None = None
    image_ids: list[str] = field(default_factory=list)
    raw: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class LoadedImage:
    image_id: str
    path: Path
    data_url: str
    width: int
    height: int
    blur_score: float
    brightness: float
    technical_quality: ImageQuality


@dataclass(slots=True)
class EvidenceRequirement:
    issue_type: str
    object_part: str | None = None
    severity: str | None = None
    min_images: int = 1
    min_quality: ImageQuality = ImageQuality.FAIR
    required_views: list[str] = field(default_factory=list)


@dataclass(slots=True)
class EvidenceValidation:
    met: bool
    reasons: list[str]
    matched_requirement: EvidenceRequirement | None


@dataclass(slots=True)
class ApiMetrics:
    latency_seconds: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    request_id: str | None = None


@dataclass(slots=True)
class ReviewResult:
    claim_id: str
    damage_claim: str
    classification: Classification
    issue_type: str
    object_part: str
    severity: Severity
    image_quality: ImageQuality
    supporting_image_ids: list[str]
    risk_flags: list[str]
    evidence_standard_met: bool
    justification: str
    metrics: ApiMetrics = field(default_factory=ApiMetrics)

