from __future__ import annotations

from .application.review_service import ReviewService
from .config import Settings
from .infrastructure.csv_repository import CsvRepository
from .infrastructure.image_repository import ImageRepository
from .infrastructure.openai_vision import OpenAIVisionAnalyzer, VisionAnalyzer
from .validation.evidence import EvidenceValidator
from .validation.risk import RiskFlagger


def build_service(
    settings: Settings,
    repository: CsvRepository,
    *,
    analyzer: VisionAnalyzer | None = None,
) -> ReviewService:
    history = repository.load_history(settings.history_path)
    requirements = repository.load_requirements(settings.requirements_path)
    image_repository = ImageRepository(
        settings.images_dir,
        max_dimension=settings.max_image_dimension,
        jpeg_quality=settings.jpeg_quality,
        max_images_per_claim=settings.max_images_per_claim,
    )
    vision_analyzer = analyzer or OpenAIVisionAnalyzer(
        model=settings.model,
        image_detail=settings.image_detail,
        timeout_seconds=settings.timeout_seconds,
    )
    return ReviewService(
        image_repository=image_repository,
        analyzer=vision_analyzer,
        evidence_validator=EvidenceValidator(requirements),
        risk_flagger=RiskFlagger(),
        history_by_user=history,
    )

