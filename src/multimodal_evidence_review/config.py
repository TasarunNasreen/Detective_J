from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


@dataclass(frozen=True, slots=True)
class Settings:
    dataset_dir: Path
    output_path: Path
    claims_filename: str = "claims.csv"
    sample_claims_filename: str = "sample_claims.csv"
    history_filename: str = "user_history.csv"
    requirements_filename: str = "evidence_requirements.csv"
    images_dirname: str = "images"
    model: str = "gpt-4.1-mini"
    image_detail: str = "high"
    timeout_seconds: float = 120.0
    max_images_per_claim: int = 8
    max_image_dimension: int = 2048
    jpeg_quality: int = 90
    log_level: str = "INFO"

    @property
    def claims_path(self) -> Path:
        return self.dataset_dir / self.claims_filename

    @property
    def sample_claims_path(self) -> Path:
        return self.dataset_dir / self.sample_claims_filename

    @property
    def history_path(self) -> Path:
        return self.dataset_dir / self.history_filename

    @property
    def requirements_path(self) -> Path:
        return self.dataset_dir / self.requirements_filename

    @property
    def images_dir(self) -> Path:
        return self.dataset_dir / self.images_dirname

    @classmethod
    def from_env(
        cls,
        dataset_dir: str | Path = "dataset",
        output_path: str | Path = "output.csv",
        **overrides: object,
    ) -> "Settings":
        load_dotenv()
        values: dict[str, object] = {
            "dataset_dir": Path(dataset_dir).resolve(),
            "output_path": Path(output_path).resolve(),
            "model": os.getenv("OPENAI_MODEL", "gpt-4.1-mini"),
            "image_detail": os.getenv("OPENAI_IMAGE_DETAIL", "high"),
            "timeout_seconds": float(os.getenv("OPENAI_TIMEOUT_SECONDS", "120")),
            "max_images_per_claim": int(os.getenv("MAX_IMAGES_PER_CLAIM", "8")),
            "max_image_dimension": int(os.getenv("MAX_IMAGE_DIMENSION", "2048")),
            "jpeg_quality": int(os.getenv("JPEG_QUALITY", "90")),
            "log_level": os.getenv("LOG_LEVEL", "INFO"),
        }
        values.update({key: value for key, value in overrides.items() if value is not None})
        return cls(**values)

