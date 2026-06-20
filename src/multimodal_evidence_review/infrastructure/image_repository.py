from __future__ import annotations

import base64
import io
import re
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError

from ..domain.enums import ImageQuality
from ..domain.models import ClaimRecord, LoadedImage


SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}


class ImageRepository:
    def __init__(
        self,
        images_dir: Path,
        *,
        max_dimension: int = 2048,
        jpeg_quality: int = 90,
        max_images_per_claim: int = 8,
    ) -> None:
        self.images_dir = images_dir.resolve()
        self.max_dimension = max_dimension
        self.jpeg_quality = jpeg_quality
        self.max_images_per_claim = max_images_per_claim
        self._index = self._build_index()

    def _build_index(self) -> dict[str, Path]:
        if not self.images_dir.exists():
            raise FileNotFoundError(f"Images directory not found: {self.images_dir}")
        index: dict[str, Path] = {}
        for path in sorted(self.images_dir.rglob("*")):
            if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS:
                relative = path.relative_to(self.images_dir).as_posix().lower()
                for key in (relative, path.name.lower(), path.stem.lower()):
                    index.setdefault(key, path)
        return index

    def _resolve_requested(self, image_id: str) -> Path | None:
        key = image_id.strip().replace("\\", "/").lower()
        if key in self._index:
            return self._index[key]
        stem = Path(key).stem
        return self._index.get(stem)

    def _discover_for_claim(self, claim_id: str) -> list[Path]:
        normalized = claim_id.lower()
        pattern = re.compile(rf"^{re.escape(normalized)}(?:[_\-.]|$)")
        unique_paths = set(self._index.values())
        return sorted(
            path
            for path in unique_paths
            if pattern.search(path.stem.lower())
            or path.parent.name.lower() == normalized
        )

    def load_for_claim(self, claim: ClaimRecord) -> tuple[list[LoadedImage], list[str]]:
        missing: list[str] = []
        paths: list[tuple[str, Path]] = []
        if claim.image_ids:
            for image_id in claim.image_ids:
                path = self._resolve_requested(image_id)
                if path is None:
                    missing.append(image_id)
                else:
                    paths.append((image_id, path))
        else:
            for path in self._discover_for_claim(claim.claim_id):
                paths.append((path.stem, path))

        deduplicated: list[tuple[str, Path]] = []
        seen: set[Path] = set()
        for image_id, path in paths:
            if path not in seen:
                seen.add(path)
                deduplicated.append((image_id, path))

        loaded: list[LoadedImage] = []
        for image_id, path in deduplicated[: self.max_images_per_claim]:
            try:
                loaded.append(self._load(image_id, path))
            except (OSError, ValueError, UnidentifiedImageError):
                missing.append(image_id)
        return loaded, missing

    def _load(self, image_id: str, path: Path) -> LoadedImage:
        with Image.open(path) as source:
            source.verify()
        with Image.open(path) as source:
            image = ImageOps.exif_transpose(source).convert("RGB")
            original_width, original_height = image.size
            array = np.asarray(image)
            gray = cv2.cvtColor(array, cv2.COLOR_RGB2GRAY)
            blur_score = float(cv2.Laplacian(gray, cv2.CV_64F).var())
            brightness = float(gray.mean())
            quality = self._technical_quality(
                original_width, original_height, blur_score, brightness
            )
            image.thumbnail((self.max_dimension, self.max_dimension), Image.Resampling.LANCZOS)
            buffer = io.BytesIO()
            image.save(buffer, format="JPEG", quality=self.jpeg_quality, optimize=True)
        encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
        return LoadedImage(
            image_id=str(image_id),
            path=path,
            data_url=f"data:image/jpeg;base64,{encoded}",
            width=original_width,
            height=original_height,
            blur_score=round(blur_score, 2),
            brightness=round(brightness, 2),
            technical_quality=quality,
        )

    @staticmethod
    def _technical_quality(
        width: int, height: int, blur_score: float, brightness: float
    ) -> ImageQuality:
        short_edge = min(width, height)
        if short_edge < 256 or blur_score < 20 or brightness < 15 or brightness > 240:
            return ImageQuality.UNUSABLE
        if short_edge < 480 or blur_score < 40 or brightness < 30 or brightness > 225:
            return ImageQuality.POOR
        if short_edge < 720 or blur_score < 70:
            return ImageQuality.FAIR
        if short_edge < 1400 or blur_score < 140:
            return ImageQuality.GOOD
        return ImageQuality.EXCELLENT

