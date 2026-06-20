from __future__ import annotations

import ast
import json
import re
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

from ..domain.enums import ImageQuality
from ..domain.models import ClaimRecord, EvidenceRequirement


def _normalized_columns(frame: pd.DataFrame) -> dict[str, str]:
    return {str(column).strip().lower(): str(column) for column in frame.columns}


def _find_column(
    frame: pd.DataFrame,
    aliases: Iterable[str],
    *,
    required: bool = False,
    source: str = "CSV",
) -> str | None:
    columns = _normalized_columns(frame)
    for alias in aliases:
        if alias.lower() in columns:
            return columns[alias.lower()]
    if required:
        raise ValueError(
            f"{source} must contain one of these columns: {', '.join(aliases)}. "
            f"Found: {', '.join(map(str, frame.columns))}"
        )
    return None


def parse_list(value: Any) -> list[str]:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return []
    if isinstance(value, (list, tuple, set)):
        return [str(item).strip() for item in value if str(item).strip()]
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "null", "[]"}:
        return []
    if text.startswith("["):
        for parser in (json.loads, ast.literal_eval):
            try:
                parsed = parser(text)
                if isinstance(parsed, (list, tuple, set)):
                    return [str(item).strip() for item in parsed if str(item).strip()]
            except (ValueError, SyntaxError, json.JSONDecodeError):
                continue
    return [part.strip().strip("'\"") for part in re.split(r"[;,|]", text) if part.strip()]


def read_csv(path: Path, *, required: bool = True) -> pd.DataFrame:
    if not path.exists():
        if required:
            raise FileNotFoundError(f"Required dataset file not found: {path}")
        return pd.DataFrame()
    try:
        return pd.read_csv(path, dtype=str, keep_default_na=False)
    except pd.errors.EmptyDataError as exc:
        raise ValueError(f"CSV file is empty: {path}") from exc


class CsvRepository:
    CLAIM_ID_ALIASES = ("claim_id", "id", "case_id")
    CLAIM_TEXT_ALIASES = ("user_claim", "claim", "claim_text", "description")
    USER_ID_ALIASES = ("user_id", "customer_id", "claimant_id", "account_id")
    IMAGE_LIST_ALIASES = ("image_ids", "images", "image_files", "image_paths")

    def load_claims(self, path: Path) -> tuple[list[ClaimRecord], pd.DataFrame]:
        frame = read_csv(path)
        claim_id_col = _find_column(
            frame, self.CLAIM_ID_ALIASES, required=True, source=path.name
        )
        claim_text_col = _find_column(
            frame, self.CLAIM_TEXT_ALIASES, required=True, source=path.name
        )
        user_id_col = _find_column(frame, self.USER_ID_ALIASES)
        image_list_col = _find_column(frame, self.IMAGE_LIST_ALIASES)
        individual_image_cols = [
            str(column)
            for column in frame.columns
            if re.match(r"^(image|photo)(?:_?id|_?file|_?path)?_?\d+$", str(column), re.I)
        ]

        claims: list[ClaimRecord] = []
        for _, row in frame.iterrows():
            claim_id = str(row[claim_id_col]).strip()
            user_claim = str(row[claim_text_col]).strip()
            if not claim_id:
                raise ValueError(f"{path.name} contains a blank claim_id")
            if not user_claim:
                raise ValueError(f"Claim {claim_id!r} has a blank user_claim")
            image_ids = parse_list(row[image_list_col]) if image_list_col else []
            for column in individual_image_cols:
                image_ids.extend(parse_list(row[column]))
            claims.append(
                ClaimRecord(
                    claim_id=claim_id,
                    user_claim=user_claim,
                    user_id=str(row[user_id_col]).strip() if user_id_col else None,
                    image_ids=list(dict.fromkeys(image_ids)),
                    raw={str(key): value for key, value in row.to_dict().items()},
                )
            )
        return claims, frame

    def load_history(self, path: Path) -> dict[str, list[dict[str, str]]]:
        frame = read_csv(path)
        user_id_col = _find_column(
            frame, self.USER_ID_ALIASES, required=True, source=path.name
        )
        history: dict[str, list[dict[str, str]]] = {}
        for _, row in frame.iterrows():
            user_id = str(row[user_id_col]).strip()
            if not user_id:
                continue
            history.setdefault(user_id, []).append(
                {str(key).strip().lower(): str(value).strip() for key, value in row.items()}
            )
        return history

    def load_requirements(self, path: Path) -> list[EvidenceRequirement]:
        frame = read_csv(path)
        issue_col = _find_column(
            frame,
            ("issue_type", "damage_type", "claim_type"),
            required=True,
            source=path.name,
        )
        part_col = _find_column(frame, ("object_part", "part", "component"))
        severity_col = _find_column(frame, ("severity", "damage_severity"))
        min_images_col = _find_column(
            frame,
            ("min_images", "minimum_images", "minimum_image_count", "required_image_count"),
        )
        min_quality_col = _find_column(
            frame, ("min_quality", "minimum_quality", "required_quality")
        )
        views_col = _find_column(
            frame, ("required_views", "required_viewpoints", "views")
        )

        requirements: list[EvidenceRequirement] = []
        valid_qualities = {quality.value: quality for quality in ImageQuality}
        for _, row in frame.iterrows():
            issue_type = str(row[issue_col]).strip().lower()
            if not issue_type:
                continue
            quality_text = (
                str(row[min_quality_col]).strip().lower() if min_quality_col else "fair"
            )
            if quality_text not in valid_qualities:
                raise ValueError(
                    f"Invalid min_quality {quality_text!r} in {path.name}; expected one of "
                    f"{', '.join(valid_qualities)}"
                )
            try:
                min_images = int(str(row[min_images_col]).strip() or "1") if min_images_col else 1
            except ValueError as exc:
                raise ValueError(f"Invalid minimum image count in {path.name}") from exc
            requirements.append(
                EvidenceRequirement(
                    issue_type=issue_type,
                    object_part=str(row[part_col]).strip().lower() or None if part_col else None,
                    severity=str(row[severity_col]).strip().lower() or None
                    if severity_col
                    else None,
                    min_images=max(0, min_images),
                    min_quality=valid_qualities[quality_text],
                    required_views=[view.lower() for view in parse_list(row[views_col])]
                    if views_col
                    else [],
                )
            )
        if not requirements:
            raise ValueError(f"No valid evidence requirements found in {path}")
        return requirements

