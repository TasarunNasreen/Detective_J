from __future__ import annotations

import logging
import os
import time
from typing import Protocol

from openai import (
    APIConnectionError,
    APITimeoutError,
    InternalServerError,
    OpenAI,
    RateLimitError,
)
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential_jitter

from ..domain.models import ApiMetrics, ClaimRecord, LoadedImage, VisionAnalysis


LOGGER = logging.getLogger(__name__)

SYSTEM_PROMPT = """
You are an image-first damage evidence reviewer. Treat the supplied images as the
primary source of truth. The claimant's text states what must be tested; it is not
proof. Extract the concrete damage allegation from the text, inspect every image,
and compare the allegation with visible evidence.

Classification rules:
- supported: relevant, usable images clearly show damage consistent with the claim.
- contradicted: relevant, usable images clearly show the claimed part and the claim
  conflicts with what is visible. Absence is contradiction only when the relevant
  part is clearly and sufficiently shown.
- not_enough_information: images are missing, irrelevant, obstructed, too poor, do
  not show the relevant part, or are otherwise insufficient to decide.

Do not infer hidden damage. Do not use metadata or claimant history as evidence.
Use only image IDs explicitly supplied in the request. supporting_image_ids must
contain only images that visibly support the classification rationale. For a
contradiction, include clear images that demonstrate the contradiction. Severity
describes visible damage, not claimed damage. If no damage is visible use "none";
if it cannot be determined use "unknown". Keep justification factual and concise.
""".strip()


class VisionAnalyzer(Protocol):
    def analyze(
        self,
        claim: ClaimRecord,
        images: list[LoadedImage],
        requirement_summary: str,
    ) -> tuple[VisionAnalysis, ApiMetrics]: ...


class OpenAIVisionAnalyzer:
    def __init__(
        self,
        *,
        model: str,
        image_detail: str = "high",
        timeout_seconds: float = 120.0,
        client: OpenAI | None = None,
    ) -> None:
        api_key = os.getenv("OPENAI_API_KEY")
        if client is None and not api_key:
            raise RuntimeError(
                "OPENAI_API_KEY is not set. Copy .env.example to .env and add your key."
            )
        self._client = client or OpenAI(api_key=api_key, timeout=timeout_seconds)
        self._model = model
        self._image_detail = image_detail

    @retry(
        retry=retry_if_exception_type(
            (RateLimitError, APIConnectionError, APITimeoutError, InternalServerError)
        ),
        wait=wait_exponential_jitter(initial=1, max=20),
        stop=stop_after_attempt(4),
        reraise=True,
    )
    def analyze(
        self,
        claim: ClaimRecord,
        images: list[LoadedImage],
        requirement_summary: str,
    ) -> tuple[VisionAnalysis, ApiMetrics]:
        image_metadata = "\n".join(
            f"- {image.image_id}: {image.width}x{image.height}, technical_quality="
            f"{image.technical_quality.value}, blur_score={image.blur_score}, "
            f"brightness={image.brightness}"
            for image in images
        )
        content: list[dict[str, object]] = [
            {
                "type": "input_text",
                "text": (
                    f"claim_id: {claim.claim_id}\n"
                    f"user_claim: {claim.user_claim}\n\n"
                    f"Applicable evidence requirements:\n{requirement_summary}\n\n"
                    f"Local image quality measurements:\n{image_metadata}\n\n"
                    "Analyze the images in the same order as their image_id labels."
                ),
            }
        ]
        for image in images:
            content.append(
                {
                    "type": "input_text",
                    "text": f"The next image has image_id={image.image_id}",
                }
            )
            content.append(
                {
                    "type": "input_image",
                    "image_url": image.data_url,
                    "detail": self._image_detail,
                }
            )

        started = time.perf_counter()
        response = self._client.responses.parse(
            model=self._model,
            input=[
                {"role": "system", "content": [{"type": "input_text", "text": SYSTEM_PROMPT}]},
                {"role": "user", "content": content},
            ],
            text_format=VisionAnalysis,
        )
        latency = time.perf_counter() - started
        if response.output_parsed is None:
            raise RuntimeError("OpenAI response did not contain a parsed evidence analysis")

        usage = getattr(response, "usage", None)
        metrics = ApiMetrics(
            latency_seconds=latency,
            input_tokens=int(getattr(usage, "input_tokens", 0) or 0),
            output_tokens=int(getattr(usage, "output_tokens", 0) or 0),
            total_tokens=int(getattr(usage, "total_tokens", 0) or 0),
            request_id=getattr(response, "_request_id", None),
        )
        LOGGER.info(
            "Vision request completed in %.3fs with %s tokens",
            latency,
            metrics.total_tokens,
            extra={"claim_id": claim.claim_id},
        )
        return response.output_parsed, metrics
