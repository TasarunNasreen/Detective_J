from __future__ import annotations

import argparse
import logging
from pathlib import Path

from openai import OpenAIError

from .config import Settings
from .factory import build_service
from .infrastructure.csv_repository import CsvRepository
from .logging_config import configure_logging
from .validation.output import write_output


LOGGER = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Review damage claims using local images and the OpenAI Responses API."
    )
    parser.add_argument("--dataset-dir", default="dataset", help="Directory containing CSVs and images/")
    parser.add_argument("--output", default="output.csv", help="Output CSV path")
    parser.add_argument("--model", default=None, help="Override OPENAI_MODEL")
    parser.add_argument("--log-level", default=None, choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return parser


def run(settings: Settings) -> Path:
    repository = CsvRepository()
    claims, _ = repository.load_claims(settings.claims_path)

    # The challenge requires both files. Validate sample_claims.csv here; the
    # evaluator performs its actual reviews separately.
    sample_claims, _ = repository.load_claims(settings.sample_claims_path)
    LOGGER.info(
        "Loaded %d production claims and validated %d sample claims",
        len(claims),
        len(sample_claims),
    )

    service = build_service(settings, repository)
    results = service.review_all(claims)
    write_output(results, settings.output_path)
    LOGGER.info("Wrote %d rows to %s", len(results), settings.output_path)
    return settings.output_path


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    settings = Settings.from_env(
        dataset_dir=args.dataset_dir,
        output_path=args.output,
        model=args.model,
        log_level=args.log_level,
    )
    configure_logging(settings.log_level)
    try:
        output_path = run(settings)
    except (FileNotFoundError, ValueError, RuntimeError, OpenAIError) as exc:
        LOGGER.error("Pipeline failed: %s", exc)
        return 1
    print(f"Wrote {output_path}")
    return 0
