from __future__ import annotations

import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path


def configure_logging(level: str = "INFO", log_dir: str | Path = "logs") -> None:
    directory = Path(log_dir)
    directory.mkdir(parents=True, exist_ok=True)
    formatter = logging.Formatter(
        "%(asctime)s %(levelname)s %(name)s claim_id=%(claim_id)s %(message)s"
    )

    class ClaimContextFilter(logging.Filter):
        def filter(self, record: logging.LogRecord) -> bool:
            if not hasattr(record, "claim_id"):
                record.claim_id = "-"
            return True

    stream = logging.StreamHandler()
    stream.setFormatter(formatter)
    stream.addFilter(ClaimContextFilter())

    file_handler = RotatingFileHandler(
        directory / "evidence_review.log",
        maxBytes=2_000_000,
        backupCount=3,
        encoding="utf-8",
    )
    file_handler.setFormatter(formatter)
    file_handler.addFilter(ClaimContextFilter())

    root = logging.getLogger()
    root.handlers.clear()
    root.setLevel(getattr(logging, level.upper(), logging.INFO))
    root.addHandler(stream)
    root.addHandler(file_handler)
