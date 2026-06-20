"""HackerRank-friendly entry point that also works before editable installation."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from multimodal_evidence_review.cli import main  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(main())
