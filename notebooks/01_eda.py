"""EDA on the Enron email dataset.

Streams a sample of emails directly from the compressed tar.gz archive
(no full extraction) and reports dataset-level statistics.
"""

from __future__ import annotations

import logging
import os
import sys
from email import policy
from email.parser import BytesParser
from pathlib import Path
from tarfile import TarFile

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-7s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

SAMPLE_SIZE = 5000
ARCHIVE_PATH = Path(
    os.environ.get("ENRON_ARCHIVE", "data/raw/enron_mail_20150507.tar.gz")
)
KEY_FIELDS = ("from", "to", "subject", "date")


def summarize(df: pd.DataFrame, total_files: int) -> None:
    """Log dataset-level statistics for the sampled emails."""
    sample_rate = 100.0 * len(df) / total_files if total_files else 0.0
    avg_subject_len = df["subject"].dropna().astype(str).str.len().mean()

    log.info("=" * 60)
    log.info("ENRON EDA SUMMARY")
    log.info("=" * 60)
    log.info("Total email count (estimate): %d", total_files)
    log.info("Sample size: %d emails (%.2f%% of archive)", len(df), sample_rate)
    log.info(
        "Sample date range: %s to %s", df["date_parsed"].min(), df["date_parsed"].max()
    )
    log.info("Unique senders (sample): %d", df["from"].nunique())
    log.info("Average subject length (chars): %.2f", avg_subject_len)
    log.info("Null rates:")
    for field in KEY_FIELDS:
        log.info("  %-8s: %.2f%%", field, 100.0 * df[field].isna().mean())
    log.info("=" * 60)


def main() -> None:
    """Run the EDA scan and print summary statistics."""
    from src.ingestion.ingest import parse_email

    if not ARCHIVE_PATH.is_file():
        log.error("Archive not found: %s", ARCHIVE_PATH)
        raise SystemExit(1)

    total_files = 0
    records: list[dict[str, str | None]] = []
    parser = BytesParser(policy=policy.default)

    log.info("Scanning %s (streaming, no full extraction)", ARCHIVE_PATH)
    with TarFile.open(ARCHIVE_PATH, "r:gz") as tar:
        for member in tar:
            if not member.isfile():
                continue
            total_files += 1
            if len(records) < SAMPLE_SIZE:
                extracted = tar.extractfile(member)
                if extracted is not None:
                    records.append(parse_email(parser.parsebytes(extracted.read())))

    if not records:
        log.error("No email files found in archive.")
        raise SystemExit(1)

    df = pd.DataFrame(records)
    df["date_parsed"] = pd.to_datetime(df["date"], errors="coerce", utc=True)
    summarize(df, total_files)


if __name__ == "__main__":
    main()
