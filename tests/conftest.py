"""Shared fixtures and helpers for the test suite."""

from __future__ import annotations

import io
from pathlib import Path
from tarfile import TarFile, TarInfo

import numpy as np
import pandas as pd
import pytest

from src.features.features import FEATURE_COLUMNS

FEATURE_ROWS = 400
FEATURE_POSITIVES = 40


def make_email(
    uid: str,
    sender: str,
    subject: str,
    to: str = "lynn.blair@enron.com",
    body: str = "Hello there.",
    include_message_id: bool = True,
) -> bytes:
    """Build a synthetic RFC 822 email as bytes."""
    headers = [
        f"From: {sender}",
        f"To: {to}",
        f"Subject: {subject}",
        "Content-Type: text/plain; charset=us-ascii",
    ]
    if include_message_id:
        headers.insert(0, f"Message-ID: <{uid}@enron.com>")
    text = "\r\n".join(headers) + "\r\n\r\n" + body
    return text.encode("utf-8")


def build_archive(path: Path, emails: dict[str, bytes]) -> Path:
    """Write emails keyed by member name into a gzipped tar archive."""
    with TarFile.open(path, "w:gz") as tar:
        for name, raw in emails.items():
            info = TarInfo(name)
            info.size = len(raw)
            tar.addfile(info, io.BytesIO(raw))
    return path


@pytest.fixture
def sample_archive(tmp_path: Path) -> Path:
    """A small tar.gz archive with three synthetic emails."""
    emails = {
        "maildir/blair-l/inbox/1.": make_email(
            "1", "kenneth.lay@enron.com", "URGENT: budget call"
        ),
        "maildir/blair-l/inbox/2.": make_email(
            "2", "jane.doe@example.com", "Team lunch plans"
        ),
        "maildir/blair-l/inbox/3.": make_email(
            "3", "bob.smith@example.com", "Deadline moved to Friday"
        ),
    }
    return build_archive(tmp_path / "sample.tar.gz", emails)


@pytest.fixture
def features_parquet(tmp_path: Path) -> Path:
    """A feature-layer parquet with a 3% positive rate, in shuffled ingest order.

    Mirrors the real features.parquet schema (uid, from, subject, date, priority
    + the 8 tabular features) so training tests can read it without the 511MB body.
    """
    rng = np.random.default_rng(1)
    n_pos = FEATURE_POSITIVES
    n_neg = FEATURE_ROWS - n_pos

    frame = pd.DataFrame(
        {
            "uid": [str(i) for i in range(FEATURE_ROWS)],
            "from": [f"user{i}@enron.com" for i in range(FEATURE_ROWS)],
            "subject": [
                "urgent deadline action required asap respond" for _ in range(n_pos)
            ]
            + [f"routine update number {i} for the team" for i in range(n_neg)],
            "date": ["2001-09-14 14:05:43-07:00"] * FEATURE_ROWS,
            "priority": [1] * n_pos + [0] * n_neg,
        }
    )
    frame = frame.sample(frac=1.0, random_state=3).reset_index(drop=True)

    values = {
        "sender_frequency": rng.integers(1, 50, FEATURE_ROWS),
        "sender_is_leadership": rng.integers(0, 2, FEATURE_ROWS),
        "hour_of_day": rng.integers(0, 24, FEATURE_ROWS),
        "day_of_week": rng.integers(0, 7, FEATURE_ROWS),
        "urgency_keyword_count": frame["priority"] * 2,
        "subject_length": rng.integers(5, 80, FEATURE_ROWS),
        "has_question_mark": rng.integers(0, 2, FEATURE_ROWS),
        "thread_depth": rng.integers(0, 4, FEATURE_ROWS),
    }
    for name, series in values.items():
        frame[name] = series.astype("int64")

    assert set(FEATURE_COLUMNS) <= set(frame.columns)

    path = tmp_path / "features.parquet"
    frame.to_parquet(path, index=False)
    return path
