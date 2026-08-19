"""Shared fixtures and helpers for the test suite."""

from __future__ import annotations

import io
from pathlib import Path
from tarfile import TarFile, TarInfo

import pytest


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
