"""Tests for src.ingestion.ingest."""

from __future__ import annotations

import logging
from email import policy
from email.parser import BytesParser
from pathlib import Path

import pandas as pd

from src.ingestion.ingest import (
    extract_emails,
    ingest,
    parse_email,
    validate_records,
)
from tests.conftest import build_archive, make_email


class _CapturingHandler(logging.Handler):
    def __init__(self) -> None:
        super().__init__()
        self.messages: list[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.messages.append(record.getMessage())


def test_parse_email_extracts_fields() -> None:
    parser = BytesParser(policy=policy.default)
    msg = parser.parsebytes(
        make_email("9", "sender@example.com", "Subject here", body="Body text")
    )
    record = parse_email(msg)
    assert record["uid"] == "<9@enron.com>"
    assert record["from"] == "sender@example.com"
    assert record["to"] == "lynn.blair@enron.com"
    assert record["subject"] == "Subject here"
    assert record["body"] == "Body text"


def test_parse_email_tolerates_malformed_to_header() -> None:
    parser = BytesParser(policy=policy.default)
    malformed_to = (
        "Sales Team@enron.com, .casaudoumecq@enron.com,        "
        "louise.kitchen@enron.com, .costa@enron.com"
    )
    msg = parser.parsebytes(
        make_email("9", "sender@example.com", "Subject here", to=malformed_to)
    )
    record = parse_email(msg)
    assert record["to"] is not None
    addresses = {a.replace('"', "").strip().lower() for a in record["to"].split(",")}
    for expected in (
        "sales team@enron.com",
        ".casaudoumecq@enron.com",
        "louise.kitchen@enron.com",
        ".costa@enron.com",
    ):
        assert expected in addresses
    assert record["subject"] == "Subject here"


def test_parse_email_decodes_encoded_subject() -> None:
    parser = BytesParser(policy=policy.default)
    raw = (
        b"From: a@enron.com\r\n"
        b"Subject: =?utf-8?q?URGENT=20budget?=\r\n"
        b"Content-Type: text/plain; charset=us-ascii\r\n"
        b"\r\n"
        b"body\r\n"
    )
    msg = parser.parsebytes(raw)
    record = parse_email(msg)
    assert record["subject"] == "URGENT budget"


def test_ingest_with_malformed_to_header(tmp_path: Path) -> None:
    malformed_to = (
        "Sales Team@enron.com, .casaudoumecq@enron.com,        "
        "louise.kitchen@enron.com, .costa@enron.com"
    )
    archive = build_archive(
        tmp_path / "malformed.tar.gz",
        {
            "maildir/user-x/inbox/1.": make_email(
                "1", "jane.doe@example.com", "urgent update", to=malformed_to
            )
        },
    )
    out = tmp_path / "processed" / "malformed.parquet"
    ingest(archive, out)
    df = pd.read_parquet(out)
    assert len(df) == 1
    assert df["priority"].iloc[0] == 1


def test_ingest_logs_progress(tmp_path: Path) -> None:
    emails = {
        f"maildir/user-x/inbox/{i}.": make_email(str(i), f"s{i}@example.com", "urgent")
        for i in range(5)
    }
    archive = build_archive(tmp_path / "progress.tar.gz", emails)
    out = tmp_path / "processed" / "progress.parquet"

    capture = logging.getLogger("ingest_test_capture")
    handler = _CapturingHandler()
    capture.addHandler(handler)
    capture.setLevel(logging.INFO)

    ingest(archive, out, batch_size=2, logger=capture)
    assert any("Wrote 2 rows so far" in msg for msg in handler.messages)
    assert any("Wrote 5 labeled emails" in msg for msg in handler.messages)


def test_extract_emails_returns_records(sample_archive: Path) -> None:
    records = extract_emails(sample_archive)
    assert len(records) == 3
    assert records[0]["subject"] == "URGENT: budget call"
    assert records[0]["uid"] == "<1@enron.com>"


def test_extract_emails_limit(sample_archive: Path) -> None:
    records = extract_emails(sample_archive, limit=2)
    assert len(records) == 2


def test_extract_emails_uid_falls_back_to_member_path(tmp_path: Path) -> None:
    archive = build_archive(
        tmp_path / "archive.tar.gz",
        {
            "maildir/kitchen-s/inbox/5.": make_email(
                "5", "sender@example.com", "No message id", include_message_id=False
            )
        },
    )
    records = extract_emails(archive)
    assert records[0]["uid"] == "maildir/kitchen-s/inbox/5."


def test_validate_records_drops_invalid() -> None:
    records = [
        {
            "uid": "1",
            "from": "a@example.com",
            "to": None,
            "subject": "ok",
            "date": None,
            "body": "",
        },
        {
            "uid": "2",
            "from": 123,
            "to": None,
            "subject": "bad",
            "date": None,
            "body": "",
        },
    ]
    validated = validate_records(records)
    assert len(validated) == 1
    assert validated[0].uid == "1"


def test_ingest_end_to_end(sample_archive: Path, tmp_path: Path) -> None:
    out = tmp_path / "processed" / "emails.parquet"
    result = ingest(sample_archive, out)
    assert result == out
    assert out.is_file()

    df = pd.read_parquet(out)
    assert len(df) == 3
    assert list(df.columns) == [
        "uid",
        "from",
        "to",
        "subject",
        "date",
        "body",
        "priority",
    ]
    assert df["priority"].tolist() == [1, 0, 1]


def test_ingest_multiple_batches(tmp_path: Path) -> None:
    emails = {
        f"maildir/user-x/inbox/{i}.": make_email(
            str(i), f"sender{i}@example.com", "urgent follow up" if i % 2 else "lunch"
        )
        for i in range(105)
    }
    archive = build_archive(tmp_path / "multi.tar.gz", emails)
    out = tmp_path / "processed" / "multi.parquet"

    ingest(archive, out, batch_size=50)

    df = pd.read_parquet(out)
    assert len(df) == 105
    assert df["priority"].sum() == 52
    assert list(df["uid"]) == [f"<{i}@enron.com>" for i in range(105)]
