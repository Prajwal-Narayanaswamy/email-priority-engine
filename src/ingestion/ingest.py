"""Streaming ingestion of Enron emails from a tar.gz archive.

Pure functions with no Prefect dependency so they are trivially testable.
Prefect tasks and flows in src/tasks/ and src/flows/ wrap these.

Memory is kept bounded by streaming: records are parsed one at a time and
written to Parquet in small batches instead of materializing the full
dataset in RAM.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable, Iterator
from email import policy
from email.header import decode_header, make_header
from email.message import Message
from email.parser import BytesParser
from logging import Logger
from pathlib import Path
from tarfile import TarFile

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from src.labels.labeling import apply_labels

log = logging.getLogger(__name__)

DEFAULT_BATCH_SIZE = 50_000

_STRING_COLUMNS = ("uid", "from", "to", "subject", "date", "body")

_SCHEMA = pa.schema(
    [
        pa.field("uid", pa.string()),
        pa.field("from", pa.string()),
        pa.field("to", pa.string()),
        pa.field("subject", pa.string()),
        pa.field("date", pa.string()),
        pa.field("body", pa.string()),
        pa.field("priority", pa.int64()),
    ]
)


class EmailRecord(BaseModel):
    """Validated schema for a single ingested email."""

    model_config = ConfigDict(extra="forbid")

    uid: str | None = None
    from_: str | None = Field(default=None, alias="from")
    to: str | None = None
    subject: str | None = None
    date: str | None = None
    body: str = ""


def iter_email_bytes(archive: Path) -> Iterator[tuple[str, bytes]]:
    """Yield (member_path, raw_bytes) for each regular file in the archive."""
    with TarFile.open(archive, "r:gz") as tar:
        for member in tar:
            if not member.isfile():
                continue
            extracted = tar.extractfile(member)
            if extracted is None:
                continue
            yield member.name, extracted.read()


def _raw_header(msg: Message, name: str) -> str | None:
    """Return the raw header value, bypassing strict structured parsing."""
    for raw_name, raw_value in msg.raw_items():
        if raw_name.lower() == name.lower():
            return raw_value
    return None


def _safe_header(msg: Message, name: str) -> str | None:
    """Return a header value, tolerating malformed structured headers.

    policy.default lazily applies Python's strict header parser on access,
    which crashes on real-world malformed headers due to a stdlib bug.
    Fall back to the raw value on any parse failure.
    """
    try:
        return msg[name]
    except Exception:
        return _raw_header(msg, name)


def _decode_subject(raw: str | None) -> str | None:
    """Decode an RFC 2047 encoded subject, falling back to the raw value."""
    if raw is None:
        return None
    try:
        return str(make_header(decode_header(raw)))
    except Exception:
        return raw


def parse_email(msg: Message) -> dict[str, str | None]:
    """Extract key fields from a parsed RFC 822 email message."""
    body = _extract_body(msg)
    return {
        "uid": _safe_header(msg, "Message-ID"),
        "from": _safe_header(msg, "From"),
        "to": _safe_header(msg, "To"),
        "subject": _decode_subject(_safe_header(msg, "Subject")),
        "date": _safe_header(msg, "Date"),
        "body": body,
    }


def _extract_body(msg: Message) -> str:
    """Return the plain-text body of the message."""
    try:
        body = msg.get_body(preferencelist=("plain",))
    except Exception:
        body = None
    if body is not None:
        return body.get_content()
    payload = msg.get_payload()
    if isinstance(payload, list):
        parts = [
            part.get_content()
            for part in payload
            if part.get_content_type() == "text/plain"
        ]
        return "\n".join(parts)
    if isinstance(payload, str):
        return payload
    return ""


def iter_records(
    archive: Path, limit: int | None = None
) -> Iterator[dict[str, str | None]]:
    """Stream parsed email records from the archive, one at a time.

    Missing Message-ID headers fall back to the archive member path as the
    uid. When limit is given, iteration stops after that many records.
    """
    parser = BytesParser(policy=policy.default)
    count = 0
    for member_path, raw in iter_email_bytes(archive):
        record = parse_email(parser.parsebytes(raw))
        if record["uid"] is None:
            record["uid"] = member_path
        yield record
        count += 1
        if limit is not None and count >= limit:
            return


def extract_emails(
    archive: Path, limit: int | None = None
) -> list[dict[str, str | None]]:
    """Parse emails from the archive into a list of record dicts.

    Convenience for small archives and tests; prefer iter_records for
    production runs to keep memory bounded.
    """
    return list(iter_records(archive, limit=limit))


def validate_records(records: list[dict[str, str | None]]) -> list[EmailRecord]:
    """Validate records against the EmailRecord schema, dropping invalid ones."""
    validated: list[EmailRecord] = []
    for record in records:
        try:
            validated.append(EmailRecord(**record))
        except ValidationError:
            log.warning("Dropping record with invalid schema: %s", record.get("uid"))
    return validated


def _write_batch(writer: pq.ParquetWriter, batch: list[EmailRecord]) -> None:
    """Label one validated batch and append it to the Parquet file."""
    df = pd.DataFrame([record.model_dump(by_alias=True) for record in batch])
    labeled = apply_labels(df).astype({col: "string" for col in _STRING_COLUMNS})
    writer.write_table(
        pa.Table.from_pandas(labeled, schema=_SCHEMA, preserve_index=False)
    )


def write_records_parquet(
    records: Iterable[dict[str, str | None]],
    output_path: Path,
    batch_size: int = DEFAULT_BATCH_SIZE,
    logger: Logger | None = None,
) -> Path:
    """Stream records into a single Parquet file in bounded-memory batches.

    Records are validated incrementally; invalid rows are dropped. When a
    logger is provided it receives per-batch progress updates.
    """
    active_logger = logger or log
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pending: list[EmailRecord] = []
    written = 0
    dropped = 0
    with pq.ParquetWriter(output_path, _SCHEMA) as writer:
        for record in records:
            try:
                pending.append(EmailRecord(**record))
            except ValidationError:
                dropped += 1
                continue
            if len(pending) >= batch_size:
                _write_batch(writer, pending)
                written += len(pending)
                active_logger.info(
                    "Wrote %d rows so far (%d dropped)", written, dropped
                )
                pending = []
        if pending:
            _write_batch(writer, pending)
            written += len(pending)
    active_logger.info(
        "Wrote %d labeled emails to %s (%d dropped)", written, output_path, dropped
    )
    return output_path


def save_parquet(records: list[EmailRecord], output_path: Path) -> Path:
    """Write validated records to Parquet with priority labels applied."""
    return write_records_parquet(
        (record.model_dump(by_alias=True) for record in records), output_path
    )


def ingest(
    archive: Path,
    output_path: Path,
    limit: int | None = None,
    batch_size: int = DEFAULT_BATCH_SIZE,
    logger: Logger | None = None,
) -> Path:
    """Run the full ingestion pipeline: extract, validate, label, save.

    Streams the archive and writes in batches so memory stays bounded for
    large datasets. When a logger is provided it receives progress updates.
    """
    active_logger = logger or log
    active_logger.info("Streaming emails from %s (batch_size=%d)", archive, batch_size)
    return write_records_parquet(
        iter_records(archive, limit=limit),
        output_path,
        batch_size=batch_size,
        logger=active_logger,
    )
