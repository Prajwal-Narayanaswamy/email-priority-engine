"""Tests for src.flows.ingest_flow."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.flows.ingest_flow import ingest_flow


def test_ingest_flow_end_to_end(sample_archive: Path, tmp_path: Path) -> None:
    out = tmp_path / "processed" / "emails.parquet"
    result = ingest_flow(tar_path=str(sample_archive), output_path=str(out))
    assert Path(result).is_file()

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


def test_ingest_flow_respects_limit(sample_archive: Path, tmp_path: Path) -> None:
    out = tmp_path / "processed" / "limited.parquet"
    result = ingest_flow(tar_path=str(sample_archive), output_path=str(out), limit=2)
    df = pd.read_parquet(Path(result))
    assert len(df) == 2
