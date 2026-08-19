"""Tests for src.flows.feature_flow."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from src.flows.feature_flow import feature_flow


@pytest.fixture
def labeled_parquet(tmp_path: Path) -> Path:
    df = pd.DataFrame(
        {
            "uid": ["1", "2"],
            "from": ["kenneth.lay@enron.com", "jane.doe@example.com"],
            "to": ["x@y.com", None],
            "subject": ["URGENT: budget", "Team lunch plans"],
            "date": ["2001-09-14 14:05:43-07:00", "2001-09-15 02:30:00-07:00"],
            "body": ["a" * 10, "b" * 10],
            "priority": [1, 0],
        }
    )
    path = tmp_path / "emails.parquet"
    df.to_parquet(path, index=False)
    return path


def test_feature_flow_end_to_end(labeled_parquet: Path, tmp_path: Path) -> None:
    out = tmp_path / "features" / "features.parquet"
    result = feature_flow(input_path=str(labeled_parquet), output_path=str(out))
    assert Path(result).is_file()

    df = pd.read_parquet(out)
    assert len(df) == 2
    assert "body" not in df.columns
    assert "to" not in df.columns
    assert list(df.columns) == [
        "uid",
        "from",
        "subject",
        "date",
        "priority",
        "sender_frequency",
        "sender_is_leadership",
        "hour_of_day",
        "day_of_week",
        "urgency_keyword_count",
        "subject_length",
        "has_question_mark",
        "thread_depth",
    ]


def test_feature_flow_respects_limit(labeled_parquet: Path, tmp_path: Path) -> None:
    out = tmp_path / "limited.parquet"
    result = feature_flow(
        input_path=str(labeled_parquet), output_path=str(out), limit=1
    )
    df = pd.read_parquet(Path(result))
    assert len(df) == 1


def test_feature_flow_feature_values(labeled_parquet: Path, tmp_path: Path) -> None:
    out = tmp_path / "values.parquet"
    result = feature_flow(input_path=str(labeled_parquet), output_path=str(out))
    df = pd.read_parquet(Path(result))
    assert df["sender_is_leadership"].tolist() == [1, 0]
    assert df["urgency_keyword_count"].tolist() == [1, 0]
