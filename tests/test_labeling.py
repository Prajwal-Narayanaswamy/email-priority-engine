"""Tests for src.labels.labeling."""

from __future__ import annotations

import pandas as pd
import pytest

from src.labels.labeling import apply_labels, label_email


@pytest.mark.parametrize(
    ("subject", "sender", "expected"),
    [
        ("URGENT: budget call", "jane.doe@example.com", 1),
        ("team lunch plans", "jane.doe@example.com", 0),
        ("deadline moved to Friday", "jane.doe@example.com", 1),
        ("please respond EOD", "jane.doe@example.com", 1),
    ],
)
def test_label_email_urgency_keywords(subject: str, sender: str, expected: int) -> None:
    assert label_email(subject, sender) == expected


def test_label_email_case_insensitive_keywords() -> None:
    assert label_email("ASAP follow up", "jane.doe@example.com") == 1
    assert label_email("asap follow up", "jane.doe@example.com") == 1


def test_label_email_leadership_sender() -> None:
    assert label_email("team lunch", "Kenneth.Lay@enron.com") == 1


def test_label_email_none_fields() -> None:
    assert label_email(None, None) == 0
    assert label_email("lunch", None) == 0
    assert label_email(None, "kenneth.lay@enron.com") == 1


def test_apply_labels_adds_priority_column() -> None:
    df = pd.DataFrame(
        {
            "from": ["kenneth.lay@enron.com", "jane.doe@example.com"],
            "subject": ["reorg update", "team lunch plans"],
        }
    )
    labeled = apply_labels(df)
    assert "priority" in labeled.columns
    assert labeled["priority"].tolist() == [1, 0]


def test_apply_labels_does_not_mutate_input() -> None:
    df = pd.DataFrame({"from": ["jane.doe@example.com"], "subject": ["lunch"]})
    original = df.copy()
    apply_labels(df)
    assert df.equals(original)
