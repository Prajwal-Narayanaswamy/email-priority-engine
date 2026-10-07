"""Tests of the pure ADR-012 dedupe and grouped split policy."""

from __future__ import annotations

import pandas as pd
import pytest

from src.training.splits import (
    SPLIT_STRATEGY,
    SPLIT_VERSION,
    dedupe_exact_keys,
    grouped_split,
    normalize_subject,
    split_fingerprint,
)


@pytest.fixture
def grouped_emails() -> pd.DataFrame:
    """800 rows, 200 equally sized groups, 40 positive groups."""
    records = []
    for group in range(200):
        for copy, prefix in enumerate(("", "RE: ", "FW: RE: ", "REW: FWD: ")):
            records.append(
                {
                    "uid": f"email-{group}-{copy}",
                    "from": f"sender-{group % 7}@example.com",
                    "subject": f"{prefix}Topic {group}",
                    "date": f"2001-01-0{copy + 1}",
                    "priority": int(group % 5 == 0),
                }
            )
    return pd.DataFrame(records)


@pytest.mark.parametrize(
    ("subject", "expected"),
    [
        ("RE: FW: FWD: REW: Quarterly Update", "quarterly update"),
        (" \tRe:\nFW:  FWD:  REW:  Some\t Topic  ", "some topic"),
        ("  Some \t Topic\n", "some topic"),
        ("Topic RE: detail", "topic re: detail"),
        ("RE:FW:FWD:REW:", ""),
        (None, ""),
        (pd.NA, ""),
    ],
)
def test_normalize_subject(subject: str | None, expected: str) -> None:
    assert normalize_subject(subject) == expected


def test_dedupe_keeps_first_raw_key_and_input_unchanged() -> None:
    frame = pd.DataFrame(
        {
            "uid": ["first", "copy", "display-name", "different-date"],
            "from": [
                "a@example.com",
                "a@example.com",
                "A <a@example.com>",
                "a@example.com",
            ],
            "subject": ["Topic"] * 4,
            "date": ["2001-01-01"] * 3 + ["2001-01-02"],
            "priority": [1, 1, 1, 0],
        },
        index=[10, 11, 12, 13],
    )
    original = frame.copy(deep=True)
    deduped = dedupe_exact_keys(frame)
    assert deduped["uid"].tolist() == ["first", "display-name", "different-date"]
    assert deduped.index.tolist() == [10, 12, 13]
    pd.testing.assert_frame_equal(frame, original)


def test_grouped_split_has_no_overlap_and_covers_every_row(
    grouped_emails: pd.DataFrame,
) -> None:
    original = grouped_emails.copy(deep=True)
    splits = grouped_split(grouped_emails)
    key_sets = [
        set(zip(part["from"], part["subject"].map(normalize_subject)))
        for part in splits
    ]
    uid_sets = [set(part["uid"]) for part in splits]
    for i, left in enumerate(key_sets):
        for right in key_sets[i + 1 :]:
            assert left.isdisjoint(right)
    assert set.union(*uid_sets) == set(grouped_emails["uid"])
    assert sum(map(len, uid_sets)) == len(grouped_emails)
    assert all(list(part.columns) == list(grouped_emails.columns) for part in splits)
    pd.testing.assert_frame_equal(grouped_emails, original)


def test_grouped_split_determinism_and_seed_change(
    grouped_emails: pd.DataFrame,
) -> None:
    first = split_fingerprint(*grouped_split(grouped_emails, seed=42))
    assert first == split_fingerprint(*grouped_split(grouped_emails, seed=42))
    assert first != split_fingerprint(*grouped_split(grouped_emails, seed=7))
    assert len(first) == 64


def test_grouped_split_proportions(grouped_emails: pd.DataFrame) -> None:
    for part, expected in zip(grouped_split(grouped_emails), (0.70, 0.15, 0.15)):
        assert abs(len(part) / len(grouped_emails) - expected) <= 0.01
        assert part["priority"].sum() > 0


def test_fingerprint_ignores_row_order_but_detects_membership_changes(
    grouped_emails: pd.DataFrame,
) -> None:
    train, val, test = grouped_split(grouped_emails)
    fingerprint = split_fingerprint(train, val, test)
    assert fingerprint == split_fingerprint(
        train.iloc[::-1], val.iloc[::-1], test.iloc[::-1]
    )
    assert fingerprint != split_fingerprint(train, test, val)
    changed = train.copy()
    changed.loc[changed.index[0], "uid"] = "different-email"
    assert fingerprint != split_fingerprint(changed, val, test)


def test_fingerprint_rejects_missing_uid(grouped_emails: pd.DataFrame) -> None:
    train, val, test = grouped_split(grouped_emails)
    train.loc[train.index[0], "uid"] = None
    with pytest.raises(ValueError, match="missing uid in train"):
        split_fingerprint(train, val, test)


def test_grouped_split_handles_empty_subject_groups(
    grouped_emails: pd.DataFrame,
) -> None:
    frame = grouped_emails.copy()
    frame.loc[:3, "subject"] = None
    frame.loc[:3, "from"] = None
    train, val, test = grouped_split(frame)
    containing = [
        part for part in (train, val, test) if part["uid"].eq("email-0-0").any()
    ]
    assert len(containing) == 1
    assert containing[0]["uid"].isin([f"email-0-{i}" for i in range(4)]).sum() == 4


def test_grouped_split_rejects_too_few_groups(grouped_emails: pd.DataFrame) -> None:
    with pytest.raises(ValueError, match="at least 20"):
        grouped_split(grouped_emails.iloc[:40])


def test_grouped_split_rejects_invalid_labels(grouped_emails: pd.DataFrame) -> None:
    frame = grouped_emails.copy()
    frame.loc[0, "priority"] = 2
    with pytest.raises(ValueError, match="binary"):
        grouped_split(frame)


def test_split_policy_constants() -> None:
    assert SPLIT_STRATEGY == "stratified-group-kfold-20-14-3-3"
    assert SPLIT_VERSION == 1
