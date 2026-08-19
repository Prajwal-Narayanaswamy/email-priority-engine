"""Heuristic email priority labeling (ADR-005).

Enron has no ground-truth priority labels. A message is labeled
priority=1 if the sender is a known Enron leader OR the subject contains
an urgency keyword. This is a documented limitation of the dataset.
"""

from __future__ import annotations

import logging

import pandas as pd

log = logging.getLogger(__name__)

URGENCY_KEYWORDS: tuple[str, ...] = (
    "urgent",
    "asap",
    "immediately",
    "deadline",
    "action required",
    "time sensitive",
    "eod",
    "eow",
    "by end of",
)

LEADERSHIP_SENDERS: frozenset[str] = frozenset(
    {
        "kenneth.lay@enron.com",
        "ken.lay@enron.com",
        "jeff.skilling@enron.com",
        "jeffrey.skilling@enron.com",
        "andrew.fastow@enron.com",
        "greg.whalley@enron.com",
        "mark.frevert@enron.com",
        "lou.pai@enron.com",
    }
)

PRIORITY_COLUMN = "priority"


def label_email(subject: str | None, sender: str | None) -> int:
    """Return 1 if the email is high priority, else 0.

    High priority when the sender is in LEADERSHIP_SENDERS or the subject
    contains any URGENCY_KEYWORDS (case-insensitive substring match).
    """
    if (sender or "").strip().lower() in LEADERSHIP_SENDERS:
        return 1
    subject_lower = (subject or "").lower()
    return 1 if any(keyword in subject_lower for keyword in URGENCY_KEYWORDS) else 0


def apply_labels(
    df: pd.DataFrame,
    subject_col: str = "subject",
    sender_col: str = "from",
) -> pd.DataFrame:
    """Return a copy of df with the binary priority column added."""
    labeled = df.copy()
    labeled[PRIORITY_COLUMN] = [
        label_email(subject, sender)
        for subject, sender in zip(labeled[subject_col], labeled[sender_col])
    ]
    positive = int(labeled[PRIORITY_COLUMN].sum())
    log.info(
        "Applied labels: %d/%d emails marked high priority", positive, len(labeled)
    )
    return labeled
