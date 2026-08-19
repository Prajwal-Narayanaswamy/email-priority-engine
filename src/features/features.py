"""Feature engineering for the email priority classifier.

Pure functions — no Prefect dependency. Prefect wrappers live in
src/tasks/ and src/flows/ (see ADR-008).
"""

from __future__ import annotations

import re

import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer

from src.labels.labeling import LEADERSHIP_SENDERS, URGENCY_KEYWORDS

_EMAIL_RE = re.compile(r"[\w.+-]+@[\w.-]+")
_THREAD_START_RE = re.compile(r"^\s*(?:FWD|FW|REW|RE)(?:\[(\d+)\])?\s*:", re.IGNORECASE)

FEATURE_COLUMNS: tuple[str, ...] = (
    "sender_frequency",
    "sender_is_leadership",
    "hour_of_day",
    "day_of_week",
    "urgency_keyword_count",
    "subject_length",
    "has_question_mark",
    "thread_depth",
)

KEEP_COLUMNS: tuple[str, ...] = ("uid", "from", "subject", "date", "priority")


def normalize_sender(raw: str | None) -> str:
    """Extract the addr-spec from a From header, falling back to the raw value.

    Handles display-name forms such as `"Lay, Kenneth" <kenneth.lay@enron.com>`
    so the same person maps to one canonical address.
    """
    if raw is None:
        return ""
    match = _EMAIL_RE.search(raw)
    if match:
        return match.group(0).lower()
    return raw.strip().lower()


def parse_dates(date_series: pd.Series) -> pd.Series:
    """Parse date strings to UTC datetime64, coercing malformed values to NaT."""
    return pd.to_datetime(date_series, utc=True, errors="coerce")


def feature_sender_frequency(df: pd.DataFrame, sender_col: str = "from") -> pd.Series:
    """Return the global per-sender email count for each row (normalized sender).

    Known limitation: frequency computed on full dataset including test split.
    Minor leakage, documented, acceptable for v1. Fix in v2 by computing on
    train only.
    """
    senders = df[sender_col].map(normalize_sender)
    counts = senders.value_counts()
    result = senders.map(counts).fillna(0).astype("int64")
    missing = df[sender_col].isna() | (senders == "")
    result.loc[missing] = 0
    return result


def feature_sender_is_leadership(
    df: pd.DataFrame, sender_col: str = "from"
) -> pd.Series:
    """Return 1 if the normalized sender is in LEADERSHIP_SENDERS, else 0."""
    return df[sender_col].map(normalize_sender).isin(LEADERSHIP_SENDERS).astype("int64")


def feature_hour_of_day(df: pd.DataFrame, date_col: str = "date") -> pd.Series:
    """Return the UTC hour (0-23); -1 when the date cannot be parsed."""
    parsed = parse_dates(df[date_col])
    return parsed.dt.hour.fillna(-1).astype("int64")


def feature_day_of_week(df: pd.DataFrame, date_col: str = "date") -> pd.Series:
    """Return the day of week (0=Monday..6=Sunday); -1 when unparseable."""
    parsed = parse_dates(df[date_col])
    return parsed.dt.dayofweek.fillna(-1).astype("int64")


def feature_urgency_keyword_count(
    df: pd.DataFrame, subject_col: str = "subject"
) -> pd.Series:
    """Return the count of distinct urgency keywords in the subject.

    Matching is case-insensitive.
    """

    def _count(subject: str | None) -> int:
        lowered = (subject or "").lower()
        return sum(1 for keyword in URGENCY_KEYWORDS if keyword in lowered)

    return df[subject_col].map(_count).astype("int64")


def feature_subject_length(df: pd.DataFrame, subject_col: str = "subject") -> pd.Series:
    """Return the subject character length; 0 for null or empty subjects."""
    return df[subject_col].fillna("").str.len().astype("int64")


def feature_has_question_mark(
    df: pd.DataFrame, subject_col: str = "subject"
) -> pd.Series:
    """Return 1 if the subject contains '?', else 0."""
    return df[subject_col].fillna("").str.contains("?", regex=False).astype("int64")


def feature_thread_depth(df: pd.DataFrame, subject_col: str = "subject") -> pd.Series:
    """Return the thread depth from leading RE:/FW:/FWD:/REW: markers.

    Bracket counts are honored (`RE[2]:` counts as 2). Subjects without
    leading markers, null, or empty subjects return 0.
    """

    def _depth(subject: str | None) -> int:
        if not subject:
            return 0
        depth = 0
        rest = subject
        while True:
            match = _THREAD_START_RE.match(rest)
            if not match:
                break
            depth += int(match.group(1)) if match.group(1) else 1
            rest = rest[match.end() :]
        return depth

    return df[subject_col].map(_depth).astype("int64")


def fit_subject_vectorizer(
    subjects: pd.Series, *, max_features: int = 100
) -> TfidfVectorizer:
    """Fit a TF-IDF vectorizer on the provided subjects.

    Fit on the TRAINING split only. Never fit on the full dataset (leakage).
    """
    vectorizer = TfidfVectorizer(max_features=max_features, stop_words="english")
    vectorizer.fit(subjects.fillna("").astype(str))
    return vectorizer


def transform_subjects(vectorizer: TfidfVectorizer, subjects: pd.Series) -> csr_matrix:
    """Transform subjects into a scipy sparse TF-IDF matrix (N x max_features).

    Returns a scipy sparse matrix, never dense — a dense 517,401 x 100 float32
    array is ~200MB and risks OOM on a 5.6GB host (ADR-008). Null subjects are
    treated as empty strings (all-zero rows).
    """
    return vectorizer.transform(subjects.fillna("").astype(str))


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add the 8 tabular features and keep only columns needed downstream.

    Drops 'body' and 'to' — carrying the 511MB body text into the feature
    parquet would bloat it. TF-IDF is not computed here; train_flow applies
    it via fit_subject_vectorizer/transform_subjects.
    """
    engineered = df.copy()
    engineered["sender_frequency"] = feature_sender_frequency(engineered)
    engineered["sender_is_leadership"] = feature_sender_is_leadership(engineered)
    engineered["hour_of_day"] = feature_hour_of_day(engineered)
    engineered["day_of_week"] = feature_day_of_week(engineered)
    engineered["urgency_keyword_count"] = feature_urgency_keyword_count(engineered)
    engineered["subject_length"] = feature_subject_length(engineered)
    engineered["has_question_mark"] = feature_has_question_mark(engineered)
    engineered["thread_depth"] = feature_thread_depth(engineered)
    columns = [c for c in KEEP_COLUMNS + FEATURE_COLUMNS if c in engineered.columns]
    return engineered[columns]
