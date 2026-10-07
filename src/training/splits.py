"""Pure deduplication and shared grouped splitting policy from ADR-012.

No MLflow, Prefect, or IO. Deduplication is a separate operation: callers must
apply it before feature engineering, not silently during split construction.
Sender keys remain raw; only subjects are normalized for grouping.
"""

from __future__ import annotations

import hashlib
import json
import re

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

SPLIT_STRATEGY: str = "stratified-group-kfold-20-14-3-3"
SPLIT_VERSION: int = 1

_EXACT_KEYS: tuple[str, ...] = ("from", "subject", "date")
_LEADING_MARKERS = re.compile(r"^(?:\s*(?:re|fw|fwd|rew):\s*)+")


def normalize_subject(subject: str | None) -> str:
    """Lowercase, strip stacked leading reply/forward markers, normalize spaces.

    Missing subjects map to the empty string. Interior markers are preserved;
    sender normalization and urgency-keyword removal are not part of this policy.
    """
    if subject is None or pd.isna(subject):
        return ""
    lowered = subject.lower()
    stripped = _LEADING_MARKERS.sub("", lowered)
    return " ".join(stripped.split())


def dedupe_exact_keys(df: pd.DataFrame) -> pd.DataFrame:
    """Keep the first row per raw (from, subject, date), preserving input order.

    Returns an independent frame with original indices and all columns intact.
    Missing key values follow pandas drop_duplicates semantics. Apply BEFORE
    engineer_features so sender_frequency counts unique emails (ADR-012).

    Raises:
        KeyError: if an exact-key column is missing.
    """
    return df.drop_duplicates(subset=list(_EXACT_KEYS), keep="first").copy()


def grouped_split(
    df: pd.DataFrame, seed: int = 42
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Assign 20 stratified group folds to train (0-13), val (14-16), test (17-19).

    Groups are raw sender plus normalized subject, including missing senders
    and empty subjects. Group IDs are assigned in first-appearance order, as
    in the ADR-012 feasibility check; replay requires the same input ordering.
    Group sizes are indivisible, so 70/15/15 proportions are approximate.
    Outputs preserve input ordering/indices and columns without mutating df.

    Raises:
        KeyError: if from, subject, or priority is absent.
        ValueError: for non-binary/missing labels, fewer than 20 groups, or
            data that sklearn cannot stratify into 20 folds.
    """
    labels = df["priority"].to_numpy()
    if len(labels) == 0 or not np.all(np.isin(labels, (0, 1))):
        raise ValueError("grouped_split requires nonempty binary 0/1 labels")
    group_frame = pd.DataFrame(
        {
            "from": df["from"].to_numpy(),
            "normalized_subject": df["subject"].map(normalize_subject).to_numpy(),
        }
    )
    groups = (
        group_frame.groupby(["from", "normalized_subject"], dropna=False, sort=False)
        .ngroup()
        .to_numpy()
    )
    if len(np.unique(groups)) < 20:
        raise ValueError("grouped_split requires at least 20 distinct groups")

    splitter = StratifiedGroupKFold(n_splits=20, shuffle=True, random_state=seed)
    fold_ids = np.full(len(df), -1, dtype=int)
    for fold, (_, held_out) in enumerate(
        splitter.split(np.zeros((len(df), 1), dtype=np.uint8), labels, groups)
    ):
        fold_ids[held_out] = fold
    return (
        df.iloc[np.flatnonzero(fold_ids < 14)].copy(),
        df.iloc[np.flatnonzero((fold_ids >= 14) & (fold_ids < 17))].copy(),
        df.iloc[np.flatnonzero(fold_ids >= 17)].copy(),
    )


def split_fingerprint(
    train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame
) -> str:
    """SHA-256 of the sorted UID membership of each named split.

    UID values are canonicalized to strings. JSON preserves split boundaries
    and escapes UID contents unambiguously; UTF-8 encoding and sorted object
    keys make the hash stable across processes and independent of row order.
    This fingerprints membership only, not feature values or split provenance.

    Raises:
        KeyError: if uid is absent in any split.
        ValueError: if any uid is missing.
    """
    membership = {}
    for name, frame in (("train", train), ("val", val), ("test", test)):
        if frame["uid"].isna().any():
            raise ValueError(f"missing uid in {name} split")
        membership[name] = sorted(frame["uid"].astype(str).tolist())
    payload = json.dumps(
        membership, sort_keys=True, ensure_ascii=False, separators=(",", ":")
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
