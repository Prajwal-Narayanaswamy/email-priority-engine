"""Model input columns shared by training and serving.

Kept in a small module so the serving image does not import the training code
(and with it matplotlib and the evaluation stack).
"""

from __future__ import annotations

from src.features.features import FEATURE_COLUMNS

#: Tabular features that do not encode the ADR-005 label rule.
BEHAVIORAL_FEATURES: tuple[str, ...] = tuple(
    c for c in FEATURE_COLUMNS if c != "urgency_keyword_count"
)
