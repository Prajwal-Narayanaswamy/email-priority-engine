"""Training-serving parity tests that run in CI (fixture only).

The fixture ``tests/fixtures/serving_parity_rows.parquet`` is a small slice of the
real feature parquet (see ``scripts/build_parity_fixture.py``). These tests
validate the six reused feature functions and the lookup mechanics. They do NOT
validate the real lookup file; the local-only tests in
``test_serving_parity_local.py`` cover that.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from src.serving.lookup import build_sender_lookup
from src.serving.predictor import build_feature_row
from src.training.train import BEHAVIORAL_FEATURES

FIXTURE_PATH = Path(__file__).parent / "fixtures" / "serving_parity_rows.parquet"


@pytest.fixture(scope="module")
def fixture_rows() -> pd.DataFrame:
    """Load the committed parity rows, failing loudly if they were never built."""
    assert FIXTURE_PATH.is_file(), (
        f"{FIXTURE_PATH} is missing; build it with "
        "`python -m scripts.build_parity_fixture` and commit it."
    )
    return pd.read_parquet(FIXTURE_PATH)


def test_fixture_has_expected_columns(fixture_rows: pd.DataFrame) -> None:
    for column in ("uid", "from", "subject", "date", "priority", *BEHAVIORAL_FEATURES):
        assert column in fixture_rows.columns
    assert "body" not in fixture_rows.columns
    assert 100 <= len(fixture_rows) <= 400


def test_serving_features_equal_stored_features(fixture_rows: pd.DataFrame) -> None:
    """Every fixture row, pushed through build_feature_row, matches the stored row."""
    lookup = build_sender_lookup(fixture_rows)
    mismatches: list[str] = []
    for record in fixture_rows.to_dict("records"):
        row, _ = build_feature_row(
            record["from"], record["subject"], record["date"], lookup
        )
        built = row.iloc[0]
        for feature in BEHAVIORAL_FEATURES:
            if built[feature] != record[feature]:
                mismatches.append(
                    f"{record['uid']}: {feature} serving={built[feature]} "
                    f"stored={record[feature]}"
                )
    assert not mismatches, "\n".join(mismatches[:20])
