"""Tests for the serving layer (no MLflow; a stub model stands in)."""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from src.labels.labeling import LEADERSHIP_SENDERS
from src.serving.app import app
from src.serving.lookup import (
    UNSEEN_SENDER_FREQUENCY,
    build_sender_lookup,
    load_sender_lookup,
    save_sender_lookup,
    sender_frequency_for,
)
from src.serving.predictor import Predictor, build_feature_row
from src.training.train import BEHAVIORAL_FEATURES


class StubModel:
    """Returns a high probability only when sender_is_leadership == 1."""

    def predict_proba(self, frame: pd.DataFrame) -> np.ndarray:
        p = frame["sender_is_leadership"].to_numpy(dtype=float) * 0.999
        return np.column_stack([1 - p, p])


@pytest.fixture
def lookup() -> dict[str, int]:
    return {"kenneth.lay@enron.com": 500, "a@b.com": 1}


@pytest.fixture
def client(lookup: dict[str, int]) -> Iterator[TestClient]:
    app.state.predictor = Predictor(StubModel(), 0.99, lookup, "run", "eval")
    with TestClient(app) as test_client:
        yield test_client
    app.state.predictor = None


def test_build_lookup_normalizes_and_roundtrips(tmp_path) -> None:
    df = pd.DataFrame(
        {
            "from": ['"Lay, Kenneth" <Kenneth.Lay@enron.com>', "x@y.com", None],
            "sender_frequency": [7, 2, 0],
        }
    )
    built = build_sender_lookup(df)
    assert built == {"kenneth.lay@enron.com": 7, "x@y.com": 2}
    path = tmp_path / "l.json"
    save_sender_lookup(built, path)
    assert load_sender_lookup(path) == built


def test_build_lookup_rejects_inconsistent_frequency() -> None:
    df = pd.DataFrame({"from": ["a@b.com", "A@B.com"], "sender_frequency": [1, 2]})
    with pytest.raises(ValueError):
        build_sender_lookup(df)


def test_sender_frequency_cases(lookup: dict[str, int]) -> None:
    assert sender_frequency_for("kenneth.lay@enron.com", lookup) == (500, True)
    assert sender_frequency_for("abcd@duck.com", lookup) == (
        UNSEEN_SENDER_FREQUENCY,
        False,
    )
    assert sender_frequency_for(None, lookup) == (0, False)
    assert sender_frequency_for("", lookup) == (0, False)


def test_feature_row_columns_and_values(lookup: dict[str, int]) -> None:
    row, seen = build_feature_row(
        "kenneth.lay@enron.com", "RE: ok?", "Mon, 14 May 2001 16:39:00 -0700", lookup
    )
    assert list(row.columns) == list(BEHAVIORAL_FEATURES)
    assert seen
    values = row.iloc[0]
    assert values["sender_frequency"] == 500
    assert values["has_question_mark"] == 1
    assert values["thread_depth"] == 1
    assert values["subject_length"] == len("RE: ok?")


def test_feature_row_bad_date_and_nulls(lookup: dict[str, int]) -> None:
    row, seen = build_feature_row(None, None, "garbage", lookup)
    assert not seen
    assert row.iloc[0]["hour_of_day"] == -1
    assert row.iloc[0]["day_of_week"] == -1
    assert row.iloc[0]["subject_length"] == 0


def test_health_and_model_info(client: TestClient) -> None:
    assert client.get("/health").json() == {"status": "ok", "model_run_id": "run"}
    info = client.get("/model-info").json()
    assert info["feature_columns"] == list(BEHAVIORAL_FEATURES)
    assert info["threshold"] == 0.99
    assert info["sender_lookup_size"] == 2
    assert "leadership" in info["caveats"]


def test_predict_unseen_sender_not_flagged(client: TestClient) -> None:
    body = client.post(
        "/predict", json={"from": "abcd@duck.com", "subject": "hi", "date": None}
    ).json()
    assert body["flagged"] is False
    assert body["sender_seen_in_training"] is False
    assert body["features"]["sender_frequency"] == UNSEEN_SENDER_FREQUENCY
    assert "caveats" in body


def test_predict_leadership_sender_flagged(client: TestClient) -> None:
    leader = next(iter(LEADERSHIP_SENDERS))
    body = client.post("/predict", json={"from": leader, "subject": "x"}).json()
    assert body["flagged"] is True
    assert body["probability"] >= body["threshold"]


def test_503_when_model_missing() -> None:
    app.state.predictor = None
    # No context manager: lifespan (and any real model load) is not run.
    plain = TestClient(app)
    assert plain.get("/health").status_code == 503
