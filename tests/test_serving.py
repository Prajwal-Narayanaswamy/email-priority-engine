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
    build_header,
    build_sender_lookup,
    load_sender_lookup,
    save_sender_lookup,
    sender_frequency_for,
)
from src.serving.predictor import CAVEATS, PREDICT_NOTE, Predictor, build_feature_row
from src.training.train import BEHAVIORAL_FEATURES
from tests.serving_helpers import StubModel, make_metadata


@pytest.fixture
def lookup() -> dict[str, int]:
    return {"kenneth.lay@enron.com": 500, "a@b.com": 1}


@pytest.fixture
def client(lookup: dict[str, int]) -> Iterator[TestClient]:
    app.state.predictor = Predictor(StubModel(), make_metadata(), lookup)
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
    save_sender_lookup(built, path, build_header("b" * 64, 3))
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
    assert client.get("/health").json() == {
        "status": "ok",
        "model_run_id": "run",
        "model_version": "3",
    }
    info = client.get("/model-info").json()
    assert info["feature_columns"] == list(BEHAVIORAL_FEATURES)
    assert info["threshold"] == 0.99
    assert info["sender_lookup_size"] == 2
    assert "leadership" in info["caveats"]
    assert info["model_name"] == "email-priority"
    assert info["model_version"] == "3"
    assert info["alias"] == "champion"
    assert info["model_uri"] == "models:/email-priority/3"
    assert info["threshold_source_run"] == "eval-run"
    assert info["tags"] == make_metadata().tags


def test_predict_unseen_sender_not_flagged(client: TestClient) -> None:
    body = client.post(
        "/predict", json={"from": "abcd@duck.com", "subject": "hi", "date": None}
    ).json()
    assert body["flagged"] is False
    assert body["sender_seen_in_training"] is False
    assert body["features"]["sender_frequency"] == UNSEEN_SENDER_FREQUENCY
    assert body["note"] == PREDICT_NOTE
    assert "caveats" not in body


def test_predict_leadership_sender_flagged(client: TestClient) -> None:
    leader = next(iter(LEADERSHIP_SENDERS))
    body = client.post("/predict", json={"from": leader, "subject": "x"}).json()
    assert body["flagged"] is True
    assert body["score"] >= body["threshold"]
    assert "probability" not in body


def test_omitted_date_field_gives_sentinel_features(client: TestClient) -> None:
    body = client.post("/predict", json={"from": "a@b.com", "subject": "hi"})
    assert body.status_code == 200
    assert body.json()["features"]["hour_of_day"] == -1
    assert body.json()["features"]["day_of_week"] == -1


@pytest.mark.parametrize("date", [None, "garbage", "", "32 Foo 2001"])
def test_missing_or_unparseable_date_gives_sentinel(
    lookup: dict[str, int], date: str | None
) -> None:
    row, _ = build_feature_row("a@b.com", "hi", date, lookup)
    assert row.iloc[0]["hour_of_day"] == -1
    assert row.iloc[0]["day_of_week"] == -1


@pytest.mark.parametrize("sender", [None, "", "   "])
def test_null_or_empty_sender_gives_zero_frequency(
    lookup: dict[str, int], sender: str | None
) -> None:
    row, seen = build_feature_row(sender, "hi", None, lookup)
    assert row.iloc[0]["sender_frequency"] == 0
    assert not seen


def test_unseen_sender_gives_frequency_one(lookup: dict[str, int]) -> None:
    row, seen = build_feature_row("new.person@example.com", "hi", None, lookup)
    assert row.iloc[0]["sender_frequency"] == UNSEEN_SENDER_FREQUENCY == 1
    assert not seen


def test_caveat_wording(client: TestClient) -> None:
    text = client.get("/model-info").json()["caveats"]
    assert text == CAVEATS
    assert "ADR-013" in text and "ADR-005" in text
    assert "not a genuine urgency predictor" in text
    assert "uncalibrated" in text and "scale_pos_weight" in text
    assert "not a probability" in text
    # The test-split statement is scoped to that split and not a guarantee.
    assert "On the held-out test split" in text
    assert "60 of 60" in text and "non-leadership F1 0" in text
    assert "not a guarantee" in text
    assert "can score below the threshold" in text
    # The old unscoped claim and internal run text are gone.
    assert "flag equals" not in text
    assert "with-keyword-features" not in text and "behavioral-only" not in text


def test_predict_returns_only_a_one_line_note(client: TestClient) -> None:
    body = client.post("/predict", json={"subject": "x"}).json()
    assert (
        body["note"] == PREDICT_NOTE == "Reference model; see /model-info for caveats."
    )
    assert "\n" not in body["note"]
    assert CAVEATS not in str(body)


def test_leadership_sender_can_score_below_threshold(
    lookup: dict[str, int],
) -> None:
    """The stated limitation is real: nothing forces a leadership flag."""

    class LowModel:
        def predict_proba(self, frame: pd.DataFrame) -> np.ndarray:
            return np.array([[0.4554, 0.5446]])

    predictor = Predictor(LowModel(), make_metadata(tuned_threshold="0.9943"), lookup)
    score, flagged, _, features = predictor.predict(
        "kenneth.lay@enron.com", "RE: budget?", None
    )
    assert features["sender_is_leadership"] == 1.0
    assert score < predictor.threshold and flagged is False


def test_503_when_model_missing() -> None:
    app.state.predictor = None
    # No context manager: lifespan (and any real model load) is not run.
    plain = TestClient(app)
    assert plain.get("/health").status_code == 503
