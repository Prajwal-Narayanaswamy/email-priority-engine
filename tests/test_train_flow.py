"""Tests for src.flows.train_flow — MLflow mocked, no live tracking server."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest
from prefect.testing.utilities import prefect_test_harness

from src.flows.train_flow import train_flow
from src.tasks import train_tasks


@pytest.fixture
def mock_mlflow(monkeypatch: pytest.MonkeyPatch) -> MagicMock:
    """Replace the whole `mlflow` module reference in the task module."""
    fake = MagicMock()
    run = MagicMock()
    run.info.run_id = "flow-test-run"
    fake.active_run.return_value = run
    fake.start_run.return_value.__enter__.return_value = run
    monkeypatch.setattr(train_tasks, "mlflow", fake)
    return fake


def test_train_flow_runs_both_variants(
    features_parquet: Path, mock_mlflow: MagicMock
) -> None:
    with prefect_test_harness():
        results = train_flow(
            input_path=str(features_parquet), limit=200, n_estimators=20
        )

    assert [r["variant"] for r in results] == [
        "with-keyword-features",
        "behavioral-only",
    ]
    assert all(r["run_id"] == "flow-test-run" for r in results)


def test_train_flow_runs_selected_variant_only(
    features_parquet: Path, mock_mlflow: MagicMock
) -> None:
    with prefect_test_harness():
        results = train_flow(
            input_path=str(features_parquet),
            limit=200,
            n_estimators=20,
            variants=["behavioral-only"],
        )

    assert len(results) == 1
    assert results[0]["variant"] == "behavioral-only"
    assert mock_mlflow.start_run.call_count == 1


def test_train_flow_shares_one_split_across_variants(
    features_parquet: Path, mock_mlflow: MagicMock
) -> None:
    with prefect_test_harness():
        results = train_flow(
            input_path=str(features_parquet), limit=200, n_estimators=20
        )

    first, second = results
    for key in ("train_rows", "val_rows", "test_rows"):
        assert first[key] == second[key]
    assert first["split_fingerprint"] == second["split_fingerprint"]
    assert first["scale_pos_weight"] == second["scale_pos_weight"]


def test_train_flow_behavioral_variant_has_fewer_columns(
    features_parquet: Path, mock_mlflow: MagicMock
) -> None:
    with prefect_test_harness():
        results = train_flow(
            input_path=str(features_parquet), limit=200, n_estimators=20
        )

    by_name = {r["variant"]: r for r in results}
    assert (
        by_name["behavioral-only"]["n_matrix_columns"]
        < by_name["with-keyword-features"]["n_matrix_columns"]
    )


def test_train_flow_returns_metrics_in_range(
    features_parquet: Path, mock_mlflow: MagicMock
) -> None:
    with prefect_test_harness():
        results = train_flow(
            input_path=str(features_parquet), limit=200, n_estimators=20
        )

    for result in results:
        for key in ("test_f1", "test_precision", "test_recall"):
            assert 0.0 <= result[key] <= 1.0
