"""Tests for src.tasks.evaluate_tasks — MLflow and Evidently fully mocked."""

from __future__ import annotations

import copy
import logging
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

from src.evaluation.evaluate import (
    SplitMismatchError,
    metrics_at_threshold,
    pr_auc,
    sweep_thresholds,
)
from src.tasks import evaluate_tasks
from src.tasks.evaluate_tasks import (
    DEFAULT_SOURCE_RUN_ID,
    EVAL_RUN_NAME,
    REFERENCE_CAP,
    DiagnosticMismatchError,
    evaluate_variant_task,
    evidently_baseline_task,
    load_model_task,
    load_split_task,
    retrain_diagnostic_task,
    summarize_drift,
)
from src.training.train import (
    BEHAVIORAL_FEATURES,
    SPLIT_SEED,
    TARGET,
    build_matrices,
    build_model,
    stratified_split,
    train_model,
)

N_ROWS = 400


@pytest.fixture
def eval_frame() -> pd.DataFrame:
    """Synthetic frame with the target plus the 7 behavioral features."""
    rng = np.random.default_rng(0)
    frame = pd.DataFrame(
        {
            TARGET: (rng.random(N_ROWS) < 0.15).astype(int),
            "sender_frequency": rng.integers(0, 50, size=N_ROWS),
            "sender_is_leadership": rng.integers(0, 2, size=N_ROWS),
            "hour_of_day": rng.integers(0, 24, size=N_ROWS),
            "day_of_week": rng.integers(0, 7, size=N_ROWS),
            "subject_length": rng.integers(5, 120, size=N_ROWS),
            "has_question_mark": rng.integers(0, 2, size=N_ROWS),
            "thread_depth": rng.integers(0, 6, size=N_ROWS),
        }
    )
    assert set(BEHAVIORAL_FEATURES) == set(frame.columns) - {TARGET}
    return frame


@pytest.fixture
def eval_splits(eval_frame: pd.DataFrame) -> dict[str, pd.DataFrame]:
    train_df, val_df, test_df = stratified_split(eval_frame, seed=11)
    return {"train": train_df, "val": val_df, "test": test_df}


@pytest.fixture
def tiny_model(eval_splits: dict[str, pd.DataFrame]):
    """Small real LightGBM model with early stopping (fast, deterministic)."""
    train_df = eval_splits["train"]
    val_df = eval_splits["val"]
    x_train = build_matrices(
        train_df, feature_columns=BEHAVIORAL_FEATURES, use_tfidf=False
    )
    x_val = build_matrices(val_df, feature_columns=BEHAVIORAL_FEATURES, use_tfidf=False)
    model = build_model(n_estimators=20, random_state=42)
    model, _ = train_model(
        model,
        x_train,
        train_df[TARGET],
        x_val,
        val_df[TARGET],
        stopping_rounds=5,
    )
    return model


@pytest.fixture
def mock_mlflow(monkeypatch: pytest.MonkeyPatch) -> MagicMock:
    """Replace the whole `mlflow` module reference in the task module."""
    fake = MagicMock()
    run = MagicMock()
    run.info.run_id = "eval-run-id"
    fake.active_run.return_value = run
    fake.start_run.return_value.__enter__.return_value = run
    monkeypatch.setattr(evaluate_tasks, "mlflow", fake)
    return fake


@pytest.fixture
def quiet_logger(monkeypatch: pytest.MonkeyPatch) -> None:
    """`get_run_logger` needs a Prefect run context; swap in a stdlib logger."""
    monkeypatch.setattr(
        evaluate_tasks, "get_run_logger", lambda: logging.getLogger("evaluate_test")
    )


@pytest.fixture
def fake_client_cls(monkeypatch: pytest.MonkeyPatch):
    """Configurable MlflowClient stand-in (params + logged models per test)."""

    class FakeClient:
        params: dict[str, Any] = {}
        models: list[Any] = []
        last_filter: str = ""

        def __init__(self, tracking_uri: str | None = None) -> None:
            self.tracking_uri = tracking_uri

        def get_experiment_by_name(self, name: str) -> Any:
            return SimpleNamespace(experiment_id="1")

        def get_run(self, run_id: str) -> Any:
            return SimpleNamespace(data=SimpleNamespace(params=dict(self.params)))

        def search_logged_models(
            self, experiment_ids: list[str] | None = None, filter_string: str = ""
        ) -> list[Any]:
            type(self).last_filter = filter_string
            return list(self.models)

    monkeypatch.setattr(evaluate_tasks, "MlflowClient", FakeClient)
    return FakeClient


def split_params(
    splits: dict[str, pd.DataFrame], variant: str = "behavioral-only"
) -> dict[str, str]:
    """Source-run params matching the given splits (stringified, like MLflow)."""
    val_df = splits["val"]
    test_df = splits["test"]
    return {
        "variant": variant,
        "val_rows": str(len(val_df)),
        "val_positives": str(int((val_df[TARGET] == 1).sum())),
        "test_rows": str(len(test_df)),
        "test_positives": str(int((test_df[TARGET] == 1).sum())),
    }


# --------------------------------------------------------------------------
# load_split_task
# --------------------------------------------------------------------------


def test_load_split_task_reads_only_needed_columns(
    tmp_path: Path, eval_frame: pd.DataFrame, quiet_logger: None
) -> None:
    bloated = eval_frame.copy()
    bloated["subject"] = "lunch"
    bloated["body"] = "see you at noon"
    bloated["uid"] = "0"
    path = tmp_path / "features.parquet"
    bloated.to_parquet(path)
    splits = load_split_task.fn(path)
    for split_df in splits.values():
        assert set(split_df.columns) == {TARGET, *BEHAVIORAL_FEATURES}
    assert sum(len(split_df) for split_df in splits.values()) == N_ROWS


def test_load_split_task_is_deterministic(
    tmp_path: Path, eval_frame: pd.DataFrame, quiet_logger: None
) -> None:
    path = tmp_path / "features.parquet"
    eval_frame.to_parquet(path)
    first = load_split_task.fn(path)
    second = load_split_task.fn(path)
    assert first["val"].equals(second["val"])
    assert first["test"].equals(second["test"])


# --------------------------------------------------------------------------
# load_model_task
# --------------------------------------------------------------------------


def test_load_model_task_loads_logged_model(
    fake_client_cls: Any, mock_mlflow: MagicMock, quiet_logger: None
) -> None:
    sentinel = MagicMock()
    fake_client_cls.models = [SimpleNamespace(model_id="m-test", name="model")]
    mock_mlflow.sklearn.load_model.return_value = sentinel
    model = load_model_task.fn(DEFAULT_SOURCE_RUN_ID)
    assert model is sentinel
    assert DEFAULT_SOURCE_RUN_ID in fake_client_cls.last_filter
    mock_mlflow.sklearn.load_model.assert_called_once_with("models:/m-test")


def test_load_model_task_raises_on_zero_models(
    fake_client_cls: Any, mock_mlflow: MagicMock, quiet_logger: None
) -> None:
    fake_client_cls.models = []
    with pytest.raises(ValueError, match="exactly one"):
        load_model_task.fn(DEFAULT_SOURCE_RUN_ID)


def test_load_model_task_raises_on_two_models(
    fake_client_cls: Any, mock_mlflow: MagicMock, quiet_logger: None
) -> None:
    fake_client_cls.models = [
        SimpleNamespace(model_id="m-a", name="model"),
        SimpleNamespace(model_id="m-b", name="model"),
    ]
    with pytest.raises(ValueError, match="exactly one"):
        load_model_task.fn(DEFAULT_SOURCE_RUN_ID)


# --------------------------------------------------------------------------
# evaluate_variant_task
# --------------------------------------------------------------------------


def test_evaluate_variant_task_gate_aborts_on_mismatch(
    eval_splits: dict[str, pd.DataFrame],
    fake_client_cls: Any,
    mock_mlflow: MagicMock,
    quiet_logger: None,
) -> None:
    fake_client_cls.params = split_params(eval_splits)
    fake_client_cls.params["val_rows"] = str(len(eval_splits["val"]) + 1)
    with pytest.raises(SplitMismatchError, match="val_rows"):
        evaluate_variant_task.fn(
            MagicMock(), eval_splits, source_run_id=DEFAULT_SOURCE_RUN_ID
        )


def test_evaluate_variant_task_rejects_wrong_variant(
    eval_splits: dict[str, pd.DataFrame],
    fake_client_cls: Any,
    mock_mlflow: MagicMock,
    quiet_logger: None,
) -> None:
    fake_client_cls.params = split_params(eval_splits, variant="with-keyword-features")
    with pytest.raises(ValueError, match="behavioral-only"):
        evaluate_variant_task.fn(
            MagicMock(), eval_splits, source_run_id=DEFAULT_SOURCE_RUN_ID
        )


def test_evaluate_variant_task_logs_eval_run_tags_and_params(
    tiny_model: Any,
    eval_splits: dict[str, pd.DataFrame],
    fake_client_cls: Any,
    mock_mlflow: MagicMock,
    quiet_logger: None,
) -> None:
    fake_client_cls.params = split_params(eval_splits)
    summary = evaluate_variant_task.fn(
        tiny_model, eval_splits, source_run_id=DEFAULT_SOURCE_RUN_ID
    )

    mock_mlflow.start_run.assert_called_once_with(run_name=EVAL_RUN_NAME)
    tags = mock_mlflow.set_tags.call_args[0][0]
    assert tags["source_run_id"] == DEFAULT_SOURCE_RUN_ID
    params = mock_mlflow.log_params.call_args[0][0]
    assert params["source_run_id"] == DEFAULT_SOURCE_RUN_ID
    assert params["tuned_threshold"] == summary["tuned_threshold"]
    assert params["evidently_reference_cap"] == REFERENCE_CAP
    logged_metrics = mock_mlflow.log_metrics.call_args[0][0]
    for key in (
        "val_f1",
        "test_f1",
        "val_pr_auc",
        "test_pr_auc",
        "val_roc_auc",
        "test_roc_auc",
        "test_flag_rate",
        "test_prevalence",
    ):
        assert key in logged_metrics
    assert summary["run_id"] == "eval-run-id"


def test_evaluate_variant_task_threshold_from_val_unchanged_on_test(
    tiny_model: Any,
    eval_splits: dict[str, pd.DataFrame],
    fake_client_cls: Any,
    mock_mlflow: MagicMock,
    quiet_logger: None,
) -> None:
    fake_client_cls.params = split_params(eval_splits)
    summary = evaluate_variant_task.fn(
        tiny_model, eval_splits, source_run_id=DEFAULT_SOURCE_RUN_ID
    )

    val_df = eval_splits["val"]
    test_df = eval_splits["test"]
    x_val = build_matrices(val_df, feature_columns=BEHAVIORAL_FEATURES, use_tfidf=False)
    x_test = build_matrices(
        test_df, feature_columns=BEHAVIORAL_FEATURES, use_tfidf=False
    )
    val_proba = np.asarray(tiny_model.predict_proba(x_val)[:, 1], dtype=float)
    test_proba = np.asarray(tiny_model.predict_proba(x_test)[:, 1], dtype=float)
    frame = sweep_thresholds(val_df[TARGET].to_numpy(), val_proba)
    expected_best = float(frame.loc[frame["f1"].idxmax(), "threshold"])
    assert summary["tuned_threshold"] == expected_best

    expected_test = metrics_at_threshold(
        test_df[TARGET].to_numpy(), test_proba, threshold=expected_best
    )
    assert summary["test"]["f1"] == expected_test["f1"]
    assert summary["test"]["precision"] == expected_test["precision"]
    assert summary["test"]["recall"] == expected_test["recall"]


def test_evaluate_variant_task_iteration_table_uses_verified_counts(
    tiny_model: Any,
    eval_splits: dict[str, pd.DataFrame],
    fake_client_cls: Any,
    mock_mlflow: MagicMock,
    quiet_logger: None,
) -> None:
    fake_client_cls.params = split_params(eval_splits)
    summary = evaluate_variant_task.fn(
        tiny_model, eval_splits, source_run_id=DEFAULT_SOURCE_RUN_ID
    )

    total = int(tiny_model.booster_.num_trees())
    best = int(tiny_model.best_iteration_)
    assert summary["loaded_num_trees"] == total
    assert summary["source_best_iteration"] == best
    expected = sorted({1, 10, 50, best, total} & set(range(1, total + 1)))
    assert [row["iteration"] for row in summary["iterations"]] == expected
    logged_metrics = mock_mlflow.log_metrics.call_args[0][0]
    for count in expected:
        assert f"iter_{count}_pr_auc" in logged_metrics
        assert f"iter_{count}_roc_auc" in logged_metrics


def test_evaluate_variant_task_logs_four_artifacts(
    tiny_model: Any,
    eval_splits: dict[str, pd.DataFrame],
    fake_client_cls: Any,
    mock_mlflow: MagicMock,
    quiet_logger: None,
) -> None:
    fake_client_cls.params = split_params(eval_splits)
    evaluate_variant_task.fn(
        tiny_model, eval_splits, source_run_id=DEFAULT_SOURCE_RUN_ID
    )
    artifact_calls = mock_mlflow.log_artifact.call_args_list
    assert len(artifact_calls) == 4
    destinations = sorted(
        (Path(call[0][0]).suffix, call[1]["artifact_path"]) for call in artifact_calls
    )
    assert destinations == [
        (".csv", "iterations"),
        (".csv", "thresholds"),
        (".png", "plots"),
        (".png", "plots"),
    ]


# --------------------------------------------------------------------------
# summarize_drift (pure helper — no mocks needed)
# --------------------------------------------------------------------------

DRIFT_FIXTURE: dict[str, Any] = {
    "metrics": [
        {
            "config": {
                "type": "evidently:metric_v2:DriftedColumnsCount",
                "drift_share": 0.5,
            },
            "value": {"count": 1.0, "share": 0.5},
        },
        {
            "config": {
                "type": "evidently:metric_v2:ValueDrift",
                "column": "sender_frequency",
                "method": "K-S p_value",
                "threshold": 0.1,
            },
            "value": 0.42,
        },
        {
            "config": {
                "type": "evidently:metric_v2:ValueDrift",
                "column": "hour_of_day",
                "method": "chi-square p_value",
                "threshold": 0.05,
            },
            "value": 0.001,
        },
    ]
}


def test_summarize_drift_extracts_counts_and_columns() -> None:
    summary = summarize_drift(DRIFT_FIXTURE)
    assert summary["n_drifted_columns"] == 1
    assert summary["drift_share"] == 0.5
    assert summary["dataset_drift"] is True
    assert summary["drifted_columns"] == ["hour_of_day"]
    assert summary["total_columns"] == 2


def test_summarize_drift_rejects_unknown_method() -> None:
    fixture = copy.deepcopy(DRIFT_FIXTURE)
    fixture["metrics"][1]["config"]["method"] = "mysterious_distance"
    with pytest.raises(ValueError, match="mysterious_distance"):
        summarize_drift(fixture)


def test_summarize_drift_rejects_missing_count_metric() -> None:
    with pytest.raises(ValueError, match="DriftedColumnsCount"):
        summarize_drift({"metrics": []})


def test_summarize_drift_rejects_count_flag_mismatch() -> None:
    fixture = copy.deepcopy(DRIFT_FIXTURE)
    fixture["metrics"][0]["value"]["count"] = 2.0
    with pytest.raises(ValueError, match="disagrees"):
        summarize_drift(fixture)


@pytest.mark.parametrize(
    ("method", "score", "threshold", "drifted"),
    [
        ("Wasserstein distance (normed)", 0.0066648467627892, 0.1, False),
        ("Wasserstein distance (normed)", 0.1, 0.1, True),
        ("Jensen-Shannon distance", 0.1, 0.1, True),
        ("Jensen-Shannon distance", 0.0019835684608966566, 0.1, False),
        ("chi-square p_value", 0.05, 0.05, False),
        ("chi-square p_value", 0.001, 0.05, True),
        ("K-S p_value", 0.05, 0.05, True),
    ],
)
def test_summarize_drift_verified_directions(
    method: str, score: float, threshold: float, drifted: bool
) -> None:
    fixture = copy.deepcopy(DRIFT_FIXTURE)
    fixture["metrics"] = fixture["metrics"][:2]
    fixture["metrics"][0]["value"] = {"count": float(drifted), "share": float(drifted)}
    metric = fixture["metrics"][1]
    metric["config"].update(method=method, threshold=threshold)
    metric["value"] = score
    summary = summarize_drift(fixture)
    assert summary["n_drifted_columns"] == int(drifted)
    assert summary["drift_share"] == float(drifted)
    assert summary["columns"] == [
        {
            "column": "sender_frequency",
            "method": method,
            "value": score,
            "threshold": threshold,
            "drifted": drifted,
        }
    ]


def test_summarize_drift_reports_all_unknown_methods() -> None:
    fixture = copy.deepcopy(DRIFT_FIXTURE)
    fixture["metrics"][1]["config"]["method"] = "unknown-distance"
    fixture["metrics"][2]["config"]["method"] = "unknown-p-value"
    with pytest.raises(ValueError) as error:
        summarize_drift(fixture)
    for detail in (
        "sender_frequency",
        "unknown-distance",
        "hour_of_day",
        "unknown-p-value",
    ):
        assert detail in str(error.value)


def test_summarize_drift_prefers_explicit_flag() -> None:
    fixture = copy.deepcopy(DRIFT_FIXTURE)
    fixture["metrics"][1]["config"]["method"] = "custom-with-explicit-flag"
    fixture["metrics"][1]["drift_detected"] = False
    fixture["metrics"][1]["value"] = 0.0
    summary = summarize_drift(fixture)
    assert summary["n_drifted_columns"] == 1
    assert summary["drifted_columns"] == ["hour_of_day"]


def test_summarize_drift_count_mismatch_includes_table() -> None:
    fixture = copy.deepcopy(DRIFT_FIXTURE)
    fixture["metrics"][0]["value"]["count"] = 2.0
    with pytest.raises(ValueError) as error:
        summarize_drift(fixture)
    message = str(error.value)
    for detail in (
        "Evidently count (2)",
        "per-column flags (1)",
        "sender_frequency",
        "hour_of_day",
        "method",
        "value",
        "threshold",
        "drifted",
    ):
        assert detail in message


def test_summarize_drift_real_evidently_both_method_families() -> None:
    """Real 80-row reports: distance/p-value methods, drift/no-drift cases."""
    from evidently import Report
    from evidently.presets import DataDriftPreset

    reference = pd.DataFrame(
        {
            "high_cardinality": np.arange(80),
            "category": np.tile([0, 1, 2, 3], 20),
        }
    )
    for shift_distance in (True, False):
        current = reference.copy()
        if shift_distance:
            current["high_cardinality"] += 100
        else:
            current["category"] = np.repeat([0, 1, 2, 3], [65, 5, 5, 5])
        snapshot = (
            Report(
                metrics=[
                    DataDriftPreset(
                        per_column_method={
                            "high_cardinality": "wasserstein",
                            "category": "chisquare",
                        }
                    )
                ]
            )
            .run(current_data=current, reference_data=reference)
            .dict()
        )
        summary = summarize_drift(snapshot)
        columns = {row["column"]: row for row in summary["columns"]}
        assert columns["high_cardinality"]["method"] == "wasserstein"
        assert columns["category"]["method"] == "chisquare"
        assert columns["high_cardinality"]["drifted"] is shift_distance
        assert columns["category"]["drifted"] is not shift_distance
        headline = next(
            metric
            for metric in snapshot["metrics"]
            if metric["config"]["type"].endswith("DriftedColumnsCount")
        )["value"]
        assert sum(row["drifted"] for row in summary["columns"]) == headline["count"]
        assert summary["n_drifted_columns"] == headline["count"] == 1
        assert summary["drift_share"] == headline["share"] == 0.5


# --------------------------------------------------------------------------
# evidently_baseline_task (Evidently mocked)
# --------------------------------------------------------------------------


@pytest.fixture
def fake_evidently(monkeypatch: pytest.MonkeyPatch):
    """Capture Report construction/run args; Snapshot writes a tiny HTML file."""

    class FakeSnapshot:
        def __init__(self, payload: dict[str, Any]) -> None:
            self.payload = payload

        def dict(self) -> dict[str, Any]:
            return copy.deepcopy(self.payload)

        def save_html(self, path: str) -> None:
            Path(path).write_bytes(b"<html></html>")

    class FakeReport:
        last: Any = None

        def __init__(self, metrics: list[Any] | None = None) -> None:
            self.metrics = metrics
            type(self).last = self

        def run(self, current: pd.DataFrame, reference: pd.DataFrame) -> FakeSnapshot:
            self.current = current
            self.reference = reference
            return FakeSnapshot(DRIFT_FIXTURE)

    preset = MagicMock()
    monkeypatch.setattr(evaluate_tasks, "Report", FakeReport)
    monkeypatch.setattr(evaluate_tasks, "DataDriftPreset", preset)
    return FakeReport, preset


def test_evidently_baseline_task_logs_html_and_summary(
    eval_splits: dict[str, pd.DataFrame],
    fake_evidently: Any,
    mock_mlflow: MagicMock,
    quiet_logger: None,
) -> None:
    fake_report_cls, preset = fake_evidently
    summary = evidently_baseline_task.fn(
        eval_splits["train"], eval_splits["test"], "eval-run-id"
    )

    preset.assert_called_once_with()
    report = fake_report_cls.last
    assert list(report.current.columns) == list(BEHAVIORAL_FEATURES)
    assert list(report.reference.columns) == list(BEHAVIORAL_FEATURES)
    assert len(report.reference) == len(eval_splits["train"])
    assert len(report.current) == len(eval_splits["test"])

    mock_mlflow.start_run.assert_called_once_with(run_id="eval-run-id")
    artifact_calls = mock_mlflow.log_artifact.call_args_list
    assert len(artifact_calls) == 2
    assert artifact_calls[0][1]["artifact_path"] == "evidently"
    assert Path(artifact_calls[1][0][0]).name == "evidently_columns.csv"
    drift_metrics = mock_mlflow.log_metrics.call_args[0][0]
    assert drift_metrics["evidently_n_drifted_columns"] == 1.0
    assert drift_metrics["evidently_drift_share"] == 0.5

    assert summary["n_drifted_columns"] == 1
    assert summary["drifted_columns"] == ["hour_of_day"]
    assert summary["reference_rows"] == len(eval_splits["train"])
    assert summary["current_rows"] == len(eval_splits["test"])


def test_evidently_baseline_task_caps_reference(
    eval_frame: pd.DataFrame,
    fake_evidently: Any,
    mock_mlflow: MagicMock,
    quiet_logger: None,
) -> None:
    fake_report_cls, _ = fake_evidently
    train_df = eval_frame.iloc[:60].reset_index(drop=True)
    test_df = eval_frame.iloc[60:80].reset_index(drop=True)
    evidently_baseline_task.fn(train_df, test_df, "eval-run-id", reference_cap=10)
    first_reference = fake_report_cls.last.reference
    evidently_baseline_task.fn(train_df, test_df, "eval-run-id", reference_cap=10)
    assert len(fake_report_cls.last.reference) == 10
    assert fake_report_cls.last.reference.equals(first_reference)
    expected = (
        train_df[list(BEHAVIORAL_FEATURES)]
        .sample(n=10, random_state=SPLIT_SEED)
        .reset_index(drop=True)
    )
    assert fake_report_cls.last.reference.equals(expected)
    params = mock_mlflow.log_params.call_args[0][0]
    assert params["evidently_reference_rows"] == 10


# --------------------------------------------------------------------------
# retrain_diagnostic_task (MLflow mocked; tiny real model as the artifact)
# --------------------------------------------------------------------------

DIAG_PARAMS: dict[str, str] = {
    "seed": "42",
    "learning_rate": "0.05",
    "num_leaves": "31",
    "scale_pos_weight": "145.7",
}


def artifact_val_pr_auc(tiny_model: Any, eval_splits: dict[str, pd.DataFrame]) -> float:
    """Val PR-AUC of the artifact surrogate, as the main task would report it."""
    val_df = eval_splits["val"]
    x_val = build_matrices(val_df, feature_columns=BEHAVIORAL_FEATURES, use_tfidf=False)
    proba = np.asarray(tiny_model.predict_proba(x_val)[:, 1], dtype=float)
    return pr_auc(val_df[TARGET].to_numpy(), proba)


def test_diagnostic_passes_gate_and_logs(
    tiny_model: Any,
    eval_splits: dict[str, pd.DataFrame],
    fake_client_cls: Any,
    mock_mlflow: MagicMock,
    quiet_logger: None,
) -> None:
    fake_client_cls.params = dict(DIAG_PARAMS)
    artifact = artifact_val_pr_auc(tiny_model, eval_splits)
    summary = retrain_diagnostic_task.fn(
        eval_splits["train"],
        eval_splits["val"],
        "eval-run-id",
        artifact,
        n_estimators=60,
    )

    assert summary["snapshots"] == [1, 10, 50]
    assert summary["metric_iterations"] == [1, 10, 50]
    assert summary["actual_trees"] == 60
    assert summary["sanity_gap"] < 1e-6
    assert summary["hyperparams"] == {
        "seed": 42,
        "learning_rate": 0.05,
        "num_leaves": 31,
        "scale_pos_weight": 145.7,
    }
    assert [row["iteration"] for row in summary["iterations"]] == [1, 10, 50]

    mock_mlflow.start_run.assert_called_once_with(run_id="eval-run-id")
    assert mock_mlflow.set_tags.call_args[0][0] == {"diagnostic_retrain": "true"}
    params = mock_mlflow.log_params.call_args[0][0]
    assert params["diagnostic_n_estimators"] == 60
    logged_metrics = mock_mlflow.log_metrics.call_args[0][0]
    for count in (1, 10, 50):
        assert f"diagnostic_retrain_{count}_pr_auc" in logged_metrics
        assert f"diagnostic_retrain_{count}_nolead_pr_auc" in logged_metrics
    assert (
        len([key for key in logged_metrics if key.startswith("diagnostic_retrain_")])
        == 6
    )
    nolead = eval_splits["val"].loc[
        eval_splits["val"]["sender_is_leadership"] == 0, TARGET
    ]
    assert logged_metrics["diagnostic_nolead_prevalence"] == float(nolead.mean())
    artifact_calls = mock_mlflow.log_artifact.call_args_list
    assert len(artifact_calls) == 1
    assert Path(artifact_calls[0][0][0]).suffix == ".csv"
    assert artifact_calls[0][1]["artifact_path"] == "diagnostics"


@pytest.mark.parametrize("rounds", [101, 1000])
def test_diagnostic_fixed_metrics_and_artifact_only_curve_points(
    eval_splits: dict[str, pd.DataFrame],
    fake_client_cls: Any,
    mock_mlflow: MagicMock,
    quiet_logger: None,
    monkeypatch: pytest.MonkeyPatch,
    rounds: int,
) -> None:
    """Controlled trajectories verify slice selection, CSVs, and fit parameters."""
    fake_client_cls.params = dict(DIAG_PARAMS)
    val = eval_splits["val"].iloc[:4].copy()
    val[TARGET] = [1, 0, 1, 0]
    val["sender_is_leadership"] = [1, 0, 0, 0]
    probes = [k for k in (1, 10, 50, 101, 300, 600, 1000) if k <= rounds]
    artifact_points = sorted(set(probes) | set(range(50, rounds + 1, 50)))
    scores = {
        k: np.array([0.9, 0.3, 0.8, 0.1]) if k == 50 else np.array([0.9, 0.8, 0.2, 0.1])
        for k in artifact_points
    }
    model = MagicMock()
    model.booster_.num_trees.return_value = rounds
    model.predict_proba.side_effect = lambda matrix, num_iteration: np.column_stack(
        (1 - scores[num_iteration], scores[num_iteration])
    )
    factory = MagicMock(return_value=model)
    monkeypatch.setattr(evaluate_tasks, "build_model", factory)
    captured = {}
    mock_mlflow.log_artifact.side_effect = lambda path, artifact_path: captured.update(
        {Path(path).name: pd.read_csv(path)}
    )
    result = retrain_diagnostic_task.fn(
        eval_splits["train"],
        val,
        "eval-run-id",
        pr_auc(val[TARGET], scores[1]),
        n_estimators=rounds,
    )
    assert [
        call.kwargs["num_iteration"] for call in model.predict_proba.call_args_list
    ] == artifact_points
    assert all(
        call.args[0].shape[0] == 4 for call in model.predict_proba.call_args_list
    )
    assert model.fit.call_args.kwargs == {}  # no eval set or early stopping callbacks
    assert len(model.fit.call_args.args) == 2
    factory.assert_called_once_with(
        scale_pos_weight=145.7,
        n_estimators=rounds,
        learning_rate=0.05,
        num_leaves=31,
        random_state=42,
    )
    for name, column, filename, labels, mask in (
        (
            "all",
            "pr_auc",
            "retrain_pr_auc_val.csv",
            val[TARGET].to_numpy(),
            np.ones(4, dtype=bool),
        ),
        (
            "nolead",
            "nolead_pr_auc",
            "retrain_pr_auc_val.csv",
            np.array([0, 1, 0]),
            np.array([False, True, True, True]),
        ),
    ):
        expected = [pr_auc(labels, scores[k][mask]) for k in artifact_points]
        assert captured[filename]["iteration"].tolist() == artifact_points
        assert captured[filename][column].tolist() == pytest.approx(expected)
        assert result["slices"][name] == {
            "status": "defined",
            "rows": len(labels),
            "positives": int(labels.sum()),
            "best_iteration": 50,
            "best_pr_auc": 1.0,
            "first_pr_auc": expected[0],
            "final_pr_auc": expected[-1],
        }
    metrics = mock_mlflow.log_metrics.call_args.args[0]
    probe_metrics = {
        key: value
        for key, value in metrics.items()
        if key.startswith("diagnostic_retrain_")
    }
    assert len(probe_metrics) == 2 * len(probes)
    assert len(probe_metrics) <= 14
    for k in probes:
        assert probe_metrics[f"diagnostic_retrain_{k}_pr_auc"] == pr_auc(
            val[TARGET], scores[k]
        )
        assert probe_metrics[f"diagnostic_retrain_{k}_nolead_pr_auc"] == pr_auc(
            np.array([0, 1, 0]), scores[k][1:]
        )
    for k in set(artifact_points) - set(probes):
        assert f"diagnostic_retrain_{k}_pr_auc" not in metrics
        assert f"diagnostic_retrain_{k}_nolead_pr_auc" not in metrics
    assert metrics["diagnostic_nolead_prevalence"] == pytest.approx(1 / 3)
    assert captured["retrain_pr_auc_val.csv"][
        "nolead_prevalence"
    ].tolist() == pytest.approx([1 / 3] * len(artifact_points))


@pytest.mark.parametrize("empty", [True, False])
def test_diagnostic_undefined_nolead_scores_are_not_logged(
    tiny_model: Any,
    eval_splits: dict[str, pd.DataFrame],
    fake_client_cls: Any,
    mock_mlflow: MagicMock,
    quiet_logger: None,
    empty: bool,
) -> None:
    fake_client_cls.params = dict(DIAG_PARAMS)
    val = eval_splits["val"].copy()
    val["sender_is_leadership"] = 1 if empty else val[TARGET]
    changed_splits = {**eval_splits, "val": val}
    artifact = artifact_val_pr_auc(tiny_model, changed_splits)
    result = retrain_diagnostic_task.fn(
        eval_splits["train"],
        val,
        "eval-run-id",
        artifact,
        n_estimators=3,
    )
    summary = result["slices"]["nolead"]
    assert summary["status"] == ("empty_slice" if empty else "no_positive_labels")
    assert summary["positives"] == 0
    for key in ("best_iteration", "best_pr_auc", "first_pr_auc", "final_pr_auc"):
        assert summary[key] is None
    metrics = mock_mlflow.log_metrics.call_args.args[0]
    assert all(np.isfinite(value) for value in metrics.values())
    assert not any(key.endswith("_nolead_pr_auc") for key in metrics)
    assert "diagnostic_nolead_best_pr_auc" not in metrics
    assert all(row["nolead_pr_auc"] is None for row in result["iterations"])
    if empty:
        assert "diagnostic_nolead_prevalence" not in metrics
    else:
        assert metrics["diagnostic_nolead_prevalence"] == 0.0


def test_diagnostic_default_is_1000_rounds() -> None:
    assert retrain_diagnostic_task.fn.__defaults__[-1] == 1000


def test_diagnostic_gate_failure_aborts_before_logging(
    tiny_model: Any,
    eval_splits: dict[str, pd.DataFrame],
    fake_client_cls: Any,
    mock_mlflow: MagicMock,
    quiet_logger: None,
) -> None:
    fake_client_cls.params = dict(DIAG_PARAMS)
    artifact = artifact_val_pr_auc(tiny_model, eval_splits)
    with pytest.raises(DiagnosticMismatchError, match="diverged"):
        retrain_diagnostic_task.fn(
            eval_splits["train"],
            eval_splits["val"],
            "eval-run-id",
            artifact + 0.05,
            n_estimators=20,
        )
    mock_mlflow.log_artifact.assert_not_called()
    mock_mlflow.log_metrics.assert_not_called()
    mock_mlflow.log_params.assert_not_called()
    mock_mlflow.set_tags.assert_not_called()
    mock_mlflow.start_run.assert_not_called()


def test_diagnostic_rejects_non_finite_artifact_pr_auc(
    eval_splits: dict[str, pd.DataFrame],
    fake_client_cls: Any,
    mock_mlflow: MagicMock,
    quiet_logger: None,
) -> None:
    with pytest.raises(DiagnosticMismatchError, match="not finite"):
        retrain_diagnostic_task.fn(
            eval_splits["train"],
            eval_splits["val"],
            "eval-run-id",
            float("nan"),
            n_estimators=20,
        )


def test_diagnostic_rejects_missing_training_param(
    eval_splits: dict[str, pd.DataFrame],
    fake_client_cls: Any,
    mock_mlflow: MagicMock,
    quiet_logger: None,
) -> None:
    fake_client_cls.params = {
        key: value for key, value in DIAG_PARAMS.items() if key != "num_leaves"
    }
    with pytest.raises(ValueError, match="num_leaves"):
        retrain_diagnostic_task.fn(
            eval_splits["train"],
            eval_splits["val"],
            "eval-run-id",
            0.5,
            n_estimators=20,
        )


def test_diagnostic_rejects_non_numeric_training_param(
    eval_splits: dict[str, pd.DataFrame],
    fake_client_cls: Any,
    mock_mlflow: MagicMock,
    quiet_logger: None,
) -> None:
    fake_client_cls.params = dict(DIAG_PARAMS)
    fake_client_cls.params["learning_rate"] = "fast"
    with pytest.raises(ValueError, match="non-numeric"):
        retrain_diagnostic_task.fn(
            eval_splits["train"],
            eval_splits["val"],
            "eval-run-id",
            0.5,
            n_estimators=20,
        )


# --------------------------------------------------------------------------
# leadership slice (ADR-010)
# --------------------------------------------------------------------------


def test_evaluate_variant_task_leadership_slice(
    tiny_model: Any,
    eval_splits: dict[str, pd.DataFrame],
    fake_client_cls: Any,
    mock_mlflow: MagicMock,
    quiet_logger: None,
) -> None:
    fake_client_cls.params = split_params(eval_splits)
    summary = evaluate_variant_task.fn(
        tiny_model, eval_splits, source_run_id=DEFAULT_SOURCE_RUN_ID
    )

    test_df = eval_splits["test"]
    x_test = build_matrices(
        test_df, feature_columns=BEHAVIORAL_FEATURES, use_tfidf=False
    )
    test_proba = np.asarray(tiny_model.predict_proba(x_test)[:, 1], dtype=float)
    y_test = test_df[TARGET].to_numpy()
    threshold = summary["tuned_threshold"]
    lead_mask = test_df["sender_is_leadership"].to_numpy() == 1

    assert summary["test_leadership_rows"] == int(lead_mask.sum())
    assert summary["test_leadership_flagged"] == int(
        ((test_proba >= threshold) & lead_mask).sum()
    )

    expected = metrics_at_threshold(
        y_test[~lead_mask], test_proba[~lead_mask], threshold=threshold
    )
    assert summary["leadership_slice"]["precision"] == expected["precision"]
    assert summary["leadership_slice"]["recall"] == expected["recall"]
    assert summary["leadership_slice"]["f1"] == expected["f1"]
    assert summary["leadership_slice"]["pr_auc"] == pr_auc(
        y_test[~lead_mask], test_proba[~lead_mask]
    )

    logged = mock_mlflow.log_metrics.call_args[0][0]
    for key in (
        "test_nolead_precision",
        "test_nolead_recall",
        "test_nolead_f1",
        "test_nolead_pr_auc",
        "test_leadership_rows",
        "test_leadership_flagged",
    ):
        assert key in logged
    assert logged["test_leadership_rows"] == float(int(lead_mask.sum()))
