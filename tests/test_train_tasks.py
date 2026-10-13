"""Tests for src.tasks.train_tasks — MLflow fully mocked, no live server."""

from __future__ import annotations

import importlib
import logging
import sys
from pathlib import Path
from unittest.mock import MagicMock

import dotenv
import pytest

from src.tasks import train_tasks
from src.tasks.train_tasks import (
    EXPERIMENT_NAME,
    MIN_POSITIVE_IN_SAMPLE,
    TRACKING_URI,
    compare_variants,
    load_features_task,
    train_variant_task,
)
from src.training.splits import (
    SPLIT_STRATEGY,
    SPLIT_VERSION,
    grouped_split,
    split_fingerprint,
)
from tests.conftest import FEATURE_POSITIVES, FEATURE_ROWS

N_ROWS = FEATURE_ROWS


@pytest.fixture
def mock_mlflow(monkeypatch: pytest.MonkeyPatch) -> MagicMock:
    """Replace the whole `mlflow` module reference in the task module."""
    fake = MagicMock()
    run = MagicMock()
    run.info.run_id = "test-run-id"
    fake.active_run.return_value = run
    fake.start_run.return_value.__enter__.return_value = run
    monkeypatch.setattr(train_tasks, "mlflow", fake)
    return fake


@pytest.fixture
def quiet_logger(monkeypatch: pytest.MonkeyPatch) -> None:
    """`get_run_logger` needs a Prefect run context; swap in a stdlib logger."""
    monkeypatch.setattr(
        train_tasks, "get_run_logger", lambda: logging.getLogger("train_test")
    )


# --------------------------------------------------------------------------
# load_features_task
# --------------------------------------------------------------------------


def test_load_features_task_reads_only_needed_columns(
    features_parquet: Path, quiet_logger: None
) -> None:
    df = load_features_task.fn(features_parquet)
    assert set(df.columns) == {
        "uid",
        "from",
        "subject",
        "priority",
        "sender_frequency",
        "sender_is_leadership",
        "hour_of_day",
        "day_of_week",
        "urgency_keyword_count",
        "subject_length",
        "has_question_mark",
        "thread_depth",
    }
    assert "body" not in df.columns
    assert len(df) == N_ROWS


def test_load_features_task_limit_returns_stratified_sample(
    features_parquet: Path, quiet_logger: None
) -> None:
    """A small --limit must still yield enough of the rare class to split on."""
    df = load_features_task.fn(features_parquet, limit=100)
    positives = int((df["priority"] == 1).sum())
    assert positives >= MIN_POSITIVE_IN_SAMPLE
    assert 0 < positives < len(df)
    assert len(df) <= 100 + FEATURE_POSITIVES


def test_load_features_task_limit_is_deterministic(
    features_parquet: Path, quiet_logger: None
) -> None:
    first = load_features_task.fn(features_parquet, limit=100)
    second = load_features_task.fn(features_parquet, limit=100)
    assert first["subject"].tolist() == second["subject"].tolist()


def test_load_features_task_limit_above_size_returns_all(
    features_parquet: Path, quiet_logger: None
) -> None:
    df = load_features_task.fn(features_parquet, limit=N_ROWS * 2)
    assert len(df) == N_ROWS


# --------------------------------------------------------------------------
# train_variant_task
# --------------------------------------------------------------------------


def test_train_variant_task_uses_local_tracking_uri(
    features_parquet: Path, quiet_logger: None, mock_mlflow: MagicMock
) -> None:
    df = load_features_task.fn(features_parquet, limit=200)
    train_variant_task.fn(df, "behavioral-only", n_estimators=20)

    tracking_uri = mock_mlflow.set_tracking_uri.call_args[0][0]
    assert tracking_uri == TRACKING_URI
    assert tracking_uri.startswith("sqlite:///")
    mock_mlflow.set_experiment.assert_called_once_with(EXPERIMENT_NAME)


def test_tracking_uri_reads_environment_override(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """MLFLOW_TRACKING_URI must be honoured over the SQLite default."""
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "file:///tmp/custom-mlruns")
    importlib.reload(train_tasks)
    assert train_tasks.TRACKING_URI == "file:///tmp/custom-mlruns"

    monkeypatch.delenv("MLFLOW_TRACKING_URI")
    importlib.reload(train_tasks)
    assert train_tasks.TRACKING_URI == train_tasks._DEFAULT_URI


def test_tracking_uri_default_is_absolute_sqlite_anchored_to_repo() -> None:
    """Fix 1: the default must not depend on the process CWD."""
    assert train_tasks.TRACKING_URI.startswith("sqlite:////")
    assert train_tasks._REPO_ROOT == Path(__file__).resolve().parents[1]
    assert train_tasks._DEFAULT_URI == (
        f"sqlite:///{train_tasks._REPO_ROOT}/mlruns/mlflow.db"
    )
    assert Path(train_tasks._REPO_ROOT, "pyproject.toml").is_file()


def test_tracking_uri_is_cwd_independent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A relative sqlite URI would create a second invisible DB off-repo."""
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
    importlib.reload(train_tasks)
    from_repo = train_tasks.TRACKING_URI

    monkeypatch.chdir(tmp_path)
    importlib.reload(train_tasks)
    assert train_tasks.TRACKING_URI == from_repo


def test_train_flow_loads_dotenv_before_importing_train_tasks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Fix 2: load_dotenv() must run before train_tasks reads os.environ.

    train_tasks evaluates TRACKING_URI at import time, so a load_dotenv() placed
    after the imports would silently never affect it. Simulates a cold start by
    dropping both modules from sys.modules, since a cached train_tasks would
    never re-read the environment.
    """
    calls: list[bool] = []
    monkeypatch.setattr(
        dotenv, "load_dotenv", lambda *a, **k: calls.append(True) or True
    )
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "sqlite:///from-dotenv.db")

    saved = {
        name: sys.modules.pop(name)
        for name in ("src.flows.train_flow", "src.tasks.train_tasks")
        if name in sys.modules
    }
    try:
        fresh = importlib.import_module("src.flows.train_flow")
        assert calls, "load_dotenv() was not called when train_flow was imported"
        assert fresh.TRACKING_URI == "sqlite:///from-dotenv.db"
        # the imported constant must be the dotenv-derived one, not the default
        assert fresh.train_variant_task.__module__ == "src.tasks.train_tasks"
    finally:
        sys.modules.update(saved)
        monkeypatch.delenv("MLFLOW_TRACKING_URI")


def test_train_variant_task_logs_derived_scale_pos_weight(
    features_parquet: Path, quiet_logger: None, mock_mlflow: MagicMock
) -> None:
    df = load_features_task.fn(features_parquet, limit=200)
    train_variant_task.fn(df, "behavioral-only", n_estimators=20)

    params = mock_mlflow.log_params.call_args[0][0]
    train, _, _ = grouped_split(df)
    positives = int((train["priority"] == 1).sum())
    negatives = int((train["priority"] == 0).sum())
    assert params["scale_pos_weight"] == negatives / positives
    assert params["primary_metric"] == "class_1_f1"


@pytest.mark.parametrize("variant", ["behavioral-only", "with-keyword-features"])
def test_training_logs_shared_grouped_split_and_passes_derived_weight(
    features_parquet: Path,
    quiet_logger: None,
    mock_mlflow: MagicMock,
    monkeypatch: pytest.MonkeyPatch,
    variant: str,
) -> None:
    df = load_features_task.fn(features_parquet)
    expected_splits = grouped_split(df, seed=42)
    build = MagicMock(wraps=train_tasks.build_model)
    monkeypatch.setattr(train_tasks, "build_model", build)
    result = train_variant_task.fn(df, variant, n_estimators=20)
    params = mock_mlflow.log_params.call_args.args[0]
    assert params["split_strategy"] == SPLIT_STRATEGY
    assert params["split_version"] == SPLIT_VERSION
    assert params["seed"] == params["split_seed"] == 42
    assert params["dedupe_policy"] == "exact (from, subject, date), keep first"
    assert params["split_fingerprint"] == split_fingerprint(*expected_splits)
    assert result["split_fingerprint"] == params["split_fingerprint"]
    for name, part in zip(("train", "val", "test"), expected_splits):
        assert params[f"{name}_rows"] == len(part)
        assert params[f"{name}_positives"] == int(part["priority"].sum())
    labels = expected_splits[0]["priority"]
    expected_weight = float((labels == 0).sum() / (labels == 1).sum())
    assert params["scale_pos_weight"] == expected_weight
    assert build.call_args.kwargs["scale_pos_weight"] == expected_weight


def test_training_smoke_sample_keeps_unique_uids(
    features_parquet: Path,
    quiet_logger: None,
) -> None:
    sample = load_features_task.fn(features_parquet, limit=100)
    assert sample["uid"].is_unique


def test_train_variant_task_logs_class_one_metrics(
    features_parquet: Path, quiet_logger: None, mock_mlflow: MagicMock
) -> None:
    df = load_features_task.fn(features_parquet, limit=200)
    train_variant_task.fn(df, "with-keyword-features", n_estimators=20)

    metrics = mock_mlflow.log_metrics.call_args[0][0]
    for key in ("test_f1", "test_precision", "test_recall", "val_f1"):
        assert key in metrics
    assert 0.0 <= metrics["test_f1"] <= 1.0
    assert {"test_true_positive", "test_false_positive"} <= set(metrics)


def test_train_variant_task_logs_model_and_vectorizer_artifacts(
    features_parquet: Path, quiet_logger: None, mock_mlflow: MagicMock
) -> None:
    df = load_features_task.fn(features_parquet, limit=200)
    train_variant_task.fn(df, "with-keyword-features", n_estimators=20)

    names = [c.kwargs["name"] for c in mock_mlflow.sklearn.log_model.call_args_list]
    assert names == ["model", "subject_vectorizer"]
    mock_mlflow.log_artifact.assert_called_once()


def test_behavioral_variant_logs_no_vectorizer(
    features_parquet: Path, quiet_logger: None, mock_mlflow: MagicMock
) -> None:
    df = load_features_task.fn(features_parquet, limit=200)
    train_variant_task.fn(df, "behavioral-only", n_estimators=20)

    names = [c.kwargs["name"] for c in mock_mlflow.sklearn.log_model.call_args_list]
    assert names == ["model"]


def test_train_variant_task_tags_limitation(
    features_parquet: Path, quiet_logger: None, mock_mlflow: MagicMock
) -> None:
    df = load_features_task.fn(features_parquet, limit=200)
    train_variant_task.fn(df, "behavioral-only", n_estimators=20)

    tags = mock_mlflow.set_tags.call_args[0][0]
    assert "ADR-005" in tags["known_limitation"]
    assert "sender_frequency" in tags["known_limitation"]


def test_train_variant_task_runs_inside_mlflow_context(
    features_parquet: Path, quiet_logger: None, mock_mlflow: MagicMock
) -> None:
    """AGENTS.md: never log a model outside a run context."""
    df = load_features_task.fn(features_parquet, limit=200)
    train_variant_task.fn(df, "behavioral-only", n_estimators=20)
    mock_mlflow.start_run.assert_called_once()
    assert mock_mlflow.start_run.call_args.kwargs["run_name"] == "behavioral-only"


def test_train_variant_task_returns_builtin_types(
    features_parquet: Path, quiet_logger: None, mock_mlflow: MagicMock
) -> None:
    df = load_features_task.fn(features_parquet, limit=200)
    result = train_variant_task.fn(df, "behavioral-only", n_estimators=20)

    assert result["run_id"] == "test-run-id"
    assert result["variant"] == "behavioral-only"
    for key in ("test_f1", "test_precision", "test_recall"):
        assert type(result[key]) is float
    for key in ("best_iteration", "n_matrix_columns", "train_rows"):
        assert type(result[key]) is int


def test_train_variant_task_records_split_sizes(
    features_parquet: Path, quiet_logger: None, mock_mlflow: MagicMock
) -> None:
    df = load_features_task.fn(features_parquet, limit=200)
    result = train_variant_task.fn(df, "behavioral-only", n_estimators=20)

    assert result["train_rows"] + result["val_rows"] + result["test_rows"] == len(df)


def test_train_variant_task_rejects_unknown_variant(
    features_parquet: Path, quiet_logger: None, mock_mlflow: MagicMock
) -> None:
    df = load_features_task.fn(features_parquet, limit=200)
    with pytest.raises(KeyError):
        train_variant_task.fn(df, "nope", n_estimators=20)


# --------------------------------------------------------------------------
# compare_variants
# --------------------------------------------------------------------------


def test_compare_variants_ranks_by_f1() -> None:
    results = [
        {
            "variant": "with-keyword-features",
            "test_f1": 0.91,
            "test_precision": 0.88,
            "test_recall": 0.94,
            "test_roc_auc": 0.99,
        },
        {
            "variant": "behavioral-only",
            "test_f1": 0.42,
            "test_precision": 0.40,
            "test_recall": 0.45,
            "test_roc_auc": 0.80,
        },
    ]
    output = compare_variants(results)
    assert "with-keyword-features" in output
    assert "behavioral-only" in output
    assert "0.9100" in output
    assert output.index("with-keyword-features") < output.index("behavioral-only")
