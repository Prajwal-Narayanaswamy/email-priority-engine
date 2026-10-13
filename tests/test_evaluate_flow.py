"""Tests for src.flows.evaluate_flow — tasks mocked, order and wiring only."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest
from click.testing import CliRunner
from prefect.testing.utilities import prefect_test_harness

import src.flows.evaluate_flow as flow_module
from src.flows.evaluate_flow import evaluate_flow, main
from src.tasks import evaluate_tasks

TASK_NAMES = (
    "load_split_task",
    "load_model_task",
    "evaluate_variant_task",
    "retrain_diagnostic_task",
    "evidently_baseline_task",
)

EVALUATION_SUMMARY: dict[str, Any] = {
    "run_id": "eval-run-id",
    "tuned_threshold": 0.9,
    "test": {"f1": 0.12, "pr_auc": 0.4},
    "val": {"pr_auc": 0.42},
    "loaded_num_trees": 1886,
    "source_best_iteration": 1886,
}


@pytest.fixture
def mock_tasks(monkeypatch: pytest.MonkeyPatch) -> dict[str, MagicMock]:
    """Replace every task in the flow module; record call order."""
    calls: list[str] = []
    fakes: dict[str, MagicMock] = {}
    returns: dict[str, Any] = {
        "load_split_task": {
            "train": MagicMock(name="train"),
            "val": MagicMock(name="val"),
            "test": MagicMock(name="test"),
        },
        "load_model_task": MagicMock(name="model"),
        "evaluate_variant_task": dict(EVALUATION_SUMMARY),
        "retrain_diagnostic_task": {"sanity_gap": 0.0},
        "evidently_baseline_task": {"drift_share": 0.1},
    }
    for name in TASK_NAMES:
        fake = MagicMock(name=name)

        def _call(*args: Any, _name: str = name, **kwargs: Any) -> Any:
            calls.append(_name)
            return returns[_name]

        fake.side_effect = _call
        monkeypatch.setattr(flow_module, name, fake)
        fakes[name] = fake
    fakes["calls"] = calls  # type: ignore[assignment]
    fakes["returns"] = returns  # type: ignore[assignment]
    return fakes


def test_evaluate_flow_calls_tasks_in_order(
    mock_tasks: dict[str, MagicMock],
) -> None:
    with prefect_test_harness():
        evaluate_flow(input_path="data/features/features.parquet", source_run_id="abc")
    assert mock_tasks["calls"] == list(TASK_NAMES)


def test_evaluate_flow_threads_run_id_threshold_and_splits(
    mock_tasks: dict[str, MagicMock],
) -> None:
    with prefect_test_harness():
        evaluate_flow(input_path="data/features/features.parquet", source_run_id="abc")

    splits = mock_tasks["returns"]["load_split_task"]
    model = mock_tasks["returns"]["load_model_task"]
    mock_tasks["load_model_task"].assert_called_once_with(source_run_id="abc")
    mock_tasks["evaluate_variant_task"].assert_called_once_with(
        model,
        splits,
        source_run_id="abc",
        reference_cap=flow_module.REFERENCE_CAP,
        seed=flow_module.SPLIT_SEED,
    )
    mock_tasks["retrain_diagnostic_task"].assert_called_once_with(
        splits["train"],
        splits["val"],
        "eval-run-id",
        EVALUATION_SUMMARY["val"]["pr_auc"],
        source_run_id="abc",
        artifact_num_trees=1886,
        artifact_best_iteration=1886,
    )
    mock_tasks["evidently_baseline_task"].assert_called_once_with(
        splits["train"],
        splits["test"],
        "eval-run-id",
        reference_cap=flow_module.REFERENCE_CAP,
        seed=flow_module.SPLIT_SEED,
    )


def test_evaluate_flow_returns_all_summaries(
    mock_tasks: dict[str, MagicMock],
) -> None:
    with prefect_test_harness():
        result = evaluate_flow(
            input_path="data/features/features.parquet", source_run_id="abc"
        )
    assert set(result) == {"evaluation", "diagnostic", "drift"}
    assert result["evaluation"]["run_id"] == "eval-run-id"


def test_cli_requires_run_id() -> None:
    runner = CliRunner()
    result = runner.invoke(main, ["--input", "data/features/features.parquet"])
    assert result.exit_code == 2
    assert "--run-id" in result.output


def test_cli_has_no_limit_option() -> None:
    assert "limit" not in {param.name for param in main.params}
    assert "n_estimators" not in {param.name for param in main.params}


def test_cli_rejects_missing_input(
    mock_tasks: dict[str, MagicMock], tmp_path: Path
) -> None:
    runner = CliRunner()
    with prefect_test_harness():
        result = runner.invoke(
            main, ["--run-id", "abc", "--input", str(tmp_path / "f.parquet")]
        )
    assert result.exit_code != 0  # input must exist


def test_cli_passes_options_to_flow(
    mock_tasks: dict[str, MagicMock], features_parquet: Path
) -> None:
    runner = CliRunner()
    with prefect_test_harness():
        result = runner.invoke(
            main,
            [
                "--run-id",
                "abc",
                "--input",
                str(features_parquet),
                "--reference-cap",
                "1000",
            ],
        )
    assert result.exit_code == 0, result.output
    mock_tasks["load_split_task"].assert_called_once()
    assert mock_tasks["evidently_baseline_task"].call_args[1]["reference_cap"] == 1000
    assert (
        mock_tasks["retrain_diagnostic_task"].call_args[1]["artifact_num_trees"] == 1886
    )


def test_evaluate_flow_computes_grouped_split_exactly_once(
    mock_tasks: dict[str, MagicMock],
    features_parquet: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Execute real split loading once; all consumers reuse those exact frames."""

    def logger() -> logging.Logger:
        return logging.getLogger("split_flow_test")

    monkeypatch.setattr(flow_module, "get_run_logger", logger)
    monkeypatch.setattr(evaluate_tasks, "get_run_logger", logger)
    spy = MagicMock(wraps=evaluate_tasks.grouped_split)
    monkeypatch.setattr(evaluate_tasks, "grouped_split", spy)
    monkeypatch.setattr(
        flow_module, "load_split_task", evaluate_tasks.load_split_task.fn
    )
    evaluate_flow.fn(input_path=str(features_parquet), source_run_id="abc")
    spy.assert_called_once()
    splits = mock_tasks["evaluate_variant_task"].call_args.args[1]
    assert mock_tasks["retrain_diagnostic_task"].call_args.args[0] is splits["train"]
    assert mock_tasks["retrain_diagnostic_task"].call_args.args[1] is splits["val"]
    assert mock_tasks["evidently_baseline_task"].call_args.args[0] is splits["train"]
    assert mock_tasks["evidently_baseline_task"].call_args.args[1] is splits["test"]
