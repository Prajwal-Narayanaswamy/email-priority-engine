"""Registration and lookup-logging against a real MLflow on a temporary sqlite DB.

Never touches the repo's mlruns: the tracking URI, experiments and artifact roots
are all under pytest's tmp_path, and the global tracking URI is restored.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from dataclasses import dataclass, replace
from pathlib import Path

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import pytest
from click.testing import CliRunner
from fastapi.testclient import TestClient
from mlflow import MlflowClient
from sklearn.linear_model import LogisticRegression

from scripts.export_model_bundle import export_bundle
from scripts.log_lookup_run import main as log_lookup_main
from scripts.register_reference_model import main as register_main
from src.serving import bundle, config
from src.serving import registration as registration_module
from src.serving.app import app, build_predictor
from src.serving.lookup import (
    build_header,
    build_sender_lookup,
    file_sha256,
    save_sender_lookup,
)
from src.serving.metadata import REQUIRED_TAGS, ServingStartupError
from src.serving.registration import (
    PrecheckError,
    apply_plan,
    build_plan,
    log_lookup_run,
)
from src.serving.registry import (
    CHAMPION_ALIAS,
    fetch_serving_metadata,
    load_model,
)


@dataclass
class Env:
    """A throwaway registry with a training run, evaluation run and lookup."""

    uri: str
    client: MlflowClient
    tmp: Path
    source_run_id: str
    eval_run_id: str
    lookup_path: Path
    features_path: Path
    logged_model_id: str

    def plan_kwargs(self) -> dict:
        return {
            "source_run_id": self.source_run_id,
            "eval_run_id": self.eval_run_id,
            "lookup_path": self.lookup_path,
            "features_path": self.features_path,
        }

    def log_lookup(self) -> str:
        return log_lookup_run(
            self.client,
            self.lookup_path,
            self.source_run_id,
            artifact_root=self.tmp / "serving",
        )


@pytest.fixture
def make_env(tmp_path: Path, mlflow_globals: None) -> Iterator[Callable[..., Env]]:
    def build(eval_params: dict[str, str] | None = None) -> Env:
        uri = f"sqlite:///{tmp_path}/mlflow.db"
        mlflow.set_tracking_uri(uri)
        client = MlflowClient(tracking_uri=uri)
        exp_id = client.create_experiment(
            config.EXPERIMENT_NAME, artifact_location=(tmp_path / "art").as_uri()
        )
        mlflow.set_experiment(experiment_id=exp_id)
        x = np.array([[0.0], [1.0], [2.0], [3.0]])
        with mlflow.start_run() as source:
            mlflow.log_param("split_fingerprint", "fp-123")
            info = mlflow.sklearn.log_model(
                LogisticRegression().fit(x, [0, 0, 1, 1]), name="model"
            )
        with mlflow.start_run() as evaluation:
            for key, value in (eval_params or {"tuned_threshold": "0.75"}).items():
                mlflow.log_param(key, value)
        features = pd.DataFrame(
            {"from": ["a@b.com", "A@B.com", "c@d.com"], "sender_frequency": [2, 2, 1]}
        )
        features_path = tmp_path / "features.parquet"
        features.to_parquet(features_path)
        lookup_path = tmp_path / "lookup.json"
        save_sender_lookup(
            build_sender_lookup(features),
            lookup_path,
            build_header(file_sha256(features_path), len(features)),
        )
        return Env(
            uri=uri,
            client=client,
            tmp=tmp_path,
            source_run_id=source.info.run_id,
            eval_run_id=evaluation.info.run_id,
            lookup_path=lookup_path,
            features_path=features_path,
            logged_model_id=info.model_id,
        )

    yield build


def test_log_lookup_run_is_separate_and_idempotent(
    make_env: Callable[..., Env],
) -> None:
    env = make_env()
    run_id = env.log_lookup()
    run = env.client.get_run(run_id)
    experiment = env.client.get_experiment(run.info.experiment_id)
    assert experiment.name == config.SERVING_EXPERIMENT_NAME != config.EXPERIMENT_NAME
    assert run.data.tags["source_run_id"] == env.source_run_id
    assert run.data.params["lookup_sha256"] == file_sha256(env.lookup_path)
    assert run.data.params["row_count"] == "3"
    assert len(run.data.params["parquet_sha256"]) == 64
    assert env.log_lookup() == run_id
    # The finished training run gained nothing.
    assert "lookup_sha256" not in env.client.get_run(env.source_run_id).data.params


def test_register_then_serve_metadata_and_load(make_env: Callable[..., Env]) -> None:
    env = make_env()
    lookup_run = env.log_lookup()
    before = dict(env.client.get_logged_model(env.logged_model_id).tags)
    plan = build_plan(env.client, **env.plan_kwargs())

    # Dry run (build_plan only) wrote nothing.
    assert env.client.search_registered_models() == []
    assert plan.tags["tuned_threshold"] == repr(0.75)
    assert set(plan.tags) == set(REQUIRED_TAGS)
    assert plan.tags["lookup_run_id"] == lookup_run
    assert plan.tags["lookup_sha256"] == file_sha256(env.lookup_path)
    assert plan.tags["split_fingerprint"] == "fp-123"
    assert any("create registered model" in line for line in plan.describe())

    assert apply_plan(env.client, plan) == "1"
    metadata = fetch_serving_metadata("models:/email-priority@champion", env.client)
    assert metadata.threshold == 0.75
    assert metadata.alias == CHAMPION_ALIAS and metadata.version == "1"
    assert metadata.tags == plan.tags
    model = load_model(metadata, env.uri)
    assert model.predict(np.array([[3.0]])).tolist() == [1]
    # The client route leaves the finished LoggedModel untouched.
    assert dict(env.client.get_logged_model(env.logged_model_id).tags) == before

    with pytest.raises(PrecheckError, match="already has version 1"):
        build_plan(env.client, **env.plan_kwargs())


def test_precheck_rejects_superseded_source_run(make_env: Callable[..., Env]) -> None:
    env = make_env()
    env.log_lookup()
    env.client.set_tag(env.source_run_id, "superseded", "duplicate-leaky-split")
    with pytest.raises(PrecheckError, match="superseded=duplicate-leaky-split"):
        build_plan(env.client, **env.plan_kwargs())


def test_precheck_rejects_changed_parquet(make_env: Callable[..., Env]) -> None:
    env = make_env()
    env.log_lookup()
    pd.DataFrame({"from": ["z@z.com"], "sender_frequency": [1]}).to_parquet(
        env.features_path
    )
    with pytest.raises(PrecheckError, match="parquet_sha256 .* != current"):
        build_plan(env.client, **env.plan_kwargs())


def test_precheck_requires_logged_lookup_run(make_env: Callable[..., Env]) -> None:
    env = make_env()
    with pytest.raises(PrecheckError, match="log_lookup_run"):
        build_plan(env.client, **env.plan_kwargs())


def test_precheck_requires_tuned_threshold(make_env: Callable[..., Env]) -> None:
    env = make_env(eval_params={"other": "1"})
    env.log_lookup()
    with pytest.raises(PrecheckError, match="no tuned_threshold"):
        build_plan(env.client, **env.plan_kwargs())


def test_precheck_lists_every_problem(make_env: Callable[..., Env]) -> None:
    env = make_env(eval_params={"tuned_threshold": "1.5"})
    env.client.set_tag(env.source_run_id, "superseded", "x")
    with pytest.raises(PrecheckError) as excinfo:
        build_plan(env.client, **env.plan_kwargs())
    message = str(excinfo.value)
    assert "superseded" in message
    assert "outside the open interval" in message
    assert "log_lookup_run" in message


def test_alias_already_set_blocks_registration(make_env: Callable[..., Env]) -> None:
    env = make_env()
    env.log_lookup()
    # An untagged foreign version already holds the alias; promotion is separate.
    env.client.create_registered_model("email-priority")
    version = env.client.create_model_version(
        "email-priority",
        source=f"models:/{env.logged_model_id}",
        model_id=env.logged_model_id,
    )
    env.client.set_registered_model_alias(
        "email-priority", CHAMPION_ALIAS, version.version
    )
    with pytest.raises(PrecheckError, match="alias 'champion' already points to"):
        build_plan(env.client, **env.plan_kwargs())


# ---- partial state (an earlier apply failed after creating the registered model)


def test_empty_registered_model_is_reused_and_reported(
    make_env: Callable[..., Env],
) -> None:
    env = make_env()
    env.log_lookup()
    env.client.create_registered_model("email-priority")
    plan = build_plan(env.client, **env.plan_kwargs())
    assert plan.create_registered_model is False
    assert any("exists with no versions" in line for line in plan.describe())
    assert apply_plan(env.client, plan) == "1"
    metadata = fetch_serving_metadata("models:/email-priority@champion", env.client)
    assert metadata.version == "1"


def test_apply_survives_registered_model_created_after_planning(
    make_env: Callable[..., Env],
) -> None:
    env = make_env()
    env.log_lookup()
    plan = build_plan(env.client, **env.plan_kwargs())
    assert plan.create_registered_model is True
    env.client.create_registered_model("email-priority")  # created behind our back
    assert apply_plan(env.client, replace(plan)) == "1"


# ---- entry points configure MLflow themselves (the Step 3a regression)


def _args(env: Env, *extra: str) -> list[str]:
    return [
        "--source-run-id",
        env.source_run_id,
        "--eval-run-id",
        env.eval_run_id,
        "--lookup",
        str(env.lookup_path),
        "--features",
        str(env.features_path),
        *extra,
    ]


def _poison_global_mlflow(
    env: Env, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Point MLflow's globals at a nonexistent file store, as a fresh shell would.

    MLFLOW_ALLOW_FILE_STORE is unset (see `mlflow_globals`), so any internal
    default client that falls back to these globals is refused. Only config's
    TRACKING_URI knows the real (temporary sqlite) location.
    """
    bogus = (tmp_path / "no_such_filestore").as_uri()
    mlflow.set_tracking_uri(bogus)
    mlflow.set_registry_uri(bogus)
    monkeypatch.setattr(config, "TRACKING_URI", env.uri)


def test_register_cli_apply_configures_mlflow_itself(
    make_env: Callable[..., Env], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    env = make_env()
    env.log_lookup()
    _poison_global_mlflow(env, tmp_path, monkeypatch)
    result = CliRunner().invoke(register_main, _args(env, "--apply"))
    assert result.exit_code == 0, result.output
    assert "Registered email-priority version 1" in result.output
    client = MlflowClient(tracking_uri=env.uri, registry_uri=env.uri)
    version = client.get_model_version_by_alias("email-priority", "champion").version
    assert str(version) == "1"


def test_serving_startup_configures_mlflow_itself(
    make_env: Callable[..., Env], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    env = make_env()
    env.log_lookup()
    apply_plan(env.client, build_plan(env.client, **env.plan_kwargs()))
    monkeypatch.setattr(config, "SENDER_LOOKUP_PATH", env.lookup_path)
    monkeypatch.setattr(config, "MODEL_URI", "models:/email-priority@champion")
    monkeypatch.setattr(config, "MODEL_SOURCE", "registry")
    _poison_global_mlflow(env, tmp_path, monkeypatch)
    app.state.predictor = None
    try:
        with TestClient(app) as client:
            health = client.get("/health").json()
        assert health["model_version"] == "1"
        assert build_predictor().model.predict(np.array([[3.0]])).tolist() == [1]
    finally:
        app.state.predictor = None


def test_log_lookup_cli_configures_mlflow_itself(
    make_env: Callable[..., Env], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    env = make_env()
    monkeypatch.setattr(
        registration_module, "DEFAULT_SERVING_ARTIFACT_ROOT", tmp_path / "serving"
    )
    _poison_global_mlflow(env, tmp_path, monkeypatch)
    result = CliRunner().invoke(
        log_lookup_main,
        ["--lookup", str(env.lookup_path), "--source-run-id", env.source_run_id],
    )
    assert result.exit_code == 0, result.output
    assert "lookup run id:" in result.output
    client = MlflowClient(tracking_uri=env.uri, registry_uri=env.uri)
    assert client.get_experiment_by_name(config.SERVING_EXPERIMENT_NAME) is not None


# ---- CLI behavior: dry run by default, refusal on re-run


def test_register_cli_dry_run_writes_nothing(
    make_env: Callable[..., Env], monkeypatch: pytest.MonkeyPatch
) -> None:
    env = make_env()
    env.log_lookup()
    monkeypatch.setattr(config, "TRACKING_URI", env.uri)
    result = CliRunner().invoke(register_main, _args(env))
    assert result.exit_code == 0, result.output
    assert "Dry run" in result.output and "tuned_threshold" in result.output
    assert env.client.search_registered_models() == []


def test_register_cli_refuses_rerun_against_existing_registration(
    make_env: Callable[..., Env], monkeypatch: pytest.MonkeyPatch
) -> None:
    env = make_env()
    env.log_lookup()
    monkeypatch.setattr(config, "TRACKING_URI", env.uri)
    first = CliRunner().invoke(register_main, _args(env, "--apply"))
    assert first.exit_code == 0, first.output
    for extra in ((), ("--apply",)):
        again = CliRunner().invoke(register_main, _args(env, *extra))
        assert again.exit_code == 1, again.output
        assert "PRECHECK FAILED" in again.output
        assert "already has version 1" in again.output
        assert "alias 'champion' already points to version 1" in again.output
    versions = env.client.search_model_versions("name='email-priority'")
    assert [str(v.version) for v in versions] == ["1"]


# ---- bundle export (read-only against the registry)


def _registered_env(make_env: Callable[..., Env]) -> Env:
    env = make_env()
    env.log_lookup()
    apply_plan(env.client, build_plan(env.client, **env.plan_kwargs()))
    return env


def test_export_bundle_is_read_only_and_matches_registry_model(
    make_env: Callable[..., Env], tmp_path: Path
) -> None:
    env = _registered_env(make_env)
    db = tmp_path / "mlflow.db"
    before = file_sha256(db)
    out = tmp_path / "bundle_out"
    metadata = export_bundle(
        env.client, "models:/email-priority@champion", out, env.lookup_path, False
    )
    assert file_sha256(db) == before, "export must not write to the registry DB"
    assert bundle.load_bundle_metadata(out) == metadata
    x = np.array([[0.0], [1.0], [2.0], [3.0]])
    np.testing.assert_allclose(
        bundle.load_bundle_model(out).predict_proba(x),
        load_model(metadata, env.uri).predict_proba(x),
        rtol=0,
        atol=1e-12,
    )


def test_export_refuses_lookup_that_differs_from_the_tag(
    make_env: Callable[..., Env], tmp_path: Path
) -> None:
    env = _registered_env(make_env)
    other = tmp_path / "other_lookup.json"
    save_sender_lookup({"x@y.com": 1}, other, build_header("c" * 64, 1))
    out = tmp_path / "bundle_out"
    with pytest.raises(ServingStartupError, match="differs from the registered"):
        export_bundle(env.client, "models:/email-priority@champion", out, other, False)
    assert not out.exists() and not out.with_name("bundle_out.partial").exists()


def test_export_refuses_to_overwrite_without_force(
    make_env: Callable[..., Env], tmp_path: Path
) -> None:
    env = _registered_env(make_env)
    out = tmp_path / "bundle_out"
    args = (env.client, "models:/email-priority@champion", out, env.lookup_path)
    export_bundle(*args, False)
    marker = out / "marker.txt"
    marker.write_text("keep")
    with pytest.raises(FileExistsError, match="--force"):
        export_bundle(*args, False)
    assert marker.exists()
    export_bundle(*args, True)
    assert not marker.exists()
    bundle.load_bundle_metadata(out)
