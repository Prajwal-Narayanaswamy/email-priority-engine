"""Stub-based tests for registry resolution, lookup v2 and every startup failure."""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient
from mlflow.exceptions import MlflowException
from mlflow.protos.databricks_pb2 import INTERNAL_ERROR, RESOURCE_DOES_NOT_EXIST

from src.serving import config
from src.serving.app import app, build_predictor
from src.serving.lookup import (
    LookupFormatError,
    build_header,
    file_sha256,
    load_lookup,
    save_sender_lookup,
)
from src.serving.registry import (
    REQUIRED_TAGS,
    ServingStartupError,
    fetch_serving_metadata,
    load_model,
    parse_model_uri,
)
from tests.serving_helpers import LOOKUP_SHA, StubModel, make_metadata, make_tags


def _missing(message: str) -> MlflowException:
    return MlflowException(message, error_code=RESOURCE_DOES_NOT_EXIST)


class FakeVersion:
    """Minimal ModelVersion stand-in."""

    def __init__(self, tags: dict[str, str], version: str = "3") -> None:
        self.tags = tags
        self.version = version


class FakeClient:
    """Minimal MlflowClient stand-in for the registry reads serving performs."""

    def __init__(
        self,
        *,
        model_exists: bool = True,
        alias_tags: dict[str, str] | None = None,
        version_tags: dict[str, str] | None = None,
        registry_error: bool = False,
    ) -> None:
        self.model_exists = model_exists
        self.alias_tags = alias_tags
        self.version_tags = version_tags
        self.registry_error = registry_error

    def get_registered_model(self, name: str) -> object:
        if self.registry_error:
            raise MlflowException("db down", error_code=INTERNAL_ERROR)
        if not self.model_exists:
            raise _missing(f"Registered Model with name={name} not found")
        return object()

    def get_model_version_by_alias(self, name: str, alias: str) -> FakeVersion:
        if self.alias_tags is None:
            raise _missing(f"Registered model alias {alias} not found.")
        return FakeVersion(self.alias_tags)

    def get_model_version(self, name: str, version: str) -> FakeVersion:
        if self.version_tags is None:
            raise _missing(f"Model Version (name={name}, version={version}) not found")
        return FakeVersion(self.version_tags, version)


ALIAS_URI = "models:/email-priority@champion"


def test_fetch_metadata_resolves_alias_once_and_reads_tags() -> None:
    client = FakeClient(alias_tags=make_tags())
    meta = fetch_serving_metadata(ALIAS_URI, client)
    assert meta.model_name == "email-priority"
    assert meta.alias == "champion"
    assert meta.version == "3"
    assert meta.threshold == 0.99
    assert meta.lookup_sha256 == LOOKUP_SHA
    assert meta.model_uri == "models:/email-priority/3"
    assert set(REQUIRED_TAGS) <= set(meta.tags)


def test_fetch_metadata_by_version_uri() -> None:
    client = FakeClient(version_tags=make_tags())
    meta = fetch_serving_metadata("models:/email-priority/3", client)
    assert meta.alias is None and meta.version == "3"


def test_missing_registered_model() -> None:
    with pytest.raises(ServingStartupError, match="registered model 'email-priority'"):
        fetch_serving_metadata(ALIAS_URI, FakeClient(model_exists=False))


def test_missing_alias() -> None:
    with pytest.raises(ServingStartupError, match="alias 'champion'.*not found"):
        fetch_serving_metadata(ALIAS_URI, FakeClient(alias_tags=None))


def test_missing_version() -> None:
    with pytest.raises(ServingStartupError, match="version 7.*not found"):
        fetch_serving_metadata("models:/email-priority/7", FakeClient())


@pytest.mark.parametrize("tag", REQUIRED_TAGS)
def test_missing_tag_is_named(tag: str) -> None:
    tags = make_tags()
    del tags[tag]
    with pytest.raises(ServingStartupError, match=f"missing tags.*{tag}"):
        fetch_serving_metadata(ALIAS_URI, FakeClient(alias_tags=tags))


@pytest.mark.parametrize("raw", ["abc", "", "nan", "inf"])
def test_non_numeric_or_non_finite_threshold(raw: str) -> None:
    client = FakeClient(alias_tags=make_tags(tuned_threshold=raw))
    with pytest.raises(ServingStartupError, match="tuned_threshold"):
        fetch_serving_metadata(ALIAS_URI, client)


@pytest.mark.parametrize("raw", ["0", "1", "1.5", "-0.1", "0.0", "1.0"])
def test_threshold_outside_open_unit_interval(raw: str) -> None:
    client = FakeClient(alias_tags=make_tags(tuned_threshold=raw))
    with pytest.raises(ServingStartupError, match=r"outside the open interval"):
        fetch_serving_metadata(ALIAS_URI, client)


def test_registry_error_is_not_reported_as_missing() -> None:
    with pytest.raises(ServingStartupError, match="cannot read the model registry"):
        fetch_serving_metadata(ALIAS_URI, FakeClient(registry_error=True))


@pytest.mark.parametrize(
    "uri", ["email-priority", "models:/m-123", "models:/a/latest", "runs:/abc/model"]
)
def test_bad_model_uri(uri: str) -> None:
    with pytest.raises(ServingStartupError, match="MODEL_URI"):
        parse_model_uri(uri)


def test_load_model_failure_is_wrapped(monkeypatch: pytest.MonkeyPatch) -> None:
    def boom(uri: str) -> object:
        raise OSError("artifact path does not exist")

    monkeypatch.setattr("src.serving.registry.mlflow.sklearn.load_model", boom)
    monkeypatch.setattr("src.serving.registry.configure_mlflow", lambda u: None)
    with pytest.raises(ServingStartupError, match="cannot load model.*/3"):
        load_model(make_metadata(), "sqlite:///unused.db")


# ---- lookup v2


def _write_lookup(path: Path, **header_overrides: Any) -> str:
    header = {**build_header("b" * 64, 10), **header_overrides}
    payload = {"header": header, "senders": {"a@b.com": 2}}
    path.write_text(json.dumps(payload), encoding="utf-8")
    return file_sha256(path)


def test_lookup_roundtrip_and_file_hash(tmp_path: Path) -> None:
    path = tmp_path / "l.json"
    digest = save_sender_lookup({"a@b.com": 2}, path, build_header("b" * 64, 10))
    loaded = load_lookup(path)
    assert loaded.senders == {"a@b.com": 2}
    assert loaded.file_sha256 == digest == file_sha256(path)
    assert loaded.header["parquet_sha256"] == "b" * 64


def test_lookup_without_header_rejected(tmp_path: Path) -> None:
    path = tmp_path / "v1.json"
    path.write_text(json.dumps({"a@b.com": 2}), encoding="utf-8")
    with pytest.raises(LookupFormatError, match="no header"):
        load_lookup(path)


def test_lookup_not_json_rejected(tmp_path: Path) -> None:
    path = tmp_path / "bad.json"
    path.write_text("{not json", encoding="utf-8")
    with pytest.raises(LookupFormatError, match="not valid JSON"):
        load_lookup(path)


@pytest.mark.parametrize(
    "overrides, message",
    [
        ({"parquet_sha256": "xyz"}, "parquet_sha256"),
        ({"parquet_sha256": "A" * 64}, "parquet_sha256"),
        ({"row_count": 0}, "row_count"),
        ({"row_count": "10"}, "row_count"),
        ({"row_count": True}, "row_count"),
        ({"built_at": "yesterday"}, "built_at"),
        ({"builder_version": ""}, "builder_version"),
        ({"format_version": 1}, "format_version"),
    ],
)
def test_lookup_malformed_header_rejected(
    tmp_path: Path, overrides: dict[str, Any], message: str
) -> None:
    path = tmp_path / "l.json"
    _write_lookup(path, **overrides)
    with pytest.raises(LookupFormatError, match=message):
        load_lookup(path)


def test_lookup_missing_header_keys_rejected(tmp_path: Path) -> None:
    path = tmp_path / "l.json"
    payload = {"header": {"format_version": 2}, "senders": {}}
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(LookupFormatError, match="missing keys"):
        load_lookup(path)


def test_save_rejects_bad_header(tmp_path: Path) -> None:
    with pytest.raises(LookupFormatError):
        save_sender_lookup({}, tmp_path / "l.json", {"format_version": 2})


# ---- startup (build_predictor / lifespan)


@pytest.fixture
def startup(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Iterator[Path]:
    """Patch registry access so build_predictor touches no MLflow store."""
    lookup_path = tmp_path / "lookup.json"
    monkeypatch.setattr(config, "SENDER_LOOKUP_PATH", lookup_path)
    monkeypatch.setattr("src.serving.app.MlflowClient", lambda **kw: object())
    monkeypatch.setattr("src.serving.app.configure_mlflow", lambda uri: None)
    app.state.predictor = None
    yield lookup_path
    app.state.predictor = None


def _patch_registry(
    monkeypatch: pytest.MonkeyPatch, lookup_sha: str, *, load_error: bool = False
) -> None:
    metadata = make_metadata(lookup_sha256=lookup_sha)
    monkeypatch.setattr(
        "src.serving.app.fetch_serving_metadata", lambda uri, client: metadata
    )

    def fake_load(meta: Any, tracking_uri: str) -> Any:
        if load_error:
            raise ServingStartupError("cannot load model models:/email-priority/3")
        return StubModel()

    monkeypatch.setattr("src.serving.app.load_model", fake_load)


def test_startup_succeeds_and_model_info_reports_registry_state(
    monkeypatch: pytest.MonkeyPatch, startup: Path
) -> None:
    digest = _write_lookup(startup)
    _patch_registry(monkeypatch, digest)
    with TestClient(app) as client:
        info = client.get("/model-info").json()
    assert info["lookup_sha256"] == digest
    assert info["alias"] == "champion"
    assert info["sender_lookup_size"] == 1


def test_startup_fails_on_lookup_hash_mismatch(
    monkeypatch: pytest.MonkeyPatch, startup: Path
) -> None:
    _write_lookup(startup)
    _patch_registry(monkeypatch, "c" * 64)
    with pytest.raises(ServingStartupError, match="does not match the registered"):
        build_predictor()


def test_startup_fails_when_lookup_has_no_header(
    monkeypatch: pytest.MonkeyPatch, startup: Path
) -> None:
    startup.write_text(json.dumps({"a@b.com": 2}), encoding="utf-8")
    _patch_registry(monkeypatch, LOOKUP_SHA)
    with pytest.raises(LookupFormatError, match="no header"):
        build_predictor()


def test_startup_fails_when_lookup_file_missing(
    monkeypatch: pytest.MonkeyPatch, startup: Path
) -> None:
    _patch_registry(monkeypatch, LOOKUP_SHA)
    with pytest.raises(FileNotFoundError):
        build_predictor()


def test_startup_fails_when_model_cannot_load(
    monkeypatch: pytest.MonkeyPatch, startup: Path
) -> None:
    digest = _write_lookup(startup)
    _patch_registry(monkeypatch, digest, load_error=True)
    with pytest.raises(ServingStartupError, match="cannot load model"):
        with TestClient(app):
            pass


def test_startup_fails_on_registry_problem_before_reading_lookup(
    monkeypatch: pytest.MonkeyPatch, startup: Path
) -> None:
    def boom(uri: str, client: Any) -> Any:
        raise ServingStartupError("alias 'champion' of registered model not found")

    monkeypatch.setattr("src.serving.app.fetch_serving_metadata", boom)
    with pytest.raises(ServingStartupError, match="alias 'champion'"):
        build_predictor()


def test_no_run_id_fallback_exists() -> None:
    assert not hasattr(config, "MODEL_RUN_ID")
    assert not hasattr(config, "EVAL_RUN_ID")
    assert not hasattr(config, "DEFAULT_MODEL_RUN_ID")
