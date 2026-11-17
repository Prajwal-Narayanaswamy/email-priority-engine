"""Bundle format, verification, MODEL_SOURCE selection and bundle-vs-mlflow parity."""

from __future__ import annotations

import json
import logging
import subprocess
import sys
from collections.abc import Callable, Iterator
from pathlib import Path

import mlflow.sklearn
import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from src.features.columns import BEHAVIORAL_FEATURES
from src.serving import bundle, config
from src.serving.app import app, build_predictor
from src.serving.lookup import file_sha256
from src.serving.metadata import ServingMetadata, ServingStartupError
from src.serving.versions import runtime_versions
from tests.serving_helpers import make_synthetic_bundle

REPO_ROOT = Path(__file__).resolve().parents[1]
FIXTURE = REPO_ROOT / "tests/fixtures/serving_parity_rows.parquet"


@pytest.fixture
def made(tmp_path: Path) -> tuple[Path, ServingMetadata]:
    return make_synthetic_bundle(tmp_path)


@pytest.fixture
def served(
    made: tuple[Path, ServingMetadata], monkeypatch: pytest.MonkeyPatch
) -> Iterator[Path]:
    """Point serving at the synthetic bundle with MODEL_SOURCE=bundle."""
    monkeypatch.setattr(config, "MODEL_SOURCE", "bundle")
    monkeypatch.setattr(config, "BUNDLE_DIR", made[0])
    app.state.predictor = None
    yield made[0]
    app.state.predictor = None


def _flip_byte(path: Path) -> None:
    data = bytearray(path.read_bytes())
    data[len(data) // 2] ^= 0x01
    path.write_bytes(bytes(data))


def _rewrite_json(path: Path, edit: Callable[[dict], None]) -> None:
    document = json.loads(path.read_text(encoding="utf-8"))
    edit(document)
    path.write_text(json.dumps(document), encoding="utf-8")


def _rehash(bundle_dir: Path) -> None:
    """Make manifest and metadata.files agree with the files on disk again."""
    metadata_path = bundle_dir / bundle.METADATA_FILE
    _rewrite_json(
        metadata_path,
        lambda d: d["files"].update(
            {n: file_sha256(bundle_dir / n) for n in bundle.PAYLOAD_FILES}
        ),
    )
    _rewrite_json(
        bundle_dir / bundle.MANIFEST_FILE,
        lambda d: d["files"].update(
            {n: file_sha256(bundle_dir / n) for n in bundle.MANIFEST_FILES}
        ),
    )


# ---- happy path


def test_bundle_metadata_matches_what_was_written(
    made: tuple[Path, ServingMetadata],
) -> None:
    bundle_dir, expected = made
    assert bundle.load_bundle_metadata(bundle_dir) == expected
    document = json.loads((bundle_dir / bundle.METADATA_FILE).read_text())
    assert set(document["files"]) == set(bundle.PAYLOAD_FILES)
    assert document["tags"] == expected.tags
    assert document["library_versions"]["python"] == "3.12"
    assert "exported_at" in document


def test_serving_from_bundle_end_to_end(served: Path) -> None:
    with TestClient(app) as client:
        info = client.get("/model-info").json()
        body = client.post(
            "/predict", json={"from": "a@b.com", "subject": "hi?"}
        ).json()
    assert info["model_version"] == "3"
    assert info["sender_lookup_size"] == 1
    assert body["sender_seen_in_training"] is True
    assert 0.0 <= body["score"] <= 1.0


# ---- MODEL_SOURCE: explicit, no default, no fallback


@pytest.mark.parametrize("value", [None, "", "REGISTRY", "file", "bundle "])
def test_model_source_unset_or_unknown_fails_startup(
    value: str | None, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(config, "MODEL_SOURCE", value)
    app.state.predictor = None
    with pytest.raises(ServingStartupError, match="MODEL_SOURCE must be one of"):
        with TestClient(app):
            pass


def test_bundle_failure_does_not_fall_back_to_registry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(config, "MODEL_SOURCE", "bundle")
    monkeypatch.setattr(config, "BUNDLE_DIR", tmp_path / "absent")
    app.state.predictor = None
    with pytest.raises(ServingStartupError, match="bundle directory"):
        build_predictor()


# ---- bundle failure modes (each names what is wrong)


def test_missing_bundle_directory(served: Path, tmp_path: Path) -> None:
    with pytest.raises(ServingStartupError, match="bundle directory .* not found"):
        bundle.load_bundle_metadata(tmp_path / "nope")


def test_missing_metadata_json(served: Path) -> None:
    (served / bundle.METADATA_FILE).unlink()
    with pytest.raises(ServingStartupError, match="metadata.json is missing"):
        build_predictor()


def test_missing_manifest(served: Path) -> None:
    (served / bundle.MANIFEST_FILE).unlink()
    with pytest.raises(ServingStartupError, match="manifest.json is missing"):
        build_predictor()


def test_flipped_byte_in_lookup(served: Path) -> None:
    _flip_byte(served / bundle.LOOKUP_FILE)
    with pytest.raises(
        ServingStartupError, match="sender_frequency.json sha256 .* manifest"
    ):
        build_predictor()


def test_flipped_byte_in_model(served: Path) -> None:
    _flip_byte(served / bundle.MODEL_FILE)
    with pytest.raises(ServingStartupError, match="model.skops sha256 .* manifest"):
        build_predictor()


def test_manifest_hash_mismatch(served: Path) -> None:
    _rewrite_json(
        served / bundle.MANIFEST_FILE,
        lambda d: d["files"].update({bundle.MLMODEL_FILE: "0" * 64}),
    )
    with pytest.raises(ServingStartupError, match="MLmodel sha256 .* manifest"):
        build_predictor()


def test_manifest_with_unexpected_file_list(served: Path) -> None:
    _rewrite_json(
        served / bundle.MANIFEST_FILE, lambda d: d["files"].update({"extra": "0" * 64})
    )
    with pytest.raises(ServingStartupError, match="must list exactly"):
        build_predictor()


def test_metadata_hashes_disagree_with_manifest(served: Path) -> None:
    _rewrite_json(
        served / bundle.METADATA_FILE,
        lambda d: d["files"].update({bundle.MODEL_FILE: "0" * 64}),
    )
    _rewrite_json(
        served / bundle.MANIFEST_FILE,
        lambda d: d["files"].update(
            {bundle.METADATA_FILE: file_sha256(served / bundle.METADATA_FILE)}
        ),
    )
    with pytest.raises(ServingStartupError, match="do not match the manifest"):
        build_predictor()


def test_lookup_replaced_consistently_still_fails_the_tag_check(served: Path) -> None:
    """A different lookup with matching manifest/metadata hashes is still refused."""
    lookup = served / bundle.LOOKUP_FILE
    document = json.loads(lookup.read_text())
    document["senders"]["z@z.com"] = 9
    lookup.write_text(json.dumps(document), encoding="utf-8")
    _rehash(served)
    with pytest.raises(ServingStartupError, match="does not match the registered"):
        build_predictor()


def test_bad_threshold_tag_in_bundle(served: Path) -> None:
    _rewrite_json(
        served / bundle.METADATA_FILE,
        lambda d: d["tags"].update({"tuned_threshold": "1.5"}),
    )
    _rehash(served)
    with pytest.raises(ServingStartupError, match="outside the open interval"):
        build_predictor()


def test_missing_tag_in_bundle(served: Path) -> None:
    _rewrite_json(
        served / bundle.METADATA_FILE, lambda d: d["tags"].pop("split_fingerprint")
    )
    _rehash(served)
    with pytest.raises(ServingStartupError, match="missing tags.*split_fingerprint"):
        build_predictor()


def test_unsupported_bundle_format_version(served: Path) -> None:
    _rewrite_json(
        served / bundle.METADATA_FILE, lambda d: d.update(bundle_format_version=99)
    )
    _rehash(served)
    with pytest.raises(ServingStartupError, match="format version 99"):
        build_predictor()


def test_model_with_untrusted_types_is_refused(served: Path) -> None:
    mlmodel = served / bundle.MLMODEL_FILE
    mlmodel.write_text(
        mlmodel.read_text().replace("lightgbm.sklearn.LGBMClassifier", "os.system"),
        encoding="utf-8",
    )
    _rehash(served)
    with pytest.raises(ServingStartupError, match="cannot load"):
        build_predictor()


# ---- bundle loader vs mlflow.sklearn.load_model (the condition for dropping mlflow)


def test_bundle_loader_matches_mlflow_loader_on_fixture(
    made: tuple[Path, ServingMetadata],
) -> None:
    bundle_dir, _ = made
    rows = pd.read_parquet(FIXTURE)[list(BEHAVIORAL_FEATURES)]
    assert len(rows) == 300
    via_bundle = bundle.load_bundle_model(bundle_dir).predict_proba(rows)
    via_mlflow = mlflow.sklearn.load_model(str(bundle_dir / "model")).predict_proba(
        rows
    )
    np.testing.assert_allclose(via_bundle, via_mlflow, rtol=0, atol=1e-12)


# ---- the serving image must not need the training stack or mlflow


@pytest.mark.parametrize(
    "module", ["src.serving.app", "src.serving.bundle", "src.serving.predictor"]
)
def test_serving_imports_exclude_training_and_mlflow(module: str) -> None:
    code = (
        "import sys, importlib\n"
        f"importlib.import_module({module!r})\n"
        "bad = [m for m in ('matplotlib', 'mlflow', 'src.training.train', 'prefect',"
        " 'evidently') if m in sys.modules]\n"
        "print(','.join(bad))\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    assert result.stdout.strip() == "", f"{module} imported: {result.stdout.strip()}"


# ---- runtime versus bundle library versions (warn, never fail)


def test_model_info_reports_runtime_and_bundle_versions(served: Path) -> None:
    with TestClient(app) as client:
        info = client.get("/model-info").json()
    assert info["runtime_versions"] == runtime_versions()
    assert info["bundle_versions"] == {"python": "3.12"}


def test_startup_warns_but_still_serves_when_versions_differ(
    served: Path, caplog: pytest.LogCaptureFixture
) -> None:
    _rewrite_json(
        served / bundle.METADATA_FILE,
        lambda d: d["library_versions"].update({"lightgbm": "0.0.1"}),
    )
    _rehash(served)
    with caplog.at_level(logging.WARNING, logger="src.serving.versions"):
        with TestClient(app) as client:
            assert client.get("/health").status_code == 200
    assert any(
        "lightgbm runtime" in r.getMessage() and "vs bundle 0.0.1" in r.getMessage()
        for r in caplog.records
        if r.levelno == logging.WARNING
    )


def test_startup_is_silent_when_versions_match(
    served: Path, caplog: pytest.LogCaptureFixture
) -> None:
    current = runtime_versions()
    _rewrite_json(
        served / bundle.METADATA_FILE,
        lambda d: d["library_versions"].update(current),
    )
    _rehash(served)
    with caplog.at_level(logging.WARNING, logger="src.serving.versions"):
        build_predictor()
    assert [r for r in caplog.records if r.levelno >= logging.WARNING] == []
