"""Model bundle: a verified, self-contained snapshot of one registered version.

Layout (ADR-015)::

    <bundle>/
      model/MLmodel            flavor config, incl. the skops trusted types
      model/model.skops        the serialized model
      sender_frequency.json    the versioned sender lookup (v2)
      metadata.json            name, version, threshold, tags, file hashes, ...
      manifest.json            sha256 of every file above, metadata.json included

The manifest guards against corruption and accidental edits; it is not a
signature, so anyone who can rewrite the bundle can rewrite the manifest too.

This module must not import mlflow: the bundle path runs in an image without it.
"""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

from src.serving.lookup import file_sha256
from src.serving.metadata import (
    ServingMetadata,
    ServingStartupError,
    check_required_tags,
    parse_threshold,
)

logger = logging.getLogger(__name__)

BUNDLE_FORMAT_VERSION: int = 1
MLMODEL_FILE: str = "model/MLmodel"
MODEL_FILE: str = "model/model.skops"
LOOKUP_FILE: str = "sender_frequency.json"
METADATA_FILE: str = "metadata.json"
MANIFEST_FILE: str = "manifest.json"
#: Files whose hashes are recorded inside metadata.json.
PAYLOAD_FILES: tuple[str, ...] = (MLMODEL_FILE, MODEL_FILE, LOOKUP_FILE)
#: Every file the manifest must list, and nothing else.
MANIFEST_FILES: tuple[str, ...] = (*PAYLOAD_FILES, METADATA_FILE)


def write_bundle(
    out_dir: Path,
    metadata: ServingMetadata,
    mlmodel_path: Path,
    model_path: Path,
    lookup_path: Path,
    library_versions: dict[str, str],
    model_requirements: list[str],
) -> dict[str, str]:
    """Copy the payload into out_dir and write metadata.json and manifest.json.

    out_dir must not exist yet (the caller decides about overwriting). Returns
    the manifest's file hashes.
    """
    out_dir.mkdir(parents=True)
    (out_dir / "model").mkdir()
    for relative, source in (
        (MLMODEL_FILE, mlmodel_path),
        (MODEL_FILE, model_path),
        (LOOKUP_FILE, lookup_path),
    ):
        (out_dir / relative).write_bytes(source.read_bytes())
    payload_hashes = {name: file_sha256(out_dir / name) for name in PAYLOAD_FILES}
    document = {
        "bundle_format_version": BUNDLE_FORMAT_VERSION,
        "model_name": metadata.model_name,
        "version": metadata.version,
        "alias": metadata.alias,
        "threshold": metadata.threshold,
        "lookup_sha256": metadata.lookup_sha256,
        "tags": metadata.tags,
        "files": payload_hashes,
        "library_versions": library_versions,
        "model_requirements": model_requirements,
        "exported_at": datetime.now(UTC).isoformat(timespec="seconds"),
    }
    (out_dir / METADATA_FILE).write_text(
        json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    manifest_hashes = {name: file_sha256(out_dir / name) for name in MANIFEST_FILES}
    manifest = {
        "bundle_format_version": BUNDLE_FORMAT_VERSION,
        "files": manifest_hashes,
    }
    (out_dir / MANIFEST_FILE).write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest_hashes


def _read_json(path: Path, label: str) -> dict[str, Any]:
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ServingStartupError(f"bundle {label} is not valid JSON: {exc}") from exc
    if not isinstance(document, dict):
        raise ServingStartupError(f"bundle {label} must be a JSON object")
    return document


def verify_manifest(bundle_dir: Path) -> dict[str, str]:
    """Check that every bundle file exists and matches the manifest's sha256.

    Returns the manifest's file hashes.

    Raises:
        ServingStartupError: for a missing directory or file, a malformed or
            inconsistent manifest, or any hash mismatch (the file is named).
    """
    if not bundle_dir.is_dir():
        raise ServingStartupError(
            f"model bundle directory {bundle_dir} not found; create it with "
            "scripts/export_model_bundle.py"
        )
    for name in (MANIFEST_FILE, METADATA_FILE):
        if not (bundle_dir / name).is_file():
            raise ServingStartupError(f"bundle file {name} is missing in {bundle_dir}")
    manifest = _read_json(bundle_dir / MANIFEST_FILE, MANIFEST_FILE)
    files = manifest.get("files")
    if not isinstance(files, dict) or set(files) != set(MANIFEST_FILES):
        raise ServingStartupError(
            f"bundle manifest must list exactly {sorted(MANIFEST_FILES)}; "
            f"found {sorted(files) if isinstance(files, dict) else files!r}"
        )
    for name in MANIFEST_FILES:
        path = bundle_dir / name
        if not path.is_file():
            raise ServingStartupError(f"bundle file {name} is missing in {bundle_dir}")
        actual = file_sha256(path)
        if actual != files[name]:
            raise ServingStartupError(
                f"bundle file {name} sha256 {actual} does not match the manifest "
                f"({files[name]}); the bundle is corrupt or was modified"
            )
    return {name: str(files[name]) for name in MANIFEST_FILES}


def load_bundle_metadata(bundle_dir: Path) -> ServingMetadata:
    """Verify the bundle and return its ServingMetadata.

    Applies the same tag and threshold validation as the registry source.

    Raises:
        ServingStartupError: on any verification or validation failure.
    """
    manifest_hashes = verify_manifest(bundle_dir)
    document = _read_json(bundle_dir / METADATA_FILE, METADATA_FILE)
    if document.get("bundle_format_version") != BUNDLE_FORMAT_VERSION:
        raise ServingStartupError(
            f"bundle format version {document.get('bundle_format_version')!r} is "
            f"not supported (expected {BUNDLE_FORMAT_VERSION})"
        )
    recorded = document.get("files")
    expected = {name: manifest_hashes[name] for name in PAYLOAD_FILES}
    if recorded != expected:
        raise ServingStartupError(
            "bundle metadata.json file hashes do not match the manifest; "
            "re-export the bundle"
        )
    tags = document.get("tags")
    if not isinstance(tags, dict):
        raise ServingStartupError("bundle metadata.json has no tags object")
    check_required_tags(tags, f"bundle {bundle_dir}")
    threshold = parse_threshold(tags["tuned_threshold"])
    if document.get("threshold") != threshold:
        raise ServingStartupError(
            f"bundle threshold {document.get('threshold')!r} does not match the "
            f"tuned_threshold tag {tags['tuned_threshold']!r}"
        )
    try:
        model_name = str(document["model_name"])
        version = str(document["version"])
    except KeyError as exc:
        raise ServingStartupError(f"bundle metadata.json lacks {exc}") from exc
    metadata = ServingMetadata(
        model_name=model_name,
        version=version,
        alias=document.get("alias"),
        threshold=threshold,
        lookup_sha256=tags["lookup_sha256"],
        tags={str(k): str(v) for k, v in tags.items()},
    )
    if document.get("lookup_sha256") != metadata.lookup_sha256:
        raise ServingStartupError(
            "bundle metadata.json lookup_sha256 differs from the lookup_sha256 tag"
        )
    return metadata


def trusted_types_from_mlmodel(mlmodel_path: Path) -> list[str]:
    """Read the skops trusted types the model was logged with from its MLmodel.

    Raises:
        ServingStartupError: if the file is unreadable or not a skops sklearn model.
    """
    try:
        config = yaml.safe_load(mlmodel_path.read_text(encoding="utf-8"))
        flavor = config["flavors"]["sklearn"]
        if flavor["serialization_format"] != "skops":
            raise ServingStartupError(
                f"{mlmodel_path} serialization_format is "
                f"{flavor['serialization_format']!r}, expected 'skops'"
            )
        types = flavor.get("skops_trusted_types") or []
    except ServingStartupError:
        raise
    except (OSError, yaml.YAMLError, KeyError, TypeError) as exc:
        raise ServingStartupError(
            f"cannot read skops_trusted_types from {mlmodel_path}: {exc!r}"
        ) from exc
    if not isinstance(types, list) or not all(isinstance(t, str) for t in types):
        raise ServingStartupError(f"{mlmodel_path} skops_trusted_types is malformed")
    return list(types)


def load_bundle_model(bundle_dir: Path) -> Any:
    """Load model.skops, trusting exactly the types listed in the bundle's MLmodel.

    Call after `load_bundle_metadata` has verified the files.

    Raises:
        ServingStartupError: if the model cannot be loaded.
    """
    import skops.io

    trusted = trusted_types_from_mlmodel(bundle_dir / MLMODEL_FILE)
    logger.info(
        "Loading %s with %d trusted types", bundle_dir / MODEL_FILE, len(trusted)
    )
    try:
        return skops.io.load(bundle_dir / MODEL_FILE, trusted=trusted)
    except Exception as exc:  # noqa: BLE001 - any load failure blocks startup
        raise ServingStartupError(
            f"cannot load {bundle_dir / MODEL_FILE}: {exc}"
        ) from exc
