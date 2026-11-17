"""Export one registered model version as a verified serving bundle.

Read-only against the registry: it resolves MODEL_URI, downloads the model
artifacts to a temporary directory and writes the bundle under --out. It refuses
if the lookup file's sha256 differs from the version's `lookup_sha256` tag, and
refuses to overwrite an existing bundle without --force.

Run from the repo root::

    python -m scripts.export_model_bundle
"""

from __future__ import annotations

import logging
import platform
import shutil
import tempfile
from importlib import metadata as importlib_metadata
from pathlib import Path

import click
import mlflow.artifacts
from mlflow import MlflowClient

from src.serving import bundle, config
from src.serving.lookup import file_sha256, load_lookup
from src.serving.registry import (
    ServingMetadata,
    ServingStartupError,
    configure_mlflow,
    fetch_serving_metadata,
)

logger = logging.getLogger(__name__)

LIBRARIES: tuple[str, ...] = (
    "mlflow",
    "lightgbm",
    "numpy",
    "pandas",
    "scikit-learn",
    "scipy",
    "skops",
    "fastapi",
    "pydantic",
)


def library_versions() -> dict[str, str]:
    """Return the Python and library versions of the exporting environment."""
    versions = {"python": platform.python_version()}
    for name in LIBRARIES:
        try:
            versions[name] = importlib_metadata.version(name)
        except importlib_metadata.PackageNotFoundError:
            versions[name] = "not installed"
    return versions


def check_lookup(metadata: ServingMetadata, lookup_path: Path) -> None:
    """Refuse a lookup that is malformed or differs from the registered tag.

    Raises:
        ServingStartupError: if the sha256 differs from the lookup_sha256 tag.
    """
    load_lookup(lookup_path)
    actual = file_sha256(lookup_path)
    if actual != metadata.lookup_sha256:
        raise ServingStartupError(
            f"lookup {lookup_path} sha256 {actual} differs from the registered "
            f"lookup_sha256 tag {metadata.lookup_sha256}; refusing to export"
        )


def export_bundle(
    client: MlflowClient, model_uri: str, out_dir: Path, lookup_path: Path, force: bool
) -> ServingMetadata:
    """Resolve model_uri and write a verified bundle to out_dir.

    Raises:
        ServingStartupError: for registry problems or a lookup hash mismatch.
        FileExistsError: if out_dir exists and force is false.
    """
    if out_dir.exists() and not force:
        raise FileExistsError(f"{out_dir} already exists; pass --force to replace it")
    metadata = fetch_serving_metadata(model_uri, client)
    check_lookup(metadata, lookup_path)
    staging = out_dir.with_name(out_dir.name + ".partial")
    if staging.exists():
        shutil.rmtree(staging)
    with tempfile.TemporaryDirectory() as download_dir:
        downloaded = Path(
            mlflow.artifacts.download_artifacts(
                artifact_uri=metadata.model_uri, dst_path=download_dir
            )
        )
        requirements = downloaded / "requirements.txt"
        lines = (
            requirements.read_text(encoding="utf-8").splitlines()
            if requirements.is_file()
            else []
        )
        bundle.write_bundle(
            staging,
            metadata,
            mlmodel_path=downloaded / "MLmodel",
            model_path=downloaded / "model.skops",
            lookup_path=lookup_path,
            library_versions=library_versions(),
            model_requirements=[line for line in lines if line.strip()],
        )
    bundle.load_bundle_metadata(staging)  # verify what was just written
    if out_dir.exists():
        shutil.rmtree(out_dir)
    staging.rename(out_dir)
    return metadata


@click.command()
@click.option("--model-uri", default=config.MODEL_URI, show_default=True)
@click.option(
    "--out",
    "out_dir",
    type=click.Path(file_okay=False, path_type=Path),
    default=config.BUNDLE_DIR,
    show_default=True,
)
@click.option(
    "--lookup",
    "lookup_path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=config.SENDER_LOOKUP_PATH,
    show_default=True,
)
@click.option("--force", is_flag=True, help="Replace an existing bundle.")
def main(model_uri: str, out_dir: Path, lookup_path: Path, force: bool) -> None:
    """Export MODEL_URI and the sender lookup as a verified bundle directory."""
    logging.basicConfig(level=logging.INFO)
    configure_mlflow(config.TRACKING_URI)
    client = MlflowClient(tracking_uri=config.TRACKING_URI)
    try:
        metadata = export_bundle(client, model_uri, out_dir, lookup_path, force)
    except (FileExistsError, ServingStartupError) as exc:
        raise click.ClickException(str(exc)) from exc
    click.echo(
        f"Exported {metadata.model_name} version {metadata.version} "
        f"(threshold {metadata.threshold}) to {out_dir}"
    )


if __name__ == "__main__":
    main()
