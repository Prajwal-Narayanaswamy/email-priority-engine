"""Prechecks and writes for logging the lookup and registering the reference model.

Everything here takes an explicit `MlflowClient`, so tests can point it at a
temporary sqlite DB. Nothing writes to a finished training or evaluation run.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from mlflow import MlflowClient
from mlflow.entities import ViewType
from mlflow.exceptions import MlflowException
from mlflow.protos.databricks_pb2 import (
    ALREADY_EXISTS,
    RESOURCE_ALREADY_EXISTS,
    ErrorCode,
)

from src.serving import config
from src.serving.lookup import LookupFormatError, file_sha256, load_lookup
from src.serving.registry import (
    CHAMPION_ALIAS,
    REGISTERED_MODEL_NAME,
    ServingStartupError,
    find_logged_model_id,
    parse_threshold,
)

logger = logging.getLogger(__name__)

DEFAULT_SERVING_ARTIFACT_ROOT: Path = config.REPO_ROOT / "mlruns" / "serving"
_ALREADY_EXISTS: tuple[str, ...] = (
    ErrorCode.Name(RESOURCE_ALREADY_EXISTS),
    ErrorCode.Name(ALREADY_EXISTS),
)


class PrecheckError(RuntimeError):
    """Raised with every failed precheck listed, before anything is written."""


@dataclass(frozen=True)
class RegistrationPlan:
    """The exact writes `apply_plan` will perform."""

    model_name: str
    alias: str
    source_run_id: str
    logged_model_id: str
    source_uri: str
    tags: dict[str, str]
    create_registered_model: bool
    findings: tuple[str, ...] = ()

    def describe(self) -> list[str]:
        """Return human-readable lines: what was found, then the planned actions."""
        lines = [f"found: {finding}" for finding in self.findings]
        if self.create_registered_model:
            lines.append(f"create registered model {self.model_name!r}")
        lines.append(
            f"create a version of {self.model_name!r} from {self.source_uri} "
            f"(logged model {self.logged_model_id}, run {self.source_run_id})"
        )
        lines.extend(f"  tag {key} = {value}" for key, value in self.tags.items())
        lines.append(f"set alias {self.alias!r} to the new version")
        return lines


def ensure_experiment(client: MlflowClient, name: str, artifact_root: Path) -> str:
    """Return the experiment ID for `name`, creating it with a fixed artifact root."""
    experiment = client.get_experiment_by_name(name)
    if experiment is not None:
        return str(experiment.experiment_id)
    artifact_root.mkdir(parents=True, exist_ok=True)
    return str(client.create_experiment(name, artifact_location=artifact_root.as_uri()))


def _find_lookup_run(
    client: MlflowClient, experiment_name: str, lookup_sha256: str
) -> str | None:
    experiment = client.get_experiment_by_name(experiment_name)
    if experiment is None:
        return None
    runs = client.search_runs(
        [experiment.experiment_id],
        filter_string=(
            f"params.lookup_sha256 = '{lookup_sha256}' and status = 'FINISHED'"
        ),
        run_view_type=ViewType.ACTIVE_ONLY,
        order_by=["attributes.start_time DESC"],
        max_results=1,
    )
    return str(runs[0].info.run_id) if runs else None


def log_lookup_run(
    client: MlflowClient,
    lookup_path: Path,
    source_run_id: str,
    experiment_name: str = config.SERVING_EXPERIMENT_NAME,
    artifact_root: Path | None = None,
) -> str:
    """Log the lookup file to a NEW run in the serving experiment; return its ID.

    Idempotent: a finished run with the same lookup_sha256 is returned instead of
    creating a duplicate. Never writes to the training experiment or its runs.
    `artifact_root` defaults to `DEFAULT_SERVING_ARTIFACT_ROOT`, read at call time.

    Raises:
        LookupFormatError: if the lookup has no or a malformed header.
    """
    artifact_root = artifact_root or DEFAULT_SERVING_ARTIFACT_ROOT
    lookup = load_lookup(lookup_path)
    existing = _find_lookup_run(client, experiment_name, lookup.file_sha256)
    if existing is not None:
        logger.info("Lookup %s already logged in run %s", lookup.file_sha256, existing)
        return existing
    experiment_id = ensure_experiment(client, experiment_name, artifact_root)
    run = client.create_run(
        experiment_id,
        tags={"source_run_id": source_run_id, "mlflow.runName": "serving-lookup"},
    )
    run_id = str(run.info.run_id)
    try:
        client.log_param(run_id, "parquet_sha256", lookup.header["parquet_sha256"])
        client.log_param(run_id, "lookup_sha256", lookup.file_sha256)
        client.log_param(run_id, "row_count", lookup.header["row_count"])
        client.log_artifact(run_id, str(lookup_path))
    except Exception:
        client.set_terminated(run_id, "FAILED")
        raise
    client.set_terminated(run_id, "FINISHED")
    logger.info("Logged lookup %s to run %s", lookup.file_sha256, run_id)
    return run_id


def build_plan(
    client: MlflowClient | Any,
    *,
    source_run_id: str,
    eval_run_id: str,
    lookup_path: Path,
    features_path: Path,
    experiment_name: str = config.EXPERIMENT_NAME,
    serving_experiment_name: str = config.SERVING_EXPERIMENT_NAME,
    model_name: str = REGISTERED_MODEL_NAME,
    alias: str = CHAMPION_ALIAS,
) -> RegistrationPlan:
    """Run every precheck and return the plan.

    Raises:
        PrecheckError: listing every failed precheck, before anything is written.
    """
    problems: list[str] = []
    split_fingerprint = threshold = lookup_sha = lookup_run_id = None
    logged_model_id = None

    try:
        source = client.get_run(source_run_id)
        if "superseded" in source.data.tags:
            problems.append(
                f"source run {source_run_id} is tagged superseded="
                f"{source.data.tags['superseded']}"
            )
        split_fingerprint = source.data.params.get("split_fingerprint")
        if not split_fingerprint:
            problems.append(f"source run {source_run_id} has no split_fingerprint")
    except MlflowException as exc:
        problems.append(f"source run {source_run_id} not readable: {exc}")

    try:
        raw = client.get_run(eval_run_id).data.params.get("tuned_threshold")
        if raw is None:
            problems.append(f"evaluation run {eval_run_id} has no tuned_threshold")
        else:
            threshold = parse_threshold(raw)
    except (MlflowException, ServingStartupError) as exc:
        problems.append(f"evaluation run {eval_run_id}: {exc}")

    try:
        lookup = load_lookup(lookup_path)
        lookup_sha = lookup.file_sha256
        parquet_now = file_sha256(features_path)
        if lookup.header["parquet_sha256"] != parquet_now:
            problems.append(
                f"lookup parquet_sha256 {lookup.header['parquet_sha256']} != current "
                f"{features_path} sha256 {parquet_now}; rebuild the lookup"
            )
        lookup_run_id = _find_lookup_run(client, serving_experiment_name, lookup_sha)
        if lookup_run_id is None:
            problems.append(
                f"no finished run in {serving_experiment_name!r} holds lookup "
                f"{lookup_sha}; run scripts/log_lookup_run.py first"
            )
    except (OSError, LookupFormatError) as exc:
        problems.append(f"lookup {lookup_path}: {exc}")

    try:
        logged_model_id = find_logged_model_id(client, experiment_name, source_run_id)
    except ValueError as exc:
        problems.append(str(exc))

    model_exists = True
    findings: list[str] = []
    try:
        client.get_registered_model(model_name)
    except MlflowException:
        model_exists = False
    if not model_exists:
        findings.append(f"registered model {model_name!r} does not exist")
    else:
        versions = list(client.search_model_versions(f"name='{model_name}'"))
        if not versions:
            findings.append(
                f"registered model {model_name!r} exists with no versions "
                "(left by an earlier failed apply); it will be reused"
            )
        else:
            findings.append(
                f"registered model {model_name!r} exists with {len(versions)} "
                "version(s)"
            )
        for version in versions:
            if version.tags.get("source_run_id") == source_run_id:
                problems.append(
                    f"{model_name!r} already has version {version.version} for "
                    f"source run {source_run_id}"
                )
        try:
            current = client.get_model_version_by_alias(model_name, alias)
            problems.append(
                f"alias {alias!r} already points to version {current.version}; "
                "promotion is a separate step"
            )
        except MlflowException:
            findings.append(f"alias {alias!r} is not set")

    if problems:
        raise PrecheckError("; ".join(problems))
    assert split_fingerprint and threshold is not None and lookup_sha
    assert lookup_run_id and logged_model_id
    return RegistrationPlan(
        model_name=model_name,
        alias=alias,
        source_run_id=source_run_id,
        logged_model_id=logged_model_id,
        source_uri=f"models:/{logged_model_id}",
        tags={
            "tuned_threshold": repr(threshold),
            "threshold_source_run": eval_run_id,
            "lookup_sha256": lookup_sha,
            "lookup_run_id": lookup_run_id,
            "split_fingerprint": split_fingerprint,
            "source_run_id": source_run_id,
        },
        create_registered_model=not model_exists,
        findings=tuple(findings),
    )


def apply_plan(client: MlflowClient | Any, plan: RegistrationPlan) -> str:
    """Create the registered model if needed, the tagged version and the alias.

    Uses the client route (not mlflow.register_model), which does not write an
    `mlflow.modelVersions` tag onto the finished LoggedModel. Returns the version.
    """
    if plan.create_registered_model:
        try:
            client.create_registered_model(plan.model_name)
        except MlflowException as exc:
            if exc.error_code not in _ALREADY_EXISTS:
                raise
            logger.info("Registered model %r already exists; reusing", plan.model_name)
    version = client.create_model_version(
        name=plan.model_name,
        source=plan.source_uri,
        model_id=plan.logged_model_id,
        tags=plan.tags,
    )
    client.set_registered_model_alias(plan.model_name, plan.alias, version.version)
    logger.info(
        "Registered %s version %s with alias %s",
        plan.model_name,
        version.version,
        plan.alias,
    )
    return str(version.version)
