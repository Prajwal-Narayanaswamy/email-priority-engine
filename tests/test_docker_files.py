"""Static checks of the serving image files (no Docker needed).

Covers the lock file, requirements-serving.in vs the pyproject `serving` extra,
the Dockerfile, .dockerignore and docker-compose.yml (ADR-015).
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
import tomllib
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
LOCK = ROOT / "requirements-serving.lock"
REQ_IN = ROOT / "requirements-serving.in"
DOCKERFILE = ROOT / "docker/api/Dockerfile"
DOCKERIGNORE = ROOT / ".dockerignore"
COMPOSE = ROOT / "docker-compose.yml"
EXPECTED = ROOT / "tests/fixtures/serving_expected_versions.json"

FORBIDDEN_IN_IMAGE = (
    "mlflow",
    "matplotlib",
    "prefect",
    "evidently",
    "dvc",
    "onnx",
    "onnxruntime",
    "skl2onnx",
    "nltk",
    "prometheus-client",
    "pytest",
)
NEVER_COPIED = ("data", "mlruns", "serving_bundle", ".venv", ".git", "tests", ".env")


def _lock_pins() -> dict[str, str]:
    pins = {}
    for line in LOCK.read_text(encoding="utf-8").splitlines():
        match = re.match(r"^([A-Za-z0-9_.-]+)==([^\s\\]+)", line)
        if match:
            pins[match.group(1).lower().replace("_", "-")] = match.group(2)
    return pins


def _dockerfile_instructions() -> list[str]:
    """Instructions with line continuations joined and comments removed."""
    text = DOCKERFILE.read_text(encoding="utf-8")
    text = re.sub(r"\\\n", " ", text)
    return [
        line.strip()
        for line in text.splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]


def _dockerignore_patterns() -> set[str]:
    return {
        line.strip()
        for line in DOCKERIGNORE.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.startswith("#")
    }


# ---- lock file


def test_model_relevant_pins_equal_the_training_environment() -> None:
    expected = json.loads(EXPECTED.read_text(encoding="utf-8"))["versions"]
    assert len(expected) == 6
    pins = _lock_pins()
    assert {name: pins.get(name) for name in expected} == expected


def test_every_pin_has_hashes_and_nothing_heavy_is_locked() -> None:
    text = LOCK.read_text(encoding="utf-8")
    pins = _lock_pins()
    assert len(pins) >= 20
    blocks = re.split(r"\n(?=[A-Za-z0-9_.-]+==)", text)
    for block in blocks:
        if re.match(r"[A-Za-z0-9_.-]+==", block):
            assert "--hash=sha256:" in block, block.splitlines()[0]
    assert not set(FORBIDDEN_IN_IMAGE) & set(pins)


def test_lock_targets_linux_python_312() -> None:
    header = LOCK.read_text(encoding="utf-8").splitlines()[:3]
    command = " ".join(header)
    assert "--generate-hashes" in command
    assert "--python-version 3.12" in command
    assert "linux" in command


def test_requirements_in_matches_the_pyproject_serving_extra() -> None:
    extra = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"][
        "optional-dependencies"
    ]["serving"]
    listed = [
        line.strip()
        for line in REQ_IN.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.startswith("#")
    ]
    assert sorted(listed) == sorted(extra)


# ---- Dockerfile


def test_dockerfile_base_and_openmp() -> None:
    lines = _dockerfile_instructions()
    assert lines[0] == "FROM python:3.12-slim-bookworm"
    run_apt = " ".join(line for line in lines if "apt-get" in line)
    assert "libgomp1" in run_apt
    assert "--no-install-recommends" in run_apt
    assert "rm -rf /var/lib/apt/lists/*" in run_apt


def test_dockerfile_runs_as_a_non_root_user() -> None:
    users = [line for line in _dockerfile_instructions() if line.startswith("USER ")]
    assert users, "no USER instruction"
    final = users[-1].split(maxsplit=1)[1].split(":")[0]
    assert final not in {"root", "0"}
    assert final == "10001"


def test_dockerfile_healthcheck_uses_only_the_standard_library() -> None:
    lines = _dockerfile_instructions()
    health = [line for line in lines if line.startswith("HEALTHCHECK")]
    assert len(health) == 1
    assert "curl" not in health[0].lower() and "wget" not in health[0].lower()
    assert "urllib.request" in health[0] and "/health" in health[0]
    assert "--start-period" in health[0]
    assert not any("curl" in line.lower() for line in lines)


def test_dockerfile_installs_from_the_hashed_lock_with_env_and_cmd() -> None:
    text = "\n".join(_dockerfile_instructions())
    assert "--require-hashes -r requirements-serving.lock" in text
    assert "--no-cache-dir" in text
    assert "PYTHONUNBUFFERED=1" in text and "PYTHONDONTWRITEBYTECODE=1" in text
    assert "MODEL_SOURCE=bundle" in text and "MODEL_BUNDLE_DIR=/bundle" in text
    assert "EXPOSE 8000" in text
    assert '"src.serving.app:app"' in text and '"--port", "8000"' in text
    assert "mlflow" not in text.lower()


def test_dockerfile_copies_nothing_it_must_not() -> None:
    copies = [line for line in _dockerfile_instructions() if line.startswith("COPY ")]
    assert copies
    for line in copies:
        sources = [
            p.removeprefix("./") for p in line.split()[1:-1] if not p.startswith("--")
        ]
        for source in sources:
            assert source not in {"", "."}, line
            first = source.split("/")[0]
            assert first not in NEVER_COPIED, line


def test_dockerfile_copies_exactly_the_import_closure_of_the_app() -> None:
    code = (
        "import sys, src.serving.app\n"
        "print('\\n'.join(sorted({m.split('.')[1] for m in sys.modules "
        "if m.startswith('src.')})))\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    closure = set(result.stdout.split())
    copied = set()
    for line in _dockerfile_instructions():
        if line.startswith("COPY src/"):
            source = line.split()[1]
            copied.add(source.split("/")[1])
    assert "COPY src/__init__.py ./src/__init__.py" in _dockerfile_instructions()
    copied.discard("__init__.py")
    assert copied == closure


# ---- .dockerignore


@pytest.mark.parametrize(
    "entry",
    [
        ".git",
        ".venv",
        "data",
        "mlruns",
        "serving_bundle",
        "notebooks",
        "reports",
        "tests",
        "docs",
        "*.log",
        ".env*",
        ".claude",
        "**/__pycache__",
    ],
)
def test_dockerignore_excludes(entry: str) -> None:
    assert entry in _dockerignore_patterns()


def test_dockerignore_does_not_exclude_what_the_image_copies() -> None:
    patterns = _dockerignore_patterns()
    for needed in ("requirements-serving.lock", "src", "src/serving", "src/features"):
        assert needed not in patterns


# ---- docker-compose.yml


def _service() -> dict:
    compose = yaml.safe_load(COMPOSE.read_text(encoding="utf-8"))
    assert list(compose["services"]) == ["api"]
    assert "version" not in compose
    return compose["services"]["api"]


def test_compose_has_only_the_hardened_api_service() -> None:
    api = _service()
    assert api["build"]["dockerfile"] == "docker/api/Dockerfile"
    assert api["ports"] == ["127.0.0.1:${API_PORT:-8000}:8000"]
    assert api["read_only"] is True
    assert "/tmp" in api["tmpfs"]
    assert api["cap_drop"] == ["ALL"]
    assert "no-new-privileges:true" in api["security_opt"]
    assert api["environment"]["MODEL_SOURCE"] == "bundle"


def test_compose_mounts_the_bundle_read_only_with_create_host_path_false() -> None:
    # Only the configuration is asserted. Docker Desktop on WSL was observed to
    # create an empty root-owned serving_bundle/ anyway (ADR-015).
    (volume,) = _service()["volumes"]
    assert volume["source"] == "./serving_bundle"
    assert volume["target"] == "/bundle" and volume["read_only"] is True
    assert volume["bind"]["create_host_path"] is False


def test_compose_healthcheck_matches_the_image() -> None:
    check = _service()["healthcheck"]
    assert "curl" not in json.dumps(check).lower()
    assert "urllib.request" in check["test"][-1]
    assert check["start_period"] == "40s"
