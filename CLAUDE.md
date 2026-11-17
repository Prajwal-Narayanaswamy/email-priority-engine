@AGENTS.md

# Claude Code notes

## Environment
- Ubuntu 24.04 on WSL2; Python 3.12.3; Docker through Docker Desktop's WSL
  integration; git workflow per AGENTS.md.
- Working directory: `~/code/project-a`. Activate with
  `source .venv/bin/activate`.

## How to run
- Tests: `pytest -q` (full suite takes about 6-7 min, so the user's terminal).
- Flows: `python -m src.flows.<name>`; export the serving bundle with
  `python -m scripts.export_model_bundle`.

## Current stage
See SESSION_LOG.md, newest entry, "Next session".
