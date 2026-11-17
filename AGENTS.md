# Agent Instructions

Email priority classification on the Enron corpus: a fully local MLOps pipeline
(Prefect, MLflow, FastAPI, Docker). No cloud inference, no external prediction
APIs. These rules apply to any coding agent.

## Working agreement
- Before any work: read SESSION_LOG.md (newest entry on top, "Next session"
  section) and ARCHITECTURE.md, and confirm the current stage.
- Sessions are short (3-5 h/week). Prefer simple working code over clever code.
  Write complete files, no placeholder `pass` functions.
- Diagnose first: explain the root cause before writing a fix. Never work around
  a problem silently; surface it.
- If a request conflicts with an ADR, stop and ask. Log deferred decisions as a
  new ADR in ARCHITECTURE.md.
- If something would add more than ~1 hour of work outside the current stage,
  say "This is out of scope for the current stage. Confirm before I proceed?"

## Execution rules
- Read-only inspection runs without asking: reading files, `git status`,
  `git log`, `git diff`, `git show`, grep/rg, ls.
- Everything else needs the exact command, the estimated time and data volume,
  and where it should run (agent session or the user's terminal), stated first,
  and then the user's explicit go-ahead. This covers lint, tests, scripts,
  docker, and anything that writes.
- After an approved step, stop and report. Never chain into the next step.
- Never run in the agent session: anything over 1,000 records, full-dataset
  runs, the full test suite, `docker build` / `docker compose up`. Print the
  command for the user's terminal and wait for the pasted output.
- Verify code with a `--limit 100` smoke test or equivalent.
- Stop immediately any long-running process started by mistake.

## Git
- Never commit to `main`; it stays deployable. One branch per feature or fix:
  `feature/<slug>` or `fix/<slug>`, from an up-to-date `main`
  (`git pull origin main` first).
- Small, focused commits with conventional prefixes: feat, fix, test, docs,
  refactor, chore. Commit only when the user asks or approves.
- Never push without the user's explicit approval.
- Stage specific files; do not run `git add -A` blindly. Never commit broken
  code (stash it).
- Never commit: `.env`, anything under `data/` or `models/`, `serving_bundle*/`
  and the sender lookup (both contain real sender addresses).
- Open the PR via the GitHub compare URL (`gh` is not installed). The user
  merges; never self-merge. After a merge, update SESSION_LOG.md and delete the
  merged branch.

## Code rules
- Python 3.12.3. Type hints on every function signature; docstrings on every
  class and public function.
- black (88), isort and ruff are enforced in CI. Check changed files with them
  (subject to the execution rules) before presenting code.
- Logging: the `logging` module, never `print()`. Inside Prefect flows and
  tasks use `get_run_logger()`.
- No hardcoded paths: `pathlib.Path` plus environment variables via
  python-dotenv. Commit only `.env.example`.
- No Jupyter notebooks in any pipeline path; `/notebooks` is for EDA only.
- Metrics: F1, precision and recall; never accuracy as the primary metric
  (imbalanced data).
- No raw data in git. `data/` is gitignored.

## Honesty in results and docs
- Never quote an F1 or other metric without its baseline (the leadership-only
  rule) and its confidence interval when one exists.
- README and other docs describe only what exists and was measured. Planned
  stages are marked NOT STARTED.
- Never present the reference model as an urgency predictor (ADR-013).

## MLflow and registry safety
- Every training run happens inside `with mlflow.start_run():`. No untracked
  runs, and never log a model outside a run.
- Log all hyperparameters, all metrics, the model artifact, feature importance
  and the confusion matrix. Call
  `mlflow.set_experiment("email-priority-classifier")` at the top of every
  train script.
- Call `configure_mlflow()` (`src/serving/registry.py`) before creating any
  MLflow client; MLflow's internal default clients otherwise ignore an
  explicit URI (ADR-014).
- Do not register, re-alias, delete or otherwise write to the model registry
  or `mlflow.db` without explicit approval. Read-only operations (export,
  serving checks) must leave `mlflow.db` byte-identical; verify its sha256
  when the task claims this.

## Prefect
- Flows in `src/flows/`, tasks in `src/tasks/`, decorated with `@flow` /
  `@task`. Every flow runs on its own: `python -m src.flows.<name>`.

## Docker and serving
- Every service has its own Dockerfile with a healthcheck. Compose is currently
  api-only; monitoring services return in their own stage.
- Never use `latest` tags; pin versions. Follow ADR-015 for the image and
  bundle design.

## Dependencies
- Setup follows README (venv + pip; uv is optional). Do not suggest
  `pip install uv` (PEP 668 on Ubuntu 24.04).
- Add a dependency to `pyproject.toml` in the right group. If the serving
  import closure changes, update `requirements-serving.in` and re-lock.

## Testing
- One test file per source module; every public function has at least one
  test; use fixtures for shared data. Write the test with the module.
- Tests must not need a live MLflow server or Docker (mock them). Tests that
  need the real local model are allowed only if they skip cleanly when the
  artifacts are absent.
- Tests use temporary MLflow stores and never point at the real `mlruns/`; use
  the `isolated_mlflow` helper (`tests/serving_helpers.py`).

## End of session
1. Add a new entry at the TOP of SESSION_LOG.md: what was done, what broke,
   decisions, "Next session: first thing to do".
2. Commit working changes per the Git rules (with approval). Do not commit
   broken code.
