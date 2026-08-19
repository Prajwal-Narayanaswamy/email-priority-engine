# Agent Instructions — OpenCode / Big Pickle

## Before Every Session
1. Read SESSION_LOG.md — find the latest entry, read "Next session — exact first action"
2. Read ARCHITECTURE.md — do not suggest changes to ADR decisions without flagging first
3. Confirm current stage before writing any code

## Coding Behavior
- Write complete, working files. Do not write placeholder functions with `pass`.
- When writing a new module, also write its test in /tests at the same time.
- Never suggest `pip install` — always suggest `uv pip install`.
- When adding a dependency, also add it to pyproject.toml [project.dependencies].
- Always run `black src/ tests/` and `ruff check src/ tests/` mentally before
  presenting code. Do not present code that would fail these checks.

## When Something Is Broken
- Diagnose first, fix second. Explain what the root cause is before writing any code.
- If the fix requires changing an ADR decision, stop and ask before proceeding.
- Do not silently work around a problem. Surface it.

## Scope Management
- If a request would add more than ~1 hour of work not in the current stage, flag it.
- Ask: "This is out of scope for current stage. Confirm before I proceed?"
- Log any deferred decisions in ARCHITECTURE.md under a new ADR.

## MLflow Rules
- Every model training call must use `with mlflow.start_run():` context manager.
- Log: all hyperparameters, all metrics, model artifact, feature importance, confusion matrix.
- Never log a model outside of a run context.
- Use `mlflow.set_experiment("email-priority-classifier")` at the top of every train script.

## Prefect Rules
- All flows in src/flows/. All tasks in src/tasks/.
- Every flow must be independently runnable: `python -m src.flows.flow_name`
- Flow functions decorated with @flow, task functions with @task.
- Use Prefect logging (get_run_logger()) not Python logging inside flow/task functions.

## Docker Rules
- Every service gets its own Dockerfile.
- docker-compose.yml at project root orchestrates all services.
- Never use `latest` as an image tag in docker-compose. Pin versions.
- Always include healthcheck in Dockerfile for services.

## Testing Rules
- One test file per source module (test_ingest.py for ingest.py, etc.)
- Minimum coverage: every public function has at least one test.
- Use pytest fixtures for reusable test data.
- No tests that require a live MLflow server or live Docker — mock those.

## Execution Rules
- Never run long-running commands inside OpenCode: anything processing more
  than 1000 records, or any full-dataset run. Print the exact command for the
  user to run in their WSL terminal, then wait for them to paste the output back.
- Always verify code inside OpenCode with a smoke test of `--limit 100` or
  equivalent. Full runs always happen in the user's terminal, never in OpenCode.
- Before ANY code execution, tell the user:
  (1) the exact command to run,
  (2) estimated time / data volume,
  (3) whether it should run in OpenCode or their terminal.
  Then wait for explicit go-ahead before running it.
- If a long-running process was started by mistake, stop it immediately.

## End of Session
Before stopping, always:
1. Update SESSION_LOG.md with what was done, what broke, what's next.
2. Run `git add -A && git commit -m "stage: brief description"` if code works.
3. Do not commit broken code. Stash it instead.
