# Email Intelligence Pipeline — OpenCode Context

## Project Summary
End-to-end MLOps pipeline for email priority classification using the Enron dataset.
Fully local — no cloud inference, no external APIs for prediction.
Target: MLOps Engineer roles. Demonstrates the full MLOps stack, not just modeling.

## Environment
- OS: Ubuntu 22.04 via WSL2 on Windows
- Python: 3.12.3 (use this, do not suggest upgrades)
- Package manager: uv (faster than pip — install if missing: `pip install uv`)
- Docker: 28.1.1 (available in WSL)
- Git: 2.43.0
- AI coding tool: OpenCode with Big Pickle (GLM-4.6 class model)
- Working directory: ~/code/project-a

## What This Project Demonstrates (for resume)
1. Data pipeline orchestration with Prefect
2. Experiment tracking and model registry with MLflow
3. NLP feature engineering on real messy email data
4. Model serving with FastAPI + Docker
5. Data drift monitoring with Evidently AI
6. CI/CD for ML with GitHub Actions (lint → test → retrain trigger → deploy)
7. ONNX model export (edge/on-device inference angle)

## Hard Rules — Never Violate These
- No Jupyter notebooks in any pipeline code path. Notebooks go in /notebooks for EDA only.
- No print() statements for logging. Use Python logging module everywhere.
- No hardcoded file paths. Use pathlib.Path + environment variables via python-dotenv.
- No accuracy as primary metric. Always use F1 + Precision + Recall (dataset is imbalanced).
- No untracked MLflow runs. Every training run must be logged. No exceptions.
- No raw data committed to git. All data/ directories are gitignored and DVC-tracked.
- No .env files committed. Only .env.example with placeholder values.
- Type hints on every function signature.
- Docstrings on every class and every public function.

## Code Style
- Formatter: black (line length 88)
- Import sorter: isort
- Linter: ruff
- Test framework: pytest
- All three run in GitHub Actions on every push

## How to Work With Me
- I have 3-5 hours per week. Sessions are short. Context switching is expensive.
- ALWAYS read SESSION_LOG.md before starting any work in a session.
- ALWAYS write a session summary to SESSION_LOG.md before ending.
- If I ask for something that conflicts with ARCHITECTURE.md decisions, flag it first.
- Explain what's wrong before fixing it. I need to understand, not just ship.
- When scope creeps, call it out. Ask me to confirm before expanding work.
- Prefer simple working code over complex clever code.

## Current Stage
See SESSION_LOG.md → "Next session — first thing to do"

## Stack Reference
| Concern | Tool | Why |
|---|---|---|
| Orchestration | Prefect 3.x | Simpler than Airflow for solo MLOps projects |
| Experiment tracking | MLflow 2.x | Industry standard, runs locally |
| Model serving | FastAPI + Uvicorn | Lightweight, async, production standard |
| Containerization | Docker + Compose | Required for reproducible deployment |
| Drift monitoring | Evidently AI | Purpose-built for ML monitoring |
| Observability | Prometheus + Grafana | Industry standard metrics stack |
| Data versioning | DVC | Git-compatible data versioning |
| CI/CD | GitHub Actions | Free for public repos |
| Model export | ONNX | On-device/edge inference compatibility |
| Data validation | Pydantic v2 | Schema enforcement at ingestion |
