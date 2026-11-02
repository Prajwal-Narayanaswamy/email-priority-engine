# Email Priority Engine

A local MLOps experiment pipeline for studying email-priority heuristics on
Enron. Inbox triage motivates the problem; Enron has no human urgency labels.
The target is **leadership sender OR urgency keyword**, so evaluation measures
recovery of that rule, not validated real-world urgency.

## Data

| Scope | Rows | Labeled positives |
|---|---:|---:|
| Raw corpus | 517,401 | 3,526 (0.68%) |
| Exact-key deduplicated corpus | 251,741 | 1,494 (0.59%) |
| Grouped train / validation / test | 176,219 / 37,761 / 37,761 | 1,045 / 224 / 225 |

Deduplication keeps the first raw `(from, subject, date)` key before features.
Groups are `(from, normalized subject)`; seed-42 stratified group folds have
zero group overlap. Class weight is derived from training labels: **about 167.6**.
The evaluated reference has seven tabular inputs, including leadership, and
no keyword-count or TF-IDF inputs. Label-rule leakage and cross-sender subject
overlap remain disclosed limitations.

The corpus is public. The parity fixture and the sender lookup contain real
sender addresses from it and no message bodies.

## Built and verified

- Streaming ingestion with OOM and malformed-CPython-header fixes; heuristic labeling.
- Eight tabular features; sparse subject TF-IDF fitted only on train where used.
- Prefect feature/training/evaluation CLIs; LightGBM runs tracked in **MLflow on SQLite**.
- Evaluation provenance gate, validation-only threshold choice and diagnostic replay.
- Leadership-rule baseline, paired cluster-bootstrap intervals, offline Evidently baseline.
- **349 tests passed** in WSL (as of the Step 3 commit); Python formatting/lint/import checks passed.

## Five key findings

1. Keyword inputs reproduce the heuristic label rule; perfect label reproduction is leakage evidence.
2. On the test split, at the tuned threshold, the reference flags only the **60 leadership test emails**, exactly matching the rule.
3. **76.8% of raw rows had another exact-key copy; 66.9% of random-split validation rows had a matching key in train.** Earlier metrics were superseded; dedupe plus grouping fixes that overlap.
4. Paired PR-AUC improves over the rule, but **classification gain is zero**. The model remains a reference, not a production urgency predictor.
5. Group uncertainty matters: with 225 test positives the F1 interval is [0.205, 0.649], so no single F1 should be quoted. Evidently found **0/7** drifted columns; this is offline evaluation, not live monitoring.

## Clean-split test results

Threshold **0.994336**, selected on validation and fixed on test. Brackets are
95% percentile cluster-bootstrap intervals.

| Metric | Behavioral reference | Leadership-only rule | Paired reference minus rule |
|---|---|---|---|
| F1 | 0.421 [0.205, 0.649] | 0.421 [0.205, 0.649] | 0.000 [0.000, 0.000] |
| All-row PR-AUC | 0.327 [0.188, 0.514] | 0.271; separate CI not logged | +0.056 [+0.013, +0.104] |
| Non-leadership PR-AUC | 0.032 [0.006, 0.087] | Prevalence benchmark: 0.0044 | Not estimated |

The F1 interval is shared because model and rule predictions are identical on
the test split. No non-leadership emails are flagged there. Precision is
1.000 [1.000, 1.000] by the heuristic label construction; recall is
0.267 [0.114, 0.480]. Intervals condition on the saved model and fixed
val-selected threshold, not training or threshold-selection variability.
No paired tree-count comparison was performed.

Measured source run: `7189707b198f47748f89965712eee4bc`;
baseline/CI evaluation: `559ccf6356494eecabb332c47eff2105`.

## Setup

Python 3.12 in a virtual environment, which leaves the system Python alone
(stock Ubuntu 24.04 refuses `pip install` outside a venv, PEP 668):

```bash
sudo apt install python3-venv   # only if the next line reports ensurepip is missing
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

[uv](https://github.com/astral-sh/uv) is an optional, faster alternative
(`uv venv` then `uv pip install -e ".[dev]"`). Install it with `pipx install uv`
or astral's installer, not `pip install uv`. These commands have not been
verified end to end on a clean machine.

### Reproducibility

The reported numbers were produced with Python 3.12.3 and the library versions
in the saved model's own `requirements.txt`: mlflow 3.15.1, lightgbm 4.7.0,
numpy 2.5.2, pandas 2.3.3, scikit-learn 1.9.0, scipy 1.18.0 and skops 0.14.0.
The `pyproject.toml` dependencies are floors with no lock file, so other
versions may change last digits or `best_iteration`.

Measured on my 5.6 GB WSL2 machine (from `SESSION_LOG.md`): ingestion about 22
minutes with batched streaming (about 1.9 minutes per 50,000-row batch); the
grouped split about 3 minutes per call; the full test suite about 6 minutes
(361 s).

## Run

From the repo root with the Setup environment active.
Full-data commands belong in your WSL terminal. Data and the local MLflow store
are not bundled in Git, so the pipeline reproduces from the raw archive.

Get the Enron email archive (the May 7, 2015 CMU release,
`enron_mail_20150507.tar.gz`) from <https://www.cs.cmu.edu/~enron/> and place it
at `data/raw/enron_mail_20150507.tar.gz`. Then:

```bash
python -m src.flows.ingest_flow \
  --tar data/raw/enron_mail_20150507.tar.gz \
  --output data/processed/emails.parquet

python -m src.flows.feature_flow \
  --input data/processed/emails.parquet \
  --output data/features/features.parquet

MLFLOW_TRACKING_URI=sqlite:///mlruns/mlflow.db \
python -m src.flows.train_flow \
  --input data/features/features.parquet --seed 42
```

Use the **behavioral-only run ID printed by training** below, replacing
`YOUR_BEHAVIORAL_ONLY_RUN_ID`. Evaluation rejects superseded sources and
mismatched split provenance. It has no subsampling or diagnostic-round override.

```bash
MLFLOW_TRACKING_URI=sqlite:///mlruns/mlflow.db \
python -m src.flows.evaluate_flow \
  --run-id YOUR_BEHAVIORAL_ONLY_RUN_ID \
  --input data/features/features.parquet --seed 42 --reference-cap 50000

python -m pytest tests/ -q
```

## Serving

A FastAPI service (`/health`, `/predict`, `/model-info`) for the ADR-013
reference model. It is a demonstration of the serving path, not a production
urgency predictor. What exists (ADR-014):

- **Registry-backed model.** `MODEL_URI` defaults to
  `models:/email-priority@champion` (currently version 1). The alias is resolved
  once at startup and that exact version is loaded.
- **Threshold and provenance from version tags.** The tuned threshold, the
  lookup hash, the source and evaluation run IDs and the split fingerprint are
  read from the registered version's tags, not from environment variables.
- **Versioned sender lookup.** `sender_frequency` is a dataset-wide count, so
  requests use a lookup built from the training features. Its header records the
  source parquet's sha256, row count and build date, and startup fails if the
  file's sha256 differs from the registered tag.
- **Training-serving parity tests.** A committed 300-row fixture checks in CI
  that the serving feature path reproduces the stored features exactly; a local
  test checks 2,000 real rows and that API scores equal the model's
  `predict_proba`.
- **Fail-loud startup.** A missing registered model, alias or tag, a bad
  threshold, a stale lookup or a hash mismatch stops the service with a specific
  message. There is no fallback to loading by run ID.

Not built: Docker image, `/predict/batch`, `/metrics`, prediction logging.

### Commands

The run IDs the scripts default to belong to **my** registry; with your own
training and evaluation runs, pass `--source-run-id` and `--eval-run-id`.
`MLFLOW_TRACKING_URI` is optional and defaults to the repo's SQLite store.

```bash
python -m src.serving.build_lookup            # versioned sender lookup
python -m scripts.log_lookup_run              # log it to email-priority-serving
python -m scripts.register_reference_model    # dry run: prints the plan, writes nothing
python -m scripts.register_reference_model --apply
uvicorn src.serving.app:app --port 8000
```

Registration refuses to run twice: it stops if the source run already has a
version or the `champion` alias is already set.

### What the serving API does, and does not, guarantee

On the held-out test split, the flagged rows were exactly the leadership-sender
rows (60 of 60; test F1 about 0.42, the same as the leadership-only rule;
non-leadership F1 0). That is an empirical property of that split, not a
guarantee: a leadership sender's email can score below the threshold.

Illustration from the live API: `POST /predict` with

```json
{"from": "kenneth.lay@enron.com", "subject": "RE: budget?",
 "date": "Mon, 14 May 2001 16:39:00 -0700"}
```

returns `score` 0.5446273611788119 with `sender_frequency` 21,
`sender_is_leadership` 1, `hour_of_day` 23, `day_of_week` 0, `subject_length` 11,
`has_question_mark` 1 and `thread_depth` 1. The threshold is 0.9943, so `flagged`
is **false** even though the sender is on the leadership list. Omitting `date`
makes `hour_of_day` and `day_of_week` -1 and changes the score. The score is
uncalibrated (inflated by `scale_pos_weight`), not a probability, and
`/model-info` returns the full caveats.

## Roadmap

| Stage | Status |
|---|---|
| Serving API | **Built:** registry-backed, versioned lookup, parity-tested; no Docker, batch, `/metrics` or prediction logging yet |
| Docker image | **NEXT** |
| Live monitoring | **NOT STARTED** |
| CI/CD for ML: automated retraining/promotion/deployment | **NOT STARTED** |
| ONNX export and runtime validation | **NOT STARTED** |

Variant 3 (`no-label-inputs`) is deferred, not built. ADR-013 retains the
current model only as the reference for future serving/monitoring work.

## Evidence and interview context

- [Architecture decisions](ARCHITECTURE.md): ADR-001 through ADR-014, including superseded history.
- [Project narrative](PROJECT_NARRATIVE.md): diagnostics, corrections, two-minute pitch and eight interview questions.
- [Session log](SESSION_LOG.md): run IDs, execution results and stage checkpoints.
