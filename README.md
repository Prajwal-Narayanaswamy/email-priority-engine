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

## Built and verified

- Streaming ingestion with OOM and malformed-CPython-header fixes; heuristic labeling.
- Eight tabular features; sparse subject TF-IDF fitted only on train where used.
- Prefect feature/training/evaluation CLIs; LightGBM runs tracked in **MLflow on SQLite**.
- Evaluation provenance gate, validation-only threshold choice and diagnostic replay.
- Leadership-rule baseline, paired cluster-bootstrap intervals, offline Evidently baseline.
- **263 tests passed** in WSL; Python formatting/lint/import checks passed.

## Five key findings

1. Keyword inputs reproduce the synthetic label rule; perfect label reproduction is leakage evidence.
2. At the tuned threshold, the reference flags only the **60 leadership test emails**, exactly matching the rule.
3. **76.8% of raw rows had another exact-key copy; 66.9% of random-split validation rows had a matching key in train.** Earlier metrics were superseded; dedupe plus grouping fixes that overlap.
4. Paired PR-AUC improves over the rule, but **classification gain is zero**. The model remains a reference, not a production urgency predictor.
5. Group uncertainty matters: **1,000 resamples over 26,676 test groups** took **1.65s**. Evidently found **0/7** drifted columns; this is offline evaluation, not live monitoring.

## Clean-split test results

Threshold **0.994336**, selected on validation and fixed on test. Brackets are
95% percentile cluster-bootstrap intervals.

| Metric | Behavioral reference | Leadership-only rule | Paired reference minus rule |
|---|---|---|---|
| F1 | 0.421 [0.205, 0.649] | 0.421 [0.205, 0.649] | 0.000 [0.000, 0.000] |
| All-row PR-AUC | 0.327 [0.188, 0.514] | 0.271; separate CI not logged | +0.056 [+0.013, +0.104] |
| Non-leadership PR-AUC | 0.032 [0.006, 0.087] | Prevalence benchmark: 0.0044 | Not estimated |

The F1 interval is shared because model and rule predictions are identical.
No non-leadership emails are flagged at this threshold. Precision is
1.000 [1.000, 1.000] by the heuristic label construction; recall is
0.267 [0.114, 0.480]. Intervals condition on the saved model and fixed
val-selected threshold, not training or threshold-selection variability.
No paired tree-count comparison was performed.

Measured source run: `7189707b198f47748f89965712eee4bc`;
baseline/CI evaluation: `559ccf6356494eecabb332c47eff2105`.

## Run

From the repo root with the project's Python 3.12 environment active.
Full-data commands belong in your WSL terminal. Prerequisite: labeled ingestion
output at `data/processed/emails.parquet`; data and the local MLflow store are
not bundled in Git.

```bash
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

## Roadmap

| Stage | Status |
|---|---|
| Serving API | **NOT STARTED** |
| Live monitoring | **NOT STARTED** |
| CI/CD for ML: automated retraining/promotion/deployment | **NOT STARTED** |
| ONNX export and runtime validation | **NOT STARTED** |

Variant 3 (`no-label-inputs`) is deferred, not built. ADR-013 retains the
current model only as the reference for future serving/monitoring work.

## Evidence and interview context

- [Architecture decisions](ARCHITECTURE.md): ADR-001 through ADR-013, including superseded history.
- [Project narrative](PROJECT_NARRATIVE.md): diagnostics, corrections, two-minute pitch and eight interview questions.
- [Session log](SESSION_LOG.md): run IDs, execution results and stage checkpoints.
