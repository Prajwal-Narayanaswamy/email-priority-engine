# Architecture Decisions

## Problem Statement
Enterprise inboxes generate thousands of emails daily, burying time-sensitive
communications. This pipeline classifies email priority, extracts calendar-worthy
events, and monitors model performance over time — fully locally, with no data
leaving the machine.

## System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        DATA LAYER                               │
│                                                                 │
│  Enron .tar.gz → [Prefect: ingest_flow] → Parquet (processed) │
│                         ↓ Pydantic validation                   │
│               [Prefect: feature_flow] → Feature parquet         │
│                         ↓                                       │
│               DVC tracks all data artifacts                     │
└─────────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────────┐
│                      TRAINING LAYER                             │
│                                                                 │
│  [Prefect: train_flow]                                          │
│    → MLflow: log params, metrics, artifacts, model              │
│    → MLflow Model Registry: register best model                 │
│    → ONNX export of best model                                  │
│                                                                 │
│  [Prefect: evaluate_flow]                                       │
│    → F1, Precision, Recall, ROC-AUC, confusion matrix          │
│    → Evidently: baseline report on training data                │
│    → HTML evaluation report saved to /reports                   │
└─────────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────────┐
│                      SERVING LAYER                              │
│                                                                 │
│  FastAPI (Docker)                                               │
│    POST /predict        → single email prediction               │
│    POST /predict/batch  → batch prediction                      │
│    GET  /health         → liveness probe                        │
│    GET  /metrics        → Prometheus metrics                    │
│                                                                 │
│  Model loaded from MLflow registry at startup (not per-request) │
│  All predictions logged to SQLite for drift monitoring          │
└─────────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────────┐
│                    MONITORING LAYER                             │
│                                                                 │
│  Evidently (background service)                                 │
│    → Compares live prediction inputs vs training distribution   │
│    → Drift report every 500 predictions or 1 hour              │
│    → Alert threshold: drift in >20% of features                │
│                                                                 │
│  Prometheus scrapes /metrics every 15s                          │
│  Grafana dashboard: latency, volume, drift score, model version │
└─────────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────────┐
│                      CI/CD LAYER                                │
│                                                                 │
│  GitHub Actions on push to main:                                │
│    1. ruff lint + black format check                            │
│    2. pytest (unit + integration)                               │
│    3. If tests pass + drift alert exists → trigger retrain      │
│    4. If new model beats champion on F1 → promote in registry   │
│    5. Rebuild and push Docker image                             │
└─────────────────────────────────────────────────────────────────┘
```

## Decision Log

### ADR-001: Prefect over Airflow
Airflow requires a metadata DB + scheduler + webserver. Too much overhead for a
solo project on WSL. Prefect runs with zero infrastructure locally. Both are
orchestrators — the concepts transfer. Documented in README for interviewers.

### ADR-002: Local MLflow, not cloud
Zero cost. Sufficient for portfolio. Mitigate lack of shareability with screenshots
and exported HTML reports committed to /reports. MLflow UI screenshots in README.

### ADR-003: LightGBM as primary model, not neural
Faster iteration. Handles sparse TF-IDF features well. Interpretable feature
importance. DistilBERT is a stretch goal only after LightGBM baseline hits F1 ≥ 0.70.
Do not start neural models before baseline is solid.

### ADR-004: Subject-only NLP, not full email body
Enron body text is noisy — forwarded threads, signatures, legal disclaimers.
Subject line is a cleaner priority signal. Reduces dimensionality significantly.
Full-body NLP added only if subject-only F1 plateaus below 0.70.

### ADR-005: Heuristic labels (no ground truth)
Enron has no priority labels. Label definition:
  priority=1 if: sender is in known leadership list OR subject contains urgency keywords
  Urgency keywords: ["urgent", "asap", "immediately", "deadline", "action required",
                     "time sensitive", "eod", "eow", "by end of"]
  Source of truth: src/labels/labeling.py
  This is a documented limitation. Mentioned explicitly in README and reports.

### ADR-006: SQLite for prediction logging
Overkill would be Kafka or PostgreSQL. SQLite is sufficient for a local serving layer
logging predictions for drift input. Swap to PostgreSQL if serving goes to production.

### ADR-007: ONNX export
Demonstrates understanding of edge/on-device inference constraints. Relevant for
Apple ML roles and any embedded/mobile ML position. LightGBM → ONNX via
sklearn-onnx. Validated by running inference through onnxruntime.

### ADR-008: Feature-layer engineering rules
Decisions for the Features stage (src/features/features.py):
- Sender normalization (extract addr-spec from the From header) is applied in
  feature engineering ONLY. Labels per ADR-005 remain keyed to the raw `from`
  header — the 3,526 positives are not recomputed. sender_frequency and
  sender_is_leadership use the normalized address; labeling does not.
- TF-IDF fitting is deferred to train_flow. The features layer only exposes
  fit_subject_vectorizer() and transform_subjects(); the vectorizer is fit on
  the training split to avoid label/data leakage. It is NEVER fit on the full
  dataset. The feature parquet holds the 8 tabular features, not TF-IDF columns.
- Unparseable dates produce sentinel -1 for hour_of_day and day_of_week, not NaN.
- TF-IDF output is stored as scipy sparse matrices, never dense float32 columns
  (517,401 x 100 dense ≈ 200MB, risks OOM on a 5.6GB host). Documented in the
  transform_subjects() docstring.
- features.py follows the same pattern as ingest.py: pure functions with no
  Prefect dependency. Prefect wrappers live in src/tasks/ and src/flows/.

## Feature Hypotheses
Every feature requires a hypothesis before implementation. Validate in EDA notebook.

| Feature | Hypothesis | Status |
|---|---|---|
| sender_frequency | High-frequency senders send more important mail | Unvalidated |
| sender_is_leadership | C-suite email = high priority by definition | Unvalidated |
| hour_of_day | Off-hours email may signal urgency | Unvalidated |
| day_of_week | Monday/Friday emails may differ in urgency | Unvalidated |
| urgency_keyword_count | Direct lexical signal for priority | Unvalidated |
| subject_length | Very short subjects may be urgent (terse = urgent) | Unvalidated |
| has_question_mark | Questions in subject may require action | Unvalidated |
| thread_depth | Deep threads are follow-ups, lower priority | Unvalidated |
| tfidf_subject_top100 | Sparse NLP features for content signal | Unvalidated |

Update Status column in EDA notebook. Remove features that show no signal.
