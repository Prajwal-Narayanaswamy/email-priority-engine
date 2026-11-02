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

### ADR-009: Label leakage in the with-keyword-features run
`urgency_keyword_count` and subject TF-IDF directly encode the ADR-005 labeling
heuristic, so the model reads the same inputs the label was derived from. Measured
consequence: test F1 = 1.00 on 517,401 rows. The run reproduces the labeling rule
rather than predicting an unseen outcome, and F1 reaching 1.00 immediately (reported
best_iteration = 1) is the expected signature of that, not a training bug.

Path A, chosen deliberately: DOCUMENT the leakage, do not "fix" the features. The run
is retained in the MLflow experiment as a labelled leakage example so the leak stays
visible and measured, and the known-limitation tag is attached to the run. Removing the
leaking features would destroy the demonstration that the leak exists and was caught.
`behavioral-only` is the production candidate — see ADR-010.

Evidence note: the F1=1.00 result and ADR-005 together establish the leakage. The
`diag_early_stopping.py` throwaway diagnostic confirmed only that the custom early-
stopping metric is wired correctly (invoked every iteration, reported as the sole
metric `f1_pos`); it did not run to completion and is not the basis for this ADR.

### ADR-010: behavioral-only is keyword-clean, leadership-leaky (amended 2026-10-06)
**Status:** Superseded by ADR-013
See ADR-013: candidate designation superseded; behavioral-only is a serving/monitoring reference only.
See ADR-012: all earlier performance metrics are superseded by confirmed duplicate leakage.
`behavioral-only` (7 tabular features, no `urgency_keyword_count`, no subject TF-IDF)
is clean of the KEYWORD channel but NOT of the leadership channel: it retains
`sender_is_leadership`, which is a direct readout of half the ADR-005 label rule
(leadership OR keyword). It is therefore NOT a production candidate until the
leadership-free variant (ADR-011) is trained and evaluated. The original
"production candidate" designation (2026-09-29) is superseded by this amendment;
the variant is retained as a measured example of partial leakage, same Path-A
treatment as ADR-009.

Evidence (read-only checks on the full 517,401 rows and the saved artifact):
- `sender_is_leadership == 1` implies `priority == 1` in all 732 rows, zero
  exceptions; 732 + 2,794 = 3,526 positives, so ~21% of positives are separable
  by this one feature alone.
- The saved model's single tree splits on `sender_is_leadership` at its ROOT
  (threshold ~0, gain 3.1M = 27.4% of total gain), ahead of `sender_frequency`
  (13.2%). `subject_length` leads on gain (53.5%) — see the ADR-011 caveat.
- Consequence: the test F1 = 0.12 mixes genuine behavioural signal with
  leadership-channel recall and must not be quoted as an honest-model number.
  The evaluation report separates the two with leadership-slice metrics.

Measured on the held-out test split (77,611 rows): F1 = 0.12, ROC-AUC = 0.85.

Low F1 is EXPECTED and is not grounds for further model work. The labels are heuristic
(ADR-005) and behavioural signals alone cannot perfectly recover a keyword-derived
label: at 0.68% prevalence, "who emails me, when, and how deep is the thread" is a
weak proxy for "did this subject contain an urgency keyword". Reporting the F1 without
this context would misrepresent the model as broken.

ROC-AUC = 0.85 shows ranking signal, but that reading is now qualified: part of it
comes through the leadership channel (see evidence above), which the
leadership-slice metrics in the evaluation report separate out. The F1/ROC-AUC
gap still indicates the 0.5 hard threshold is poorly calibrated for this base
rate, so threshold selection and PR-AUC belong in the evaluation-stage report
before anyone calls this model weak.

RESOLVED 2026-10-06 (was OPEN): best_iteration = 1 is explained, not a bug.
The artifact's own `evals_result_` shows 101 rounds ran with val F1 peaking at
round 1 (0.115967, never beaten), so early stopping halted at 1+100 correctly;
then `lgb.train()` re-serialized through `model_to_string()` with
`num_iteration=None` (the default unless `keep_training_booster=True`), which
saves the best iteration only — verified empirically on LightGBM 4.7.0. The
artifact is one tree by construction of the pipeline. Whether later trees
would have added RANKING signal (F1@0.5 never improved, but PR-AUC might have)
is tested by the diagnostic retrain, not by relitigating early stopping.

### ADR-011: third variant "no-label-inputs" (planned, not trained)
See ADR-013: whether to pursue this variant remains a pending decision.
A third training variant excluding every direct label input: 6 features —
`sender_frequency`, `hour_of_day`, `day_of_week`, `subject_length`,
`has_question_mark`, `thread_depth` (drops `sender_is_leadership` AND
`urgency_keyword_count`; no TF-IDF). Rationale: ADR-010 shows behavioral-only
leaks through the leadership channel; this variant removes it, so its score
measures behavioural recovery of the heuristic without either direct channel.

Caveat, stated before training: `sender_frequency` and `subject_length` may
still proxy label inputs (frequent senders overlap leadership; subject length
may correlate with keyword-bearing subjects), so even this variant's results
measure recovery of a heuristic, not true urgency. Report them with that
framing, never as production-ready priority prediction.

Status: scoped only — no code, no training run. Training it is out of scope
for the evaluation stage; it is the defined next step if a production
candidate is still wanted after this evaluation lands.

### ADR-012: Exact-key deduplication and grouped split (approved, not implemented)
**Context:** Read-only checks confirmed that 76.8% of the 517,401 rows share
`(from, subject, date)` with another row (78.8% of positives). In the seed-42
random split, 66.9% of validation rows (71.1% of validation positives) had an
identical email in train. After exact deduplication plus a random stratified
split, 28.3% of validation rows (30.4% of positives) still share an exact
`(from, subject)` pair with train, and 59.7% (61.2% of positives) share a
normalized subject. Exact deduplication alone is therefore insufficient to
evaluate generalization to unseen sender/subject groups.

**Decision:**
- Deduplicate by exact `(from, subject, date)`, keeping the first row in input
  order, BEFORE `engineer_features`, so `sender_frequency` counts unique emails.
  `from` remains raw in these keys, matching the measured analysis. The
  sender-normalization check found 2,430 of 517,401 rows where raw `from`
  differs from `normalize_sender(from)`, but both keys retain 251,741 unique
  emails (difference 0) and 1,494 positives. The raw dedupe key is confirmed.
- Group key is `(from, normalized subject)`. Normalize subjects by lowercasing,
  repeatedly stripping leading `RE:`/`FW:`/`FWD:`/`REW:` markers, collapsing
  whitespace, and stripping surrounding whitespace. Missing subjects map to
  the empty string, matching the feasibility check.
- Split with `StratifiedGroupKFold(n_splits=20, shuffle=True, random_state=42)`:
  folds 0-13 train, 14-16 validation, 17-19 test. Training, evaluation, and
  diagnostic reconstruction must share the same split function and group policy.
  Do not select folds based on scores or positive counts.
- Extend the split-reproducibility gate to check strategy/version, seed,
  deduplication policy, row and positive counts, and a membership fingerprint.
  Abort on mismatch before tuning thresholds or interpreting diagnostic scores.
- Derive `scale_pos_weight` at runtime from the training labels as
  `train_negatives / train_positives`, and log it. This supersedes the hard-coded
  145.7: the measured new training split gives 175,174 / 1,045, about 167.6.

**Measured outcome:** Exact-key deduplication retained 251,741 rows and 1,494
positives, including 295 leadership positives. No key group has mixed labels.
The grouped split produced:

| Split | Rows | Positives | Leadership positives |
|---|---:|---:|---:|
| Train | 176,219 | 1,045 | 175 |
| Validation | 37,761 | 224 | 60 |
| Test | 37,761 | 225 | 60 |

Zero `(from, normalized subject)` group overlap between all pairs of splits.

**Consequences:** Every earlier training and evaluation metric was measured on
a duplicate-leaky split and is superseded: F1 0.12, ROC-AUC 0.85, tuned threshold
0.9121, test F1 0.3765, and the tree-count curve must not be quoted as current
performance or compared against the new split. Preserve historical runs and
artifacts, but tag old MLflow runs `superseded` with the duplicate-leakage reason
and legacy data/split provenance. Rerun deduplication -> features -> training ->
evaluation -> diagnostic -> Evidently before drawing model conclusions or
choosing a tree count. Validation and test are thin (224/225 positives;
164/165 non-leadership positives), so evaluation must report confidence intervals.

**Known limitations:**
- Same-subject overlap across different senders remains; grouping by sender
  and normalized subject does not guarantee subject-only disjointness.
- Empty-subject senders form large single groups; the largest has 1,291 rows.
- `sender_frequency` is computed over all unique emails, including held-out
  ones; that existing transductive leakage is retained and disclosed.
- Labels remain heuristic (ADR-005), not independently assessed true urgency.
  Non-leadership positives are keyword labels by construction.

**Out of scope:** Early stopping, feature definitions, model settings other
than the explicitly approved runtime class-weight derivation, and variants.
No training-v2 ADR, new variant, or model promotion in this pass. This ADR
records the approved design only; pipeline code and stored models are unchanged.

### ADR-013: Clean-split evaluation — behavioral-only is a reference, not a production candidate
**Context:** ADR-012's deduplication and grouped split are implemented and
verified. Clean behavioral-only training run `7189707b198f47748f89965712eee4bc`
was first evaluated in run `db09d4a9c99040b2aefd2d374982410c`; Step 2c run
`559ccf6356494eecabb332c47eff2105` adds the baseline and cluster intervals.
The split/provenance gate passed; the saved model contains 1,886 trees.
Full suite verified in the user's WSL terminal: 263 passed.

**Evidence:**
- At the validation-selected threshold (~0.9943), the model flags exactly
  the 60 leadership rows in BOTH validation and test, identical to the
  leadership-only rule. Test precision=1.0, recall=0.267, F1=0.421,
  flag rate=0.00159; validation F1=0.423. Non-leadership F1=0.
- Non-leadership test PR-AUC=0.032 against prevalence ~0.0044 (about 7x
  chance). Validation non-leadership probes range 0.012-0.021 against
  prevalence 0.00435 (about 3-5x chance): weak behavioral ranking signal,
  not useful non-leadership recall at the chosen operating point.
- All-row validation PR-AUC: 0.293 at 1 tree, 0.322 at 1,000, 0.318 at
  1,886. Validation ROC-AUC: 0.795 at 1 tree versus 0.786 at 1,886.
  The tree-count curve is flat/non-monotonic rather than showing sustained
  gains; small differences are treated as within noise, not as established
  improvements. No paired tree-count comparison has been performed; Step 2c's
  intervals compare the saved model against the rule, not different tree counts.
- A 2,000-round diagnostic without early stopping reproduced the artifact's
  validation PR-AUC at 1,886 trees with gap ZERO; no deterministic retry.
  Evidently reported 0/7 drifted columns with a 50k train reference versus test.
- PRE-REGISTERED expectation: CONFIRMED descriptively. Non-leadership PR-AUC
  at 1,000 trees fell from 0.643 on the superseded leaky split to ~0.021 on
  the clean split, and the clean tree-count curve is flat. This historical
  contrast is NOT a controlled comparison: validation set and scale_pos_weight
  also changed (along with deduplicated training data/sender frequencies).
  It does not prove duplicates alone caused the earlier climb. Leaky-run
  scores remain superseded, not comparable current-performance benchmarks.

**Decision:**
- behavioral-only is NOT a production candidate for urgency prediction.
  ADR-010's original candidate designation is superseded. It stays the
  reference model for the serving and monitoring stages, explicitly documented
  as reproducing the leadership half of the label rule plus weak behavioral
  signal. Reference status does not establish genuine urgency prediction.
- F1 ~0.42 must NEVER be quoted without the leadership-only-rule baseline,
  its identical predictions at this threshold, and non-leadership F1=0.
- Training-v2 with early stopping on PR-AUC is NOT pursued: there is no
  convincing evidence that more trees help. Do not select a new tree count,
  replace the saved model, or change early stopping based on this curve.
- Variant 3 (ADR-011, no-label-inputs): decision pending; no implementation
  or training authorized by this ADR.

**Step 2c uncertainty results:** Test-only 95% percentile cluster bootstrap,
1,000 resamples, seed 42, 26,676 `(from, normalized subject)` groups; zero
skipped resamples for all metrics. Threshold fixed at 0.9943359717873594.

| Test metric | Point estimate | 95% cluster interval |
|---|---:|---|
| Precision | 1.000 | [1.000, 1.000] |
| Recall | 0.267 | [0.114, 0.480] |
| F1 | 0.421 | [0.205, 0.649] |
| All-row PR-AUC | 0.327 | [0.188, 0.514] |
| Non-leadership PR-AUC | 0.032 | [0.006, 0.087] |

Non-leadership prevalence is ~0.0044. Leadership-only baseline: F1=0.421,
PR-AUC=0.271. Paired model-minus-baseline F1=0.000 [0.000, 0.000]
(identical predictions); PR-AUC=+0.056 [+0.013, +0.104]. Precision=1 with
interval [1,1] is by construction for these leadership-only predictions,
since leadership implies priority in the heuristic labels, not evidence of
perfect urgency recognition. The positive paired PR-AUC interval supports
ranking value over this rule under the specified resampling procedure.

Ranking value over the rule is supported, but no usable operating point exists
past the leadership rows. At the chosen threshold non-leadership recall/F1
remain zero; ranking improvement does not add classification gain here.

**Consequences:** Serving and monitoring may demonstrate the MLOps stack
using this documented reference model; they must preserve its limitations
and label-rule baseline. This decision does not itself implement those stages.

**Known limitations:**
- Only about 164 non-leadership positives per held-out split (164 validation,
  165 test); the measured recall/F1 and non-leadership AP intervals are wide.
- Intervals condition on the saved model and fixed validation-selected
  threshold. They do not cover training or threshold-selection variability,
  and precision [1,1] is not a future-email precision guarantee.
- No paired tree-count comparison was done. "Within noise" remains a working
  interpretation of the validation curve, not a significance test or proof
  that tree counts are equivalent; these model-vs-rule CIs do not change that.
- Rows within sender/subject groups are correlated; the measured intervals
  resample whole groups rather than treating individual emails as independent.
- Heuristic labels, leadership-label leakage, potential sender-frequency and
  subject-length proxies (ADR-011), cross-sender subject overlap, and frequency
  computed using held-out emails remain as disclosed in ADR-012.

### ADR-014: Serving contract and decisions (approved and implemented; registered as `email-priority` version 1 with alias `champion`)
**Context:** A minimal API exists (`GET /health`, `POST /predict`,
`GET /model-info` in `src/serving/`). It serves the ADR-013 reference model
and departs from the architecture diagram above:
- The model is loaded by training run ID via a LoggedModel search, not from the
  MLflow registry (no `register_model` exists in the code).
- No prediction logging (ADR-006 SQLite), no `/metrics`, no `/predict/batch`,
  no Docker image, no ONNX export.
- The threshold is read from the evaluation run's `tuned_threshold` param, and
  the `sender_frequency` lookup is an unversioned, untracked JSON file.
- Earlier text calling behavioral-only "the production candidate" (ADR-010
  and the line pointing to it) is left as history; ADR-010 is superseded by
  ADR-013.

**Decisions:**
1. **Training-serving parity.** Serving reuses the `src/features/features.py`
   functions for six of the seven features (confirmed in `build_feature_row`).
   A parity test on real rows from `data/features/features.parquet` guards that
   the serving path reproduces the stored offline features exactly.
2. **Versioned sender_frequency lookup.** The lookup artifact embeds the
   parquet sha256, row count and build date. Startup fails on a hash mismatch.
   It is logged to a NEW MLflow run (`serving-artifacts`) carrying
   `source_run_id`; it is never written to the finished training run.
3. **Registry with alias.** The reference model is registered once as
   `email-priority` with alias `champion`. This is documented as reference-only
   per ADR-013; the future CI/CD stage promotes by moving the alias. The
   registered version carries tags `tuned_threshold`, `threshold_source_run`,
   `lookup_sha256`, `lookup_run_id`, `split_fingerprint` and `source_run_id`.
   Serving loads `models:/email-priority@champion` through a `MODEL_URI`
   environment variable, with NO automatic fallback to run-ID loading: a
   missing alias fails startup.
4. **Score, not probability.** The response field `probability` is renamed
   `score` and documented as an uncalibrated score inflated by
   `scale_pos_weight`. The caveats state that, on the held-out test split, the
   flagged rows were exactly the leadership-sender rows (60 of 60; test F1 about
   0.42, same as the leadership-only rule; non-leadership F1 0), and that this is
   an empirical property of that split, not a guarantee. The full caveats are
   returned only by `/model-info`; `/predict` returns a one-line `note`.
   (Corrected 2026-11-01: an earlier draft said the flag "equals the
   leadership-only rule" at the tuned threshold, which a live request disproved:
   `kenneth.lay@enron.com`, `RE: budget?` scored 0.5446 against a 0.9943
   threshold and was not flagged.)
5. **Fail loudly.** Startup fails if the model, threshold or lookup is missing
   or the lookup hash mismatches. No degraded or partial serving mode.

**Known limitations:**
- Unseen senders default to `sender_frequency` 1, and a known sender's stored
  count excludes the incoming email, whereas training counts included the email
  itself. Serving is therefore off by one relative to training for new emails
  from known senders. A seen-once sender and an unseen sender look identical.
- A parity test on historical rows cannot detect this, because those rows are
  already counted in the lookup.
- `sender_frequency` was computed over all unique emails including held-out
  ones (ADR-012), and the lookup is a static training snapshot.
- On the held-out test split the reference model's flags coincided with the
  leadership-only rule (ADR-013), but that is not a property of arbitrary
  inputs: a leadership sender's email can score below the threshold. Registry
  promotion does not make it a production model.
- MLflow internals create default clients (for example the registry store's
  `create_model_version` builds `MlflowClient()`), which ignore a client's
  explicit URI and read the global tracking and registry URIs. With
  `MLFLOW_TRACKING_URI` unset the default is the `./mlruns` file store, which
  MLflow 3.x refuses. Every entry point (serving startup, registration and
  lookup-logging scripts) therefore calls `configure_mlflow(config.TRACKING_URI)`
  before creating any client, and the Docker container's environment must set
  `MLFLOW_TRACKING_URI` (or run the same entry points). Found when the first
  `--apply` failed and left an empty registered model, which a re-run now reuses.
- Registered versions point at an absolute WSL artifact path
  (`/home/<user>/code/project-a/mlruns/<exp>/models/<model_id>/artifacts`,
  recorded in the tracking DB). A container or another machine cannot load them
  as-is. How the model reaches a container (identical-path mount, a baked
  bundle, or an artifact-serving MLflow server) is a Docker-stage decision to
  be recorded in its own ADR.
- `lookup_sha256` is the sha256 of the lookup FILE bytes; the file's header holds
  the sha256 of the source PARQUET. Registration verifies the header against the
  current parquet, and startup verifies the file against the tag.
- Registration uses the MLflow client route (`create_model_version` with a
  `models:/<logged-model-id>` source), verified on a temporary DB to leave the
  finished LoggedModel's tags untouched; `mlflow.register_model` would add an
  `mlflow.modelVersions` tag to it.

**Out of scope for this pass:** Docker, `/predict/batch`, `/metrics`, SQLite
prediction logging and ONNX. Each needs its own approval.

### ADR-015: Serving image and model bundle (approved; D1 bundle code implemented, Dockerfile and compose not yet rewritten)
**Context:** The existing `docker/api/Dockerfile` and `docker-compose.yml` predate
the serving code: the `CMD` names a module that does not exist
(`src.serving.main:app`), `HEALTHCHECK` uses `curl` (absent in
`python:3.12-slim`), `pip install -e .` runs before the source is copied and
installs the whole project dependency list, there is no `.dockerignore` and no
non-root user, and the compose file pins an MLflow 2.x server against the
project's 3.15.1 store and wires monitoring services that have nothing to scrape.
ADR-014 also recorded that registered versions point at absolute WSL artifact
paths, so a container cannot load them from the registry as-is.

**Decisions:**
1. **Model bundle.** `scripts/export_model_bundle.py` resolves
   `email-priority@champion` (read-only against the registry) and writes a
   directory (default `./serving_bundle`, gitignored): `model/MLmodel`,
   `model/model.skops`, `sender_frequency.json`, `metadata.json` (model name,
   version, alias, threshold, all tags, sha256 of every payload file, library
   versions of the exporting environment, the model's own `requirements.txt`,
   export time) and `manifest.json` (sha256 of all four files). Export refuses if
   the lookup's sha256 differs from the registered `lookup_sha256` tag, and
   refuses to overwrite an existing bundle without `--force`.
2. **Verification, no fallback.** `MODEL_SOURCE` is `registry` or `bundle`, with
   no default; unset or unknown fails startup, and a failing source never falls
   back to the other. The bundle loader verifies that the manifest lists exactly
   the four files and that every hash matches, that `metadata.json` agrees with
   the manifest, that the tags are complete and the threshold valid (same checks
   as the registry source), and the app re-checks the lookup hash against the
   `lookup_sha256` tag. Both sources return the same `ServingMetadata`. The
   manifest detects corruption and accidental edits; it is not a signature, so
   whoever can rewrite the bundle can rewrite the manifest too.
3. **mlflow is dropped from the image.** In bundle mode the model is loaded with
   `skops.io.load`, trusting exactly the types in the bundle's `MLmodel`
   (`skops_trusted_types`), which is the call `mlflow.sklearn.load_model` makes.
   The condition was a parity test: the same bundle loaded both ways must give
   identical `predict_proba` (rtol 0, atol 1e-12). It holds on the 300-row
   fixture and on 2,000 real rows for the real model (local-only test), and a
   served Lay request returns the same score, 0.5446273611788119, as registry
   mode. A subprocess test asserts that importing the serving app imports neither
   mlflow, matplotlib, `src.training.train`, prefect nor evidently.
   `BEHAVIORAL_FEATURES` moved to `src/features/columns.py` (training re-exports
   it) so serving does not import the training module. Registry mode keeps
   mlflow and stays the local-development path.
4. **The image is generic.** It contains code and pinned dependencies only; the
   bundle is mounted read-only at run time (`MODEL_BUNDLE_DIR`). Alternative
   rejected: baking the model into the image. CI cannot reach the registry or
   the artifacts (they are local and gitignored), so a baked image could not be
   built or tested there, and every model change would force a rebuild.
5. **Dependencies.** `pyproject.toml` declares `skops` (previously only a
   transitive pin of full mlflow) and a `serving` extra with only the API's
   runtime needs. The image installs a hashed lock compiled from that extra, with
   versions matching the model's own `requirements.txt` (lightgbm 4.7.0, numpy
   2.5.2, pandas 2.3.3, scikit-learn 1.9.0, scipy 1.18.0, skops 0.14.0) and a
   pinned base image tag. The existing project dependencies are floors with no
   lock and are unchanged.
6. **Container hygiene.** Non-root user, `PYTHONUNBUFFERED`,
   `PYTHONDONTWRITEBYTECODE`, a strict `.dockerignore` (`.venv`, `data`, `mlruns`,
   `.git`, `.env`, logs, bundles), and a stdlib `HEALTHCHECK` (`urllib.request`
   against `/health`; startup blocks until the model loads, so a passing check
   means the model is ready).
7. **Compose scope: the `api` service only,** with the bundle mounted read-only,
   `MODEL_SOURCE=bundle`, a read-only root filesystem and a tmpfs for `/tmp`.

**Out of scope:** MLflow server, Prometheus and Grafana services, `/metrics`,
prediction logging, `/predict/batch`, ONNX, and registry mode inside the
container.

**Known limitations:**
- A bundle is a snapshot of one version: rolling back means re-exporting a
  different version, and nothing in the container follows later alias moves.
- The bundle contains the 20,312-sender lookup (real Enron addresses), so
  bundles and any image or volume holding one stay local and private.
- Library versions are pinned to what the model was logged with; the bundle
  loader does not yet compare runtime versions with the recorded ones.
- Dockerfile, compose, lock file and the CI docker job are still the old,
  broken versions until step D2.

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
