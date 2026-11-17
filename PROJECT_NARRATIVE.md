# Project Narrative — Email Priority Engine

Recruiter and interview-prep account of the implemented pipeline and measured
results. Decision history is in [ARCHITECTURE.md](ARCHITECTURE.md); run evidence
and corrections are in [SESSION_LOG.md](SESSION_LOG.md).

## 1. The pipeline and the two ingestion failures

This project studies email-priority modeling on Enron through a local,
reproducible experiment pipeline. Enterprise inbox triage motivates the task,
but the measured target is a heuristic label, not independently judged urgency.
Ingestion, labeling, features, training, and evaluation are implemented.
Serving, live monitoring, ML deployment automation, and ONNX export are roadmap
items, not completed components.

The pipeline starts with the email archive and writes a labeled Parquet file.
The original full ingestion completed with **517,401 emails, zero dropped**,
in about **22 minutes**. Two failures changed its design:

- **OOM:** on the 5.6GB host, the first implementation held raw records,
  validated records, and a DataFrame in memory simultaneously. The process was
  killed. The replacement streams records into a pyarrow ParquetWriter in
  50,000-row batches, bounding the record buffer instead of materializing the
  full corpus three times.
- **CPython header parsing:** malformed Enron address headers triggered
  `TypeError: 'ValueTerminal' object does not support item assignment` during
  Python 3.12's lazy structured-header parsing. `_safe_header()` attempts parsing
  and falls back to the original header value through `raw_items()`; subject
  decoding also has a raw fallback. Regression tests cover malformed headers.

Prefect flows expose standalone `python -m src.flows.<name>` CLIs and task logs
(ADR-001). Pure processing functions are separated from orchestration and
MLflow/Evidently access. Full-data runs were executed in the user's WSL terminal.

## 2. Heuristic labels and the raw-to-clean data boundary

Enron supplies no priority ground truth. The rule in `src/labels/labeling.py`
(ADR-005) is:

> priority = 1 if the sender is in the leadership list OR the subject contains
> an urgency keyword.

Recovering that rule is measurable; demonstrating human urgency detection is
not possible from these labels alone. In particular, non-leadership positives
are keyword labels by construction.

| Data scope | Emails / rows | Labeled positives | Positive rate |
|---|---:|---:|---:|
| Raw ingested corpus | 517,401 | 3,526 | 0.68% |
| One row per exact raw `(from, subject, date)` key | 251,741 | 1,494 | 0.59% |

Deduplication keeps the first row per key before feature engineering. No priority
labels are recomputed. This is header-key deduplication, not a claim that email
bodies are byte-identical. `sender_frequency` consequently counts retained
emails. The current feature parquet is **251,741 × 13**: five metadata/label
columns plus eight tabular features; it contains no body.

The ADR-012 split groups raw sender with normalized subject: lowercase, strip
stacked leading RE:/FW:/FWD:/REW: markers, and collapse whitespace. Twenty
StratifiedGroupKFold folds with seed 42 are assigned 14/3/3:

| Split | Rows | Positives | Leadership positives |
|---|---:|---:|---:|
| Train | 176,219 | 1,045 | 175 |
| Validation | 37,761 | 224 | 60 |
| Test | 37,761 | 225 | 60 |

Read-only checks found zero group and UID overlap between splits. Training
derives and logs `scale_pos_weight = train_negatives / train_positives`:
**175,174 / 1,045 ≈ 167.630622**, about **167.6**. The ratio is not a fixed
corpus-wide constant.

## 3. Three leaks, each exposed by a diagnostic

### Leak 1 — keyword inputs reproduce the target

Comparing training variants exposed direct label leakage: `urgency_keyword_count`
and subject TF-IDF encode inputs used to create priority. The keyword-input
variant reproduced all held-out labels in the clean-split run: diagnostic
**F1=1.000, model interval not estimated**. For the same test split, the
leadership-only baseline has **F1=0.421 [0.205, 0.649]**. The perfect score is
retained only as label-leakage evidence, not an urgency-performance claim
(ADR-009). Fitting TF-IDF on train only does not remove this direct target leak.

### Leak 2 — leadership is the other half of the rule

The seven-feature `behavioral-only` variant removed keyword count and TF-IDF,
but retained `sender_is_leadership`. A crosstab showed all **732 raw leadership
rows** had priority=1. Inspection of the original artifact found leadership at
the root split, accounting for **27.4% of its gain**. Those are historical
diagnostics, not feature-importance claims about the current forest.

After deduplication there are **295 leadership positives**. At the clean
validation-selected threshold, the reference flags exactly the **60 leadership
emails** in both validation and test. Reference and leadership-only rule both
have test **F1=0.421 [0.205, 0.649]**; their paired difference is
**0.000 [0.000, 0.000]**. No non-leadership emails are flagged. Calling this
variant "clean" had confused removing one label channel with removing both
(ADR-010, superseded by ADR-013).

### Leak 3 — copies crossed the random split

An exact-key overlap diagnostic found that **76.8% of raw rows** shared
`(from, subject, date)` with another row. In the seed-42 random split,
**66.9% of validation rows** had an identical key in train. All earlier metrics
were contaminated as estimates of held-out generalization.

Exact deduplication alone was insufficient: random stratification still put
**28.3% of validation rows** alongside an exact sender/subject pair in train;
**59.7%** shared a normalized subject. This motivated the grouped split above.
The implemented provenance gate checks strategy/version, seed, dedupe policy,
per-split counts, sorted-UID membership fingerprint, and training class weight
before scoring. Missing provenance or a superseded source run causes an abort.

## 4. The honest result: ranking value, no classification gain over the rule

The reference artifact contains **1,886 trees**. Validation chooses the operating
threshold **0.9943359717873594**; test is never used to tune it.

| Test metric | Behavioral reference, 95% CI | Leadership-only baseline | Paired reference minus rule, 95% CI |
|---|---|---|---|
| F1 | 0.421 [0.205, 0.649] | 0.421 [0.205, 0.649] | 0.000 [0.000, 0.000] |
| All-row PR-AUC | 0.327 [0.188, 0.514] | 0.271; separate interval not logged | +0.056 [+0.013, +0.104] |
| Non-leadership PR-AUC | 0.032 [0.006, 0.087] | Prevalence benchmark: 0.0044 | Paired slice difference not estimated |

The baseline's F1 interval is identical because its predictions are identical
to the reference's in every resample. Model precision is **1.000 [1.000, 1.000]**
and recall **0.267 [0.114, 0.480]**. Perfect observed precision is structural:
leadership implies priority in the labels. It does not establish perfect urgency
recognition. The model gives **no classification gain over the rule** at this
threshold; heuristic labels cap what any model can demonstrate about urgency.

Intervals use **1,000 whole-group bootstrap resamples**, seed 42, over
**26,676 test groups**, with zero skipped draws. Model and baseline share draws;
the model and threshold remain fixed. Runtime was **1.65 seconds**. These
intervals cover test-cluster sampling, not training or threshold-selection
variability, and do not guarantee future-email performance.

A diagnostic retrained for the source run's **2,000 rounds**, without early
stopping or an eval set. Truncating it to 1,886 trees reproduced the artifact's
validation PR-AUC with **zero gap**, without a deterministic retry. Clean
non-leadership validation PR-AUC ranged **0.012-0.021**, flattening rather than
showing the old sustained climb. No paired tree-count comparison was performed;
the model was not replaced and PR-AUC-based training-v2 is not pursued.

Evidently compared a **50,000-row train reference** with the **37,761-row test
split** and found **0/7 drifted columns**. This is an offline baseline report,
not a live monitoring service, and absence of marginal drift does not resolve
label leakage.

ADR-013 retains behavioral-only as the **reference model for planned serving
and monitoring**, not a production urgency model. Variant 3, `no-label-inputs`,
is deferred and not built (ADR-011).

## 5. Corrections are part of the engineering evidence

The old pitch treated removing keywords as making the model honest, used a
fixed class weight, and omitted duplicate overlap. Those claims were corrected.
Seven MLflow runs carry `superseded=duplicate-leaky-split`; their artifacts
were preserved. The session log describes them as old runs tagged before the
documented clean training runs, but a later metadata audit found a discrepancy:
`bfcaf6e3d17246a38f40b77a652dfe7c` (keyword) and
`e4970b2cbe414ce3b1ec91b007d73cfe` (behavioral) record the same grouped-split
fingerprint, deduplication policy, split counts, and class weight as the
documented clean runs. Their duplicate-leaky supersession reason is therefore
unresolved, not established by the stored provenance. The exact tagging command,
its selected-run output, and an individual explanation for these runs were not
found in the inspected records. Existing tags are preserved as audit evidence;
matching metadata and metrics do not establish identical model artifacts.
ADR-013's evaluated reference remains `7189707b198f47748f89965712eee4bc`, with
baseline/CI evaluation `559ccf6356494eecabb332c47eff2105`.

The following are **audit-only legacy records**, not current performance to
quote. No valid clean-split baseline-and-interval comparison accompanies these
old F1 values; historical intervals were not estimated:

| Superseded record | Why it must not be quoted as current performance |
|---|---|
| F1 0.12 / ROC-AUC 0.85 | Duplicate-leaky random split; leadership still encoded the target; baseline/CI comparison unavailable for this legacy score. |
| Tuned threshold 0.9121 / test F1 0.3765 | Same contaminated data/split; no valid baseline-and-interval comparison for the legacy F1. |
| Non-leadership validation PR-AUC 0.643 at 1,000 trees | Duplicate-leaky curve, superseded by the grouped-split experiment. |

The pre-registered expectation of a sharply smaller, flatter non-leadership
curve was confirmed descriptively. The historical contrast is uncontrolled:
validation data, training data/sender frequencies, and class weight changed.
It is not a causal proof that duplicates alone produced the old climb.

## 6. Concepts I can explain with this experiment

- **Accuracy versus rare-class retrieval:** the retained corpus has 250,247
  negative labels and 1,494 positives. Flagging nothing gives about **99.41%
  accuracy** while recovering no positives. Precision measures correctness of
  flags; recall measures positive coverage; F1 balances them at an operating
  threshold. Here reference and rule share **F1=0.421 [0.205, 0.649]**, so
  quoting the score alone hides the absence of classification gain.
- **Train-only TF-IDF:** vocabulary and term weights are fit only on train,
  then transformed on validation/test and saved with the keyword variant.
  This avoids held-out vocabulary/frequency fitting; it does not prevent
  subjects from directly encoding a keyword-derived target. CSR matrices
  avoid dense TF-IDF materialization (ADR-008).
- **Class weighting:** `scale_pos_weight` changes the positive-class loss
  contribution. It comes from training labels only, about 167.6 here, and is
  logged per run. Weighted scores are uncalibrated; a threshold near 0.9943
  is not a 99.43% probability of real urgency.
- **PR-AUC versus ROC-AUC:** PR-AUC is reported as average precision and is
  the primary ranking metric. Its chance benchmark is prevalence; the
  non-leadership test benchmark is about 0.0044. ROC-AUC measures ordering
  against negatives but does not show how many flagged emails are correct
  at such low prevalence. Both ranking and a fixed-threshold rule comparison
  are needed.
- **Label leakage versus split leakage:** keyword/leadership inputs reveal
  how the target was constructed. Copies or related sender/subject groups
  across splits expose held-out content to training. Train-only preprocessing,
  input ablations, and group-overlap audits address different failure modes.
- **Remaining limitations:** sender frequency still uses all retained emails,
  including held-out ones; sender frequency and subject length may proxy the
  label inputs; same-subject overlap across different senders remains. There
  are only **164 validation / 165 test non-leadership positives**. A grouped
  split is not a claim of leakage-free human-urgency evaluation.

## 7. Implemented stack and traceability

Prefect orchestration, Parquet ingestion/features, LightGBM training, local
SQLite-backed MLflow, provenance-gated evaluation, diagnostic retraining,
Evidently's offline baseline, and group-aware uncertainty are implemented.
The latest full suite passed **263 tests** in WSL. MLflow and Evidently are
mocked in task tests except for the explicitly small real-Evidently regression.

The measured behavioral source run is `7189707b198f47748f89965712eee4bc`;
the baseline/CI evaluation is `559ccf6356494eecabb332c47eff2105`. The split
fingerprint is logged in both. Evaluation writes a new run, never overwrites
training metrics, logs full threshold/diagnostic tables as artifacts, and
computes the grouped split once per evaluation flow. Its CSV-only diagnostic
points do not become extra MLflow metrics.

LightGBM was chosen for tabular/sparse features and inspectable importances
(ADR-003); body NLP is excluded (ADR-004). Local MLflow records actual
experiments (ADR-002). Prediction logging (ADR-006) and ONNX export (ADR-007)
remain designs for unstarted stages. A final combined HTML evaluation summary
has not been built; the existing Evidently HTML is a drift baseline only.

## 8. Two-minute pitch

> I built a local email-priority experiment pipeline on Enron, from streaming
> ingestion through provenance-gated evaluation. The raw corpus has 517,401
> emails and 3,526 heuristic positives. Deduplication leaves 251,741 rows and
> 1,494 positives, and I use a sender/normalized-subject grouped split.
>
> The ingestion initially ran out of memory and crashed on a malformed header
> inside CPython. I fixed those with batched Parquet writing and parsing
> fallbacks. Then the modeling diagnostics exposed three leaks: keyword inputs
> reproduced the target, leadership reproduced its other half, and copies
> crossed the random split. I preserved and superseded the contaminated runs
> instead of presenting them as successful models.
>
> On the corrected split, the reference and leadership-only rule both have
> test F1 0.421 with a 95% cluster interval of 0.205 to 0.649; their paired
> difference is exactly zero. The model does improve ranking PR-AUC over the
> rule by 0.056, with a paired interval of 0.013 to 0.104, but flags no
> non-leadership emails at the selected threshold. Those labels cannot prove
> real urgency detection.
>
> The engineering result is reproducible splitting, artifact-length replay,
> baseline comparisons and cluster uncertainty, with 263 passing tests. MLflow
> uses local SQLite. Serving, live monitoring, ML deployment automation and
> ONNX export are not started. The saved model remains a documented reference,
> not a production urgency claim.

## 9. Eight questions to answer cold

### 1. What did you actually build?
Archive-to-Parquet ingestion, heuristic labeling, deduplicated feature
engineering, grouped LightGBM training with MLflow, and evaluation with split
provenance, diagnostic replay, a rule baseline, cluster CIs and an offline
Evidently report. Serving and live monitoring are not implemented.

### 2. What surprised you?
Removing keyword features did not make the remaining model label-clean.
The leadership crosstab and original root split exposed the other target
channel. Duplicate-overlap checks then invalidated the earlier random-split
interpretation. A high score required explaining, not celebrating.

### 3. Why doesn't your model beat a trivial rule?
At the val-chosen threshold it makes exactly the rule's test predictions:
both have **F1=0.421 [0.205, 0.649]**, paired difference **0 [0, 0]**.
Leadership is part of the label definition. Ranking improves by paired
PR-AUC **+0.056 [+0.013, +0.104]**, but the operating point adds no detections
outside leadership. That is a label/task limitation, not a production win.

### 4. How did you catch duplicate leakage?
I counted exact `(from, subject, date)` groups and joined validation keys
against train: 76.8% of raw rows had another copy, and 66.9% of validation
rows had a matching key in train. After deduplication I checked residual
sender/subject overlap, adopted grouped folds, and logged a membership hash
that evaluation must reproduce before scoring.

### 5. Why fit TF-IDF on train only, and why wasn't that enough?
Fitting on held-out text leaks vocabulary/frequency information. Train-only
fitting prevents that path. It cannot stop subject features from reproducing
labels explicitly derived from subject keywords. Preprocessing leakage and
target leakage are separate problems.

### 6. Why weight classes and prefer PR-AUC to accuracy or ROC-AUC?
Only 0.59% of retained labels are positive. The class weight is the training
negative/positive ratio, about 167.6, not a fixed raw-corpus number. PR-AUC
shows precision/recall ranking against a prevalence benchmark; accuracy is
dominated by negatives and ROC-AUC alone does not establish useful precision.
Class weighting also means scores should not be read as calibrated probabilities.

### 7. What do your confidence intervals establish?
They describe test-cluster resampling with this model and its val-selected
threshold fixed. The paired PR-AUC interval supports ranking value over the
rule; identical predictions explain the zero paired F1 difference interval.
They do not include training/threshold-selection variability or compare
tree counts. Precision's [1,1] comes from the heuristic label rule.

### 8. What would you do with real labels?
I would first define urgency with users and obtain independent annotations,
including non-leadership emails. I would compare leadership and keyword rules
against those labels, assess group/time generalization, then evaluate calibrated
scores and operating costs on held-out data. Those are proposed steps, not
completed work or a promise of improved performance.

## 10. Roadmap — not implemented

| Stage | Status | Planned scope |
|---|---|---|
| Serving | **NOT STARTED** | FastAPI/Docker for the documented reference model |
| Live monitoring | **NOT STARTED** | Prediction logging, ongoing drift, Prometheus/Grafana |
| CI/CD for ML | **NOT STARTED** | Automated retraining, registry promotion and deployment |
| ONNX export | **NOT STARTED** | Export and onnxruntime validation |

Variant 3 (`no-label-inputs`, ADR-011) is **deferred, not built**. Training-v2
with PR-AUC early stopping is not pursued (ADR-013). No saved-model replacement
or new tree-count selection follows from the current diagnostic.
