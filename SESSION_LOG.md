# Session Log — Email Intelligence Pipeline

## How to Use This File
Read the LATEST entry at the TOP before touching any code.
Add a new entry at the TOP when you finish a session.
Be specific. "Worked on features" is useless. "Ran feature_flow.py, got KeyError on
sender column, traced to ingest output missing sender_email field" is useful.
---

### Session: 2026-09-29 (Training close-out + Evaluation planning)
**Duration:** —
**Stage:** Training (closed) → Evaluation (planned, not started)

**What I completed:**
- TRAINING STAGE CLOSED. 133 tests pass (62 new across test_train.py 38,
  test_train_tasks.py 19, test_train_flow.py 5; 71 pre-existing). black + ruff + isort
  clean. Two variants trained on the full 517,401 rows and logged to MLflow as separate
  runs: `with-keyword-features` (test F1 = 1.00) and `behavioral-only` (test F1 = 0.12,
  ROC-AUC = 0.85). Split 362,180 / 77,610 / 77,610, identical seed for both variants.
- LABEL LEAKAGE CONFIRMED AND DOCUMENTED. The F1 = 1.00 is `with-keyword-features`
  re-deriving the ADR-005 heuristic from `urgency_keyword_count` + subject TF-IDF, not
  predicting anything. `behavioral-only` is the honest model and the production
  candidate. Path A chosen: document the leak, do NOT "fix" the features.
- Added ADR-009 (label leakage in with-keyword-features) and ADR-010 (behavioral-only is
  the production model candidate) to ARCHITECTURE.md.
- MLflow config fixes: absolute default tracking URI anchored to the repo root via
  `_REPO_ROOT = Path(__file__).resolve().parents[2]` (a relative sqlite:// URI resolved
  against the CWD and silently created a second invisible DB), and `load_dotenv()` in
  train_flow.py placed ABOVE the train_tasks import because train_tasks reads os.environ
  at import time.
- Evaluation stage scoped and APPROVED. No code written for it yet.

**Approved decisions for the evaluation stage (decide now, do not relitigate):**
- NEW MLflow run for evaluation (`run_name="evaluation"`), NOT a mutation of the
  training run. Writing threshold/F1/PR-AUC into the source run would overwrite the
  F1=1.00 vs 0.12 training record irreversibly. Source `run_id` goes in as a param AND a
  tag, nested via `mlflow.parentRunId`.
- Evidently: `reference=train` (capped at 50,000 rows) vs `current=test`. A baseline on
  training data ALONE says nothing — the drift comparison is the point. The 50k cap is
  deliberate (5.6GB host OOM'd on a full run during EDA) and must be disclosed in the
  report rather than hidden.
- PR-AUC IS THE PRIMARY METRIC, not ROC-AUC. At 0.68% prevalence ROC-AUC 0.85 badly
  overstates the model. The HTML report leads with PR-AUC and precision at the operating
  point, and explains why.
- Split reproducibility safety gate: `stratified_split(seed=42)` should reproduce
  training's val/test split, but VERIFY rather than assume — compare the reconstructed
  `val_rows`/`val_positives`/`test_rows`/`test_positives` against the values logged in
  the training run (train_tasks.py) and ABORT LOUDLY on mismatch. Silent split drift
  would invalidate every number in the report.
- Evaluation reads the FULL 517,401 rows with NO --limit. A subsample produces a
  different split, and a threshold tuned on the wrong val set is meaningless.

**Corrections made to the stage plan before approval (three real errors caught):**
1. "load model from MLflow by run_id" does NOT belong in src/evaluation/evaluate.py.
   train_tasks.py is the ONLY module that touches MLflow, and ADR-008 fixes the
   pattern: pure functions in src/<stage>/, MLflow/IO in src/tasks/. A new
   src/tasks/evaluate_tasks.py is required.
2. behavioral-only has NO vectorizer artifact. train_tasks.py logs `subject_vectorizer`
   only `if vectorizer is not None`, and behavioral-only sets use_tfidf=False so it was
   never fit. "Load the fitted vectorizer" would have raised on the exact run we are
   evaluating. Dropped; evaluation reads only `priority` + the 7 features, no `subject`.
3. diag_early_stopping.py was NOT given a --variant flag. Per the user, the
   behavioral-only best_iteration=1 question is to be settled by threshold tuning and
   PR-AUC analysis in the evaluation stage, not by more instrumentation.

**What is broken or blocked:**
- Nothing blocked. Session closing on a clean note.
- UNRESOLVED CARRIED FORWARD: `behavioral-only` reported best_iteration = 1 on the full
  run, and label leakage CANNOT explain it (that variant excludes the leaking features
  by construction). The threshold sweep in the evaluation stage is the agreed test: if
  the model was halted after one boosting round it is under-trained, and F1 = 0.12 /
  ROC-AUC = 0.85 are a floor rather than final. Do not write the report as if these
  numbers are settled.
- Still true, unaddressed: docker-compose.yml and .github/workflows/ci.yml export
  MLFLOW_TRACKING_URI pointing at a server on :5000, so training/evaluation under CI
  targets that server rather than a local store. No CI changes made, deliberate.
- Still true: `python -c` counts as "interactive" to python-dotenv and searches CWD, so
  the load_dotenv() fix only behaves correctly for `python -m`, not for `python -c`.

**Decisions made this session:**
- Path A for the confirmed leakage: document, don't fix. See ADR-009.
- `behavioral-only` is the production candidate; it goes to serving. See ADR-010.
- The five evaluation-stage decisions listed above are APPROVED and frozen for next
  session.

**Next session — exact first action:**
> Send the approved evaluation stage prompt to OpenCode and implement, in this order:
>   1. src/evaluation/evaluate.py — PURE functions only (no MLflow, no Prefect, no
>      Evidently): sweep_thresholds (predict once, sweep in memory, never re-predict per
>      threshold), metrics_at_threshold, pr_auc, roc_auc, plot_pr_curve, plot_roc_curve
>      (matplotlib Agg backend for WSL, per ADR-008), threshold_frame.
>   2. src/tasks/evaluate_tasks.py — the ONLY MLflow/Evidently contact:
>      load_model_task(run_id), load_split_task, evaluate_variant_task,
>      evidently_baseline_task.
>   3. src/flows/evaluate_flow.py — @flow + click CLI, --run-id required, logs the
>      tracking URI on startup, standalone `python -m src.flows.evaluate_flow`.
>   4. tests/test_evaluate.py — pure functions only; MLflow and Evidently mocked per the
>      testing rules. No live servers.
>   5. reports/evaluation/ HTML summary leading with PR-AUC.
> Respect the approved decisions above. Wire the split-reproducibility gate so it runs
> BEFORE any threshold is tuned. The full run command goes in the user's WSL terminal,
> never in OpenCode.

---
### Session: 2026-09-28 (Training)
**Duration:** —
**Stage:** Training

**What I completed:**
- Implemented src/training/train.py — pure functions, no Prefect, no MLflow (same
  pattern as ingest.py/features.py, ADR-008). stratified_split (70/15/15, stratified
  on `priority`, one seed shared by both splits so both variants see an identical
  split), build_matrices (scipy CSR, never dense), build_model, train_model,
  evaluate, feature_importance_frame (gain-based), plot_feature_importance (Agg
  backend for WSL), variant_by_name. SCALE_POS_WEIGHT = 145.7 is a module constant
  locked by a test, not computed at runtime.
- Implemented src/tasks/train_tasks.py — the ONLY module that touches MLflow.
  load_features_task (reads just subject/priority/8 features so uid/from/date never
  materialise) + train_variant_task (splits, fits vectorizer on TRAIN ONLY per
  ADR-008, trains, scores, logs). Every run inside `with mlflow.start_run():`.
- Implemented src/flows/train_flow.py — @flow + click CLI, runs both variants and
  logs a side-by-side comparison. `python -m src.flows.train_flow` standalone.
- TWO variants, per plan: `with-keyword-features` (all 8 features + subject TF-IDF,
  108 cols) and `behavioral-only` (7 features, no urgency_keyword_count, no TF-IDF,
  7 cols). Each is its own MLflow run. LIMITATION_NOTE is tagged onto BOTH runs
  describing the ADR-005 label leakage and the sender_frequency leakage.
- Logged per run: all params (incl. scale_pos_weight, per-split rows/positives/rate,
  best_iteration, n_matrix_columns, lib version), class-1 F1/P/R separately for val
  AND test, ROC-AUC, accuracy, all four confusion-matrix cells, model artifact,
  fitted vectorizer as a SEPARATE artifact (serving needs it), feature-importance PNG.
- Tests: 62 new across test_train.py (38), test_train_tasks.py (19, MLflow fully
  mocked), test_train_flow.py (5, prefect_test_harness + MLflow mocked). 133 total
  pass. black + ruff + isort clean. Added a shared features_parquet fixture to
  conftest.py instead of duplicating it per test file.
- Smoke-tested with --limit 500 inside OpenCode. Verified end to end against real
  MLflow: 2 FINISHED runs, params/metrics/tags correct, both feature-importance PNGs
  on disk, and all 3 model artifacts load back (LGBMClassifier + TfidfVectorizer,
  vectorizer.transform returns (1, 100)). Smoke runs deleted afterwards so the first
  real run is clean.

**Tracking backend migrated to SQLite (mid-session, after the code above was written):**
- train_tasks.py now reads `TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI",
  "sqlite:///mlruns/mlflow.db")` and calls `mlflow.set_tracking_uri(TRACKING_URI)`
  directly with the URI string, instead of building a file:// URI from a Path.
  Rationale: bug 2 above — MLflow 3.x considers the filesystem store maintenance-mode,
  so SQLite is the supported default and needs no MLFLOW_ALLOW_FILE_STORE opt-in.
- Removed the MLFLOW_ALLOW_FILE_STORE setdefault (dead code once the default is
  SQLite). .env.example now documents MLFLOW_TRACKING_URI=sqlite:///mlruns/mlflow.db
  and leaves MLFLOW_ALLOW_FILE_STORE commented out, only relevant if the URI is ever
  pointed back at a file store.
- Renamed the constant TRACKING_DIR -> TRACKING_URI and updated the one consumer,
  train_flow.py's startup log line (`TRACKING_DIR.resolve()` would have raised
  AttributeError on a str).
- Updated test_train_variant_task_uses_local_tracking_dir -> ..._uses_local_tracking_uri,
  and added test_tracking_uri_reads_environment_override (reloads the module with the
  env var set/unset). 130 tests pass. black/ruff/isort clean.
- Verified both URI forms in a scratch dir (NOT the real mlruns/): the relative default
  `sqlite:///mlruns/mlflow.db` and an absolute `sqlite:////abs/path`. Artifact location
  is CWD-relative and separate from the DB: DB at mlruns/mlflow.db, artifacts under
  mlruns/<experiment_id>/.
- CAVEAT: the relative default resolves against the CWD, so the flow must be launched
  from the repo root or the DB lands somewhere else. A fixed absolute path would remove
  that footgun; not done because it needs a decision on where the DB should live.
- CAVEAT: nothing in src/ calls load_dotenv(). Despite CLAUDE.md, .env is currently
  documentation only — MLFLOW_TRACKING_URI set in a .env file has NO effect unless it is
  exported in the shell. docker-compose.yml and .github/workflows/ci.yml already export
  MLFLOW_TRACKING_URI (pointing at a server on :5000), so training under CI would now
  target that server rather than a local store. Left alone, no CI changes made.

**FULL RUN COMPLETED (517,401 rows, user's WSL terminal) — both variants:**
- Split: 362,180 train / 77,610 val / 77,610 test, identical seed for both variants.
- `with-keyword-features`: test F1 = 1.00. This is LABEL LEAKAGE, not a good model.
  The variant reads `urgency_keyword_count` and subject TF-IDF, which are the exact
  inputs the ADR-005 heuristic derives `priority` from, so it is re-deriving the
  label rather than predicting it. Retained deliberately as a documented leakage
  example (ADR-009). Path A chosen: document the leakage, do NOT "fix" the features.
- `behavioral-only`: test F1 = 0.12, ROC-AUC = 0.85. This is the HONEST model and the
  production candidate (ADR-010). Low F1 with high ROC-AUC is the expected signature:
  the model ranks priority emails well but the 0.5 hard threshold misclassifies. At
  0.68% prevalence a keyword-derived label is not recoverable from behavioural
  signals alone. ROC-AUC 0.85 shows genuine signal in sender_frequency, time-of-day,
  and thread_depth.
- Report BOTH numbers together, always. Quoting the 1.00 alone would be misleading;
  quoting the 0.12 alone would hide that a leak was available and measured.

**Early-stopping diagnostic (`diag_early_stopping.py`, repo root, throwaway):**
- Built to explain best_iteration=1. Confirmed the metric plumbing is CORRECT:
  instrumenting `early_stopping_f1` on a small frame showed it is invoked every
  iteration, `evals_result_ == {'valid_0': {'f1_pos': [...]}}`, and f1_pos is the only
  reported metric. So the custom-metric wiring is not the problem.
- Mechanism observed: F1 sits at EXACTLY 0.0 for the opening iterations because no
  predicted probability clears 0.5 yet (iteration 1: max prob 0.14, 0 predicted
  positives; crosses 0.5 around iteration 11-12 on a small frame). Early stopping
  therefore burns patience before the model has learned anything. The suspected fix is
  the decision threshold, NOT patience.
- Two proposed remedies were checked and REJECTED as no-ops: patience is ALREADY 100
  (raising it is not possible; "increase to 20" would LOWER it), and min_child_samples
  is never set so it is already LightGBM's default of 20.
- The script was never run to completion: it crashed twice (my own bug — I unpacked two
  values from build_matrices(), which returns a single CSR matrix; then LIMIT was
  hardcoded so LIMIT=50000 silently reused 5000). Both fixed, neither run by me.
  It also has a structural limit — at 0.68% prevalence LIMIT=50000 yields only ~51
  validation positives vs ~529 in the real run, so it may never reproduce the failure.
- OPEN ITEM, do not close: both full runs reported best_iteration=1. Leakage explains
  it for `with-keyword-features`, but CANNOT explain it for `behavioral-only`, which
  excludes `urgency_keyword_count` and subject TF-IDF by construction. The pre-registered
  trigger in this log ("revisit only if the full run also collapses to best_iteration ~1")
  has fired. The tiny-val-set explanation above is ruled out at full scale. Cause of
  behavioral-only's best_iteration=1 remains UNEXPLAINED.

**MLflow config fixes (late session, after the SQLite migration above):**
- Absolute default tracking URI. `sqlite:///mlruns/mlflow.db` is relative and resolves
  against the process CWD, so launching from anywhere but the repo root silently created
  a second invisible database. Now anchored to the repo root via
  `_REPO_ROOT = Path(__file__).resolve().parents[2]`. This RESOLVED the caveat above.
- `load_dotenv()` added to train_flow.py, placed ABOVE the train_tasks import on purpose.
  train_tasks reads os.environ at IMPORT time, so load_dotenv() below the imports would
  have silently done nothing. Needs `# noqa: E402` on the late imports.
  CAVEAT: `python -c` counts as "interactive" to python-dotenv and searches CWD, so this
  only behaves correctly for `python -m`, not for `python -c` smoke tests.
- Still true: docker-compose.yml and .github/workflows/ci.yml export MLFLOW_TRACKING_URI
  pointing at a server on :5000, so training under CI targets that server. Untouched.

**What is broken or blocked:**
- Nothing blocked. Training stage is complete; no further changes to training code.
- OPEN QUESTION carried into the evaluation stage: behavioral-only's best_iteration=1
  (see above) is unexplained. It does not invalidate the F1/ROC-AUC result — the model
  was evaluated at its best iteration and ROC-AUC 0.85 stands — but if it stopped at
  iteration 1 it may be under-trained, and a longer/threshold-tuned fit could score
  better. Do not silently treat the current numbers as final.
- LABEL LEAKAGE CONFIRMED EMPIRICALLY, not just theoretically: on the 500-row smoke
  test, val class-1 F1 was 0.86 (with-keyword) and 1.00 (behavioral-only). The
  heuristic label is derived from subject keywords and the subject is a feature, so
  high F1 means "re-derived the heuristic". This is exactly why behavioral-only
  exists. Expect the same divergence at full scale — report both, never one alone.
- SMOKE-TEST F1 = 0.0000 IS NOT A BUG (verified, do not "fix" it): with 361 train
  rows / 14 positives, early stopping sees a val set of 3 positives where any
  all-negative prediction scores F1 = 0, so it halts at the first iteration. Disabling
  early stopping on the same 517 rows gives F1 0.75, recall 1.00, ROC-AUC 1.00. The
  full run has ~529 val positives and will not hit this. Left unhardened on purpose
  — revisit only if the full run also collapses to best_iteration ~1.

**Bugs found and fixed this session (all caught by running things, not by reading):**
1. LIGHTGBM HAS NO BUILT-IN "f1" METRIC. `eval_metric="f1"` is silently IGNORED and
   falls back to binary_logloss — so early stopping was optimising log loss, not the
   project's primary class-1 F1. Confirmed by inspecting evals_result_, which showed
   only 'binary_logloss'. Fixed with a custom `early_stopping_f1(y_true, y_pred)`
   returning the (name, value, higher_is_better) triple LightGBM requires, PLUS
   `metric="None"` in build_model — without disabling built-ins, binary_logloss is
   listed first and early stopping tracks IT, not F1. Regression-locked by
   test_train_model_early_stops_on_class_one_f1, which asserts f1_pos is the ONLY
   reported metric.
2. MLflow 3.15 put the FILE STORE in "maintenance mode" and hard-raises on any read
   or write unless MLFLOW_ALLOW_FILE_STORE=true. Originally worked around with
   os.environ.setdefault in train_tasks.py, but this was superseded later the same
   session — see "Tracking backend migrated to SQLite" below.
3. MLflow 3.x serialises sklearn models via skops and refuses untrusted types —
   log_model raised for LGBMClassifier/Booster/OrderedDict. Fixed with explicit
   `skops_trusted_types` (only types this project constructs itself).
4. Prefect 3 tasks called inside a flow return the RESOLVED VALUE, not a future, so
   my `.result()` calls raised AttributeError. Removed. Caught by test_train_flow.
5. `target-version = ["py312"]` under [tool.ruff] is rejected by ruff 0.16.3 once a
   [tool.ruff.lint] table exists; it must be a string. Migrated select to
   [tool.ruff.lint] (also silences the deprecation warning) and added per-file-ignores
   for N803/N806 on the ML modules — X_train/X_val are deliberate sklearn convention,
   and pep8-naming stays enforced everywhere else.
6. My own bug caught by tests: a column-list comprehension iterated each feature
   string CHARACTER-BY-CHARACTER, so load_features_task asked pyarrow for columns
   like "s","e","n". Fixed to a plain list concat.
7. lightgbm eval_set is deprecated in favour of eval_X/eval_y (which take the single
   val set UNWRAPPED, not in a list). Bumped the floor to lightgbm>=4.7.0 to match.

**Decisions made this session:**
- MLflow tracking moved from the file store to SQLite (see "Tracking backend migrated
  to SQLite" above). ADR-002 (local, zero-cost tracking) is unchanged — only the backend
  implementation moved.
- --limit does a STRATIFIED sample, not a head slice. The parquet is in ingest order,
  so the first N rows may contain zero of the 0.68% positives and a 3-way stratified
  split becomes impossible. Tops up to MIN_POSITIVE_IN_SAMPLE (20) positives.
- Registry registration and ONNX export deliberately EXCLUDED this session (evaluation
  stage), as agreed.
- Kept the v1 sender_frequency behaviour; leakage documented in code and in both MLflow
  run tags, per the plan.
- Bumped lightgbm floor 4.3.0 -> 4.7.0; added matplotlib>=3.8.0 (was already installed,
  3.11.1).
- PATH A CHOSEN for the confirmed label leakage: DOCUMENT IT, DO NOT FIX IT.
  `with-keyword-features` is kept in the MLflow experiment as a deliberate, labelled
  leakage example (ADR-009) so the leak is visible and measured rather than quietly
  removed. `behavioral-only` is the production candidate (ADR-010). This is a
  documentation decision only — no feature code was changed to accommodate it.

**Next session — exact first action:**
> Start the EVALUATION stage. Do not modify training code.
> 1. Identify the `behavioral-only` run_id from MLflow (it is the production candidate,
>    NOT the F1=1.00 with-keyword-features run). Confirm which backend the full run used
>    before reading — it was launched BEFORE the SQLite migration:
>      sqlite:  python -c "from mlflow import MlflowClient; c=MlflowClient(tracking_uri='sqlite:///mlruns/mlflow.db'); e=c.get_experiment_by_name('email-priority-classifier'); [print(r.info.run_id, r.data.tags.get('mlflow.runName'), r.data.metrics.get('test_f1'), r.data.metrics.get('test_roc_auc')) for r in c.search_runs([e.experiment_id])]"
>      file:    same but tracking_uri=pathlib.Path('mlruns').resolve().as_uri() and
>               MLFLOW_ALLOW_FILE_STORE=true exported.
>    The default is now an absolute SQLite path, so the file store is only relevant if
>    the run predates the migration.
> 2. Build evaluate_flow: load the model by run_id, report class-1 F1/P/R, the confusion
>    matrix, and the ROC-AUC. State the decision-threshold caveat explicitly — F1 is
>    threshold-sensitive and behavioral-only scored F1=0.12 at ROC-AUC=0.85, so a
>    threshold/PR-AUC view is worth including before concluding the model is weak.
> 3. Generate an Evidently baseline report on the training data (data drift reference).
> 4. Write the HTML evaluation report to reports/evaluation/.
> Carry forward: the unexplained best_iteration=1 on behavioral-only. Check whether the
> model was effectively under-trained before writing the report as final.

---
### Session: 2026-08-17 (EDA + Ingestion)
**Duration:** —
**Stage:** EDA

**What I completed:**
- Wrote notebooks/01_eda.py: streamed 5,000 emails from the tar.gz (no extraction).
  Found 517,401 total emails, date range 2000-01-21 → 2002-03-08 (UTC), 455 unique
  senders in sample, avg subject length 34.1 chars, `to` null rate 26.4%
  (`from`/`subject`/`date` at 0%).
- Implemented src/labels/labeling.py (ADR-005 heuristic labels: urgency keywords +
  leadership senders) and src/ingestion/ingest.py (pure functions: parse_email,
  extract_emails, validate_records, save_parquet, ingest). Refactored EDA to reuse
  the shared parser via sys.path bootstrap + lazy import in main().
- Wrote tests/test_labeling.py and tests/test_ingest.py (15 tests) using a synthetic
  tar.gz fixture. All pass; black + ruff clean.
- Wired ingestion into Prefect: src/tasks/ingest_tasks.py (ingest_emails_task) and
  src/flows/ingest_flow.py (ingest_flow, click CLI, --limit/--batch-size). Flow
  verified with --limit 100 smoke test in OpenCode (100 rows, correct schema).

**What is broken or blocked:**
- `to` field 26.4% null in EDA — decision: keep nulls, do not drop.
- FULL RUN OOM: first full-dataset attempt (517k emails) was killed by the Linux
  OOM killer — machine has only 5.6GB RAM (committed > commit limit). Original code
  held 3 full in-memory copies (raw list + validated list + DataFrame).
- Fixed with batched streaming: iter_records generator + write_records_parquet
  (pyarrow ParquetWriter, batch_size=50k). Memory now bounded regardless of dataset
  size. 18 tests pass. Full run still pending in user's terminal.
- SECOND FULL RUN FAILED (stdlib bug): crashed ~11 min in with
  `TypeError: 'ValueTerminal' object does not support item assignment`. Root cause:
  policy.default lazily applies Python's strict structured-address parser to headers;
  a malformed Enron header (`To: Sales Team@enron.com, .casaudoumecq@enron.com, ...`
  — display name followed by @ with no angle brackets) triggers HeaderParseError and
  then a CPython 3.12 recovery bug. Left a PARTIAL parquet (250,000/517,401 rows)
  which was deleted.
- FIXED: `_safe_header()` (strict parse → raw fallback via msg.raw_items()) applied
  to all headers; subject decoded via email.header.make_header with raw fallback.
  Also added `logger` param to ingest/write_records_parquet; task passes
  get_run_logger() so per-batch progress lines now show in the terminal
  (verified with --batch-size 10 --limit 100 smoke). 22 tests pass.
- FULL RUN COMPLETED (2026-08-18): 517,401 emails → data/processed/emails.parquet
  (511MB, 11 row groups). ~1.9 min per 50k batch, ~22 min total, 0 dropped.
  Verified: priority 0.68% positive (3,526/517,401) — heavily imbalanced; null
  rates from 0%, to 4.22%, subject/date/body 0%. NOTE: EDA estimated `to` null at
  26.4% but the full dataset shows 4.22% — the 5,000-email sample was dominated
  by one mailbox (blair-l) and was not representative. Full-data null rate is the
  correct one.

**Decisions made this session:**
- Shared RFC 822 parser lives in src/ingestion/ingest.py; notebooks import it.
- EDA direct-run bootstrap: `sys.path.insert` of repo root + import inside main().
- Leadership sender list is a small documented set in labeling.py.
- DAG collapsed to a single streaming task (ingest_emails_task) instead of
  extract→validate→save: the 3-task split forced materializing the full dataset
  between tasks, which was the OOM root cause.
- `to` null rate on full data is 4.22% (EDA's 26.4% was sample bias) — keep nulls.

**Next session — exact first action:**
> Build the feature layer: src/features/features.py (per ARCHITECTURE.md feature
> hypotheses) + src/features pipeline, validate each feature in EDA before
> implementing, then wire src/tasks/feature_tasks.py + src/flows/feature_flow.py.

---

### Session: 2026-08-18 (Features)
**Duration:** —
**Stage:** Features

**What I completed:**
- Wrote ADR-008 in ARCHITECTURE.md: sender normalization features-only (labels/3,526
  positives untouched), TF-IDF fit deferred to train_flow (never full dataset),
  sentinel -1 for bad dates, TF-IDF as scipy sparse (never dense ~200MB on 5.6GB),
  features.py pure functions with Prefect wiring after.
- Implemented src/features/features.py (pure functions, no Prefect): normalize_sender
  (addr-spec extraction w/ raw fallback), parse_dates (utc, errors='coerce'), and 8
  tabular features — sender_frequency, sender_is_leadership, hour_of_day, day_of_week,
  urgency_keyword_count, subject_length, has_question_mark, thread_depth. Reuses
  URGENCY_KEYWORDS/LEADERSHIP_SENDERS from src.labels.labeling (single source of truth).
- TF-IDF layer: fit_subject_vectorizer() and transform_subjects() returning scipy
  csr_matrix (never dense). Added scipy to pyproject.toml dependencies.
- engineer_features() output = uid, from, subject, date, priority + 8 features
  (13 cols) — explicitly DROPS body and to (no 511MB body payload in features.parquet).
- Wrote tests/test_features.py: 46 tests (>=3 per function, incl. null handling,
  sentinel -1, display-name normalization, sparse TF-IDF shape). Full suite 68 pass.
- Fixed 3 real bugs caught by tests: (1) null sender counted in "" frequency bucket →
  explicitly zeroed; (2) thread_depth regex matched "re" inside "reorg?" because the
  colon was optional → colon now required after RE/FW/FWD/REW markers; (3) bad test
  data (stop-word subjects) triggered sklearn "empty vocabulary".
- Verified in OpenCode on a 1000-row sample from data/processed/emails.parquet
  (columns-only read to avoid loading body): shape (1000, 13), all feature dtypes
  int64, body/to dropped. black + ruff clean.
- Wired Prefect layer: src/tasks/feature_tasks.py (feature_engineering_task, reads
  only KEEP_COLUMNS so body never loads) + src/flows/feature_flow.py (click CLI:
  --input/--output/--limit, python -m src.flows.feature_flow). tests/test_features_flow.py
  (3 tests). 71 total tests pass. Flow smoke-tested with --limit 100 in OpenCode:
  100 rows x 13 cols written correctly.

**What is broken or blocked:**
- Nothing blocked. Full feature_flow run (517k) NOT yet done — goes in the user's
  WSL terminal (next step).

**Decisions made this session:**
- ADR-008 (see ARCHITECTURE.md): normalization features-only; TF-IDF fit in train_flow
  only; -1 sentinel; sparse TF-IDF; pure-function features layer.
- sender_frequency known limitation documented in docstring: frequency computed on
  full dataset incl. test split — minor leakage, acceptable for v1, fix in v2.
- engineer_features keeps subject + date (needed for TF-IDF in train_flow and drift).

**Next session — exact first action:**
> Run the full feature_flow in the WSL terminal (NOT OpenCode — 517k rows):
>   python -m src.flows.feature_flow --input data/processed/emails.parquet \
>     --output data/features/features.parquet
> Expected ~2-4 min (columns-only read, no body). Verify output shape (517401, 13)
> and feature dtypes, then start the Training stage (train_flow: TF-IDF
> fit_subject_vectorizer on train split only, LightGBM, MLflow tracking).

---

## [COPY THIS TEMPLATE FOR EACH SESSION]
### Session: YYYY-MM-DD
**Duration:** X hrs
**Stage:** Setup | EDA | Ingestion | Features | Training | Evaluation | Serving | Monitoring | CI/CD

**What I completed:**
-

**What is broken or blocked:**
-

**Decisions made this session:**
-

**Next session — exact first action:**
> Write one specific sentence. Not "continue X". Write "Run Y command and check Z output."

---

## Session History

### Session: 2026-08-16 (Initial Setup)
**Duration:** —
**Stage:** Setup

**What I completed:**
- Defined project scope and architecture
- Created CLAUDE.md, ARCHITECTURE.md, SESSION_LOG.md, AGENTS.md
- Created directory structure and all config files

**What is broken or blocked:**
- Docker WSL integration not yet verified (docker --version works but test with docker run hello-world)
- Enron dataset not downloaded
- Python virtual environment not created
- uv not installed

**Decisions made this session:**
- LightGBM as primary model (not neural) until F1 ≥ 0.70
- Subject-only NLP (not full body) for v1
- Local MLflow via Docker Compose
- Heuristic labels — documented limitation

**Next session — exact first action:**
> Run `docker run hello-world` in WSL to confirm Docker works, then run
> `pip install uv && uv venv .venv && source .venv/bin/activate && uv pip install -r requirements.txt`
> then download the Enron dataset tar.gz and place it in data/raw/
