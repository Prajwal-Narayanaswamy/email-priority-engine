# Project Narrative — Email Priority Engine

*Recruiter / interview-prep document. Every number below is from the actual
pipeline run; you can defend all of it.*

---

## 1. What problem this solves

Enterprise inboxes flood people with thousands of emails a day, and the
time-sensitive ones get buried. This project builds an end-to-end MLOps
pipeline that reads raw email archives and predicts whether each message is
"priority" — something that needs action today — so a real-time service could
triage an inbox automatically. It's the full machine-learning production stack,
not a notebook: streaming data ingestion, feature engineering, model training
with experiment tracking, a serving API, drift monitoring, and CI/CD. Everything
runs locally with no cloud inference and no external APIs, so the email data
never leaves the machine.

## 2. Why the Enron dataset

The Enron corpus is the canonical public benchmark for email analysis: ~517,000
real corporate emails from senior executives, released in 2003 after the FERC
investigation. It's attractive for three reasons. First, it's **real and messy** —
malformed headers, encoded subjects, forwarded threads, and legal disclaimers —
which is exactly what production email data looks like, unlike toy datasets.
Second, it's **large enough** to exercise the engineering problems that matter
(517k rows can't fit comfortably in a 5.6GB laptop's memory if you're careless).
Third, it's **ethically clean to use publicly** — the corpus was already released
by regulators, so it's safe to demo and commit to a public repo. The trade-off:
Enron has **no priority labels**, which drove the labeling decision below.

## 3. The labeling problem and how we solved it

Email priority is a subjective label that Enron never recorded, so there's no
ground truth to train against. Options: (a) hand-label thousands of emails —
expensive and inconsistent; (b) use a synthetic label — which is what we did,
and we were explicit about it.

Our label (ADR-005, single source of truth in `src/labels/labeling.py`):

> `priority = 1` if the sender is in a known leadership list **or** the subject
> contains an urgency keyword (urgent, asap, immediately, deadline, action
> required, time sensitive, eod, eow, by end of).

This is a **documented, heuristic label** — it encodes the domain belief that
urgency is signaled by *who* wrote and *what* the subject says. It is not a
subjective human label, and we say so in the README and reports. The model learns
to predict our definition of priority, which is a defensible engineering choice
for a portfolio when no ground truth exists. On the full dataset this produced
**3,526 priority emails out of 517,401 (0.68%)**.

## 4. Why accuracy is the wrong metric — the actual numbers

The dataset has **513,875 negatives vs 3,526 positives (0.68% positive)**. A model
that predicts "not urgent" for every single email scores:

> 513,875 / 517,401 = **99.33% accuracy**

...while catching **zero** urgent emails. That's the trap: on a 99:1 imbalanced
dataset, accuracy rewards a do-nothing model. The metric that matters is
performance on the rare class. We report **precision, recall, and F1 on class 1**
(plus class-level breakdowns for both classes — class 1 is the one we care
about). F1 is the harmonic mean of precision and recall, so it only looks good
if the model finds urgent emails *and* doesn't cry wolf. Accuracy is reported but
never used as the primary decision metric — it's in our hard rules and enforced
in CI (AGENTS/CLAUDE rules).

## 5. Why LightGBM and not neural networks

Three reasons (ADR-003):

1. **Sparse features, fast training.** Our NLP signal is a TF-IDF matrix of the
   subject line — naturally sparse. Gradient-boosted trees handle sparse matrices
   natively and efficiently; a neural net needs dense embeddings (with
   pre-training) to compete. On a 5.6GB laptop, LightGBM trains on 517k rows in
   minutes.
2. **Iteration speed.** We get a trainable baseline in one session, which matters
   when development time is limited. The faster we iterate, the more production
   problems we can demonstrate.
3. **Interpretability.** Tree models expose feature importances we can log and
   monitor for drift — essential for the monitoring layer (Evidently compares
   live inputs vs training distribution). A black-box transformer would make
   drift debugging harder.

The ADR is explicit: **DistilBERT is a stretch goal only if the LightGBM baseline
fails to reach F1 ≥ 0.70**. We don't reach for the impressive-sounding model
first; we establish a solid, explainable baseline.

## 6. Two production problems we actually hit and solved

These are the stories interviewers remember.

**Problem 1 — Out-of-memory (OOM) on a 5.6GB machine.**
The first full ingestion run (517k emails) was **killed by the Linux OOM killer**.
Root cause: the original code held **three full in-memory copies** — the raw
extracted list, the validated list, and the DataFrame. On a 5.6GB host that's a
hard ceiling, not a tuning issue. Fix: **batched streaming** — a generator
(`iter_records`) yields emails one at a time, and a pyarrow `ParquetWriter`
buffers 50,000 emails per write. Memory is now bounded regardless of dataset
size; the full run completed (517,401 emails, 0 dropped, ~22 min) with memory
flat. The lesson: measure the data's memory footprint before designing the
pipeline, and design for bounded memory from the start.

**Problem 2 — A crash inside CPython 3.12's email parser.**
Halfway through a full run (11 minutes in) the process crashed with
`TypeError: 'ValueTerminal' object does not support item assignment`. Root cause:
`email.policy.default` *lazily* applies Python's strict structured-address
parser when a header is accessed. A real Enron header —
`To: Sales Team@enron.com, .casaudoumecq@enron.com, ...` (a display name followed
by `@` with no angle brackets) — is malformed, so the strict parse raised
`HeaderParseError`, and CPython 3.12 has a recovery bug that then crashes the
interpreter. Fix: `_safe_header()`, which tries the strict parse and falls back
to the **raw header bytes** via `msg.raw_items()` — plus `_decode_subject()` for
RFC 2047 encoded subjects, also with a raw fallback. The full run then completed
cleanly. The lesson: real-world data breaks standard-library assumptions, and
"tolerate malformed input" is a design requirement, not an afterthought. (This
failure was later caught as a CI regression on CPython 3.12.13, where the same
malformed header stopped crashing but re-rendered differently — a version
behavior change we caught with a semantic-preservation test.)

## 7. The TF-IDF leakage decision and why it matters

TF-IDF learns a vocabulary and term weights from a corpus. If we fit the
vectorizer on **all** data including the test split, the model has implicitly
seen test-token frequencies during training — that's **data leakage**, and it
inflates evaluation scores that won't hold in production.

Decision (ADR-008): the features layer only exposes `fit_subject_vectorizer()`
and `transform_subjects()`; **the vectorizer is fit on the training split only,
never the full dataset**, and is applied to train/validation/test. This is the
same discipline as fitting a StandardScaler on train only. The fitted vectorizer
is saved as an MLflow artifact so serving applies exactly the same transform.

Second part of the decision, born from the OOM problem: the TF-IDF output is
stored as **scipy sparse matrices**, never dense float32 columns. A 517,401 × 100
dense array is ~200MB — on a 5.6GB host that's a real OOM risk. Sparse storage
keeps it at a few MB. Two-for-one: correct methodology *and* memory discipline.

## 8. Why Prefect over Airflow

Airflow is the industry standard, but its operational cost is real: a metadata
database, a scheduler, and a webserver just to get started. For a solo, fully
local project on WSL that's disproportionate overhead. **Prefect 3.x runs with
zero infrastructure locally** — flows execute immediately, and you get the same
core concepts (directed acyclic graphs, retries, task/flow boundaries, logging,
the ability to later run against a real server). The orchestrator concepts
transfer directly if a team is on Airflow; the difference is operational
footprint, not capability (ADR-001). We also design every flow to be
independently runnable (`python -m src.flows.<name>`) with a CLI — so nothing
hides behind the orchestrator, and each stage is testable in isolation.

## 9. Current project state

| Stage | Status | Details |
|---|---|---|
| EDA | ✅ Done | 517,401 emails, date range 2000-01-21 → 2002-03-08, field null analysis |
| Ingestion | ✅ Done | Streaming/batched; 423MB tar.gz → 511MB parquet (11 row groups), 0 dropped, ~22 min |
| Labeling | ✅ Done | Heuristic (ADR-005); 3,526 positives (0.68%) |
| Feature engineering | ✅ Done | 8 tabular features + sparse TF-IDF (fit at train time); feature parquet 517,401 × 13, 0 nulls |
| CI/CD | ✅ Done | GitHub Actions: lint (ruff/black/isort) + tests + coverage; 71 tests passing, 92% coverage |
| Git workflow | ✅ Done | Branch-per-change, PRs, branch protection on `main`, SSH auth |
| Training | 🔜 In progress | LightGBM + `scale_pos_weight=145.7`, MLflow local `./mlruns` |
| Evaluation | ⬜ Not started | Per-class F1/P/R, ROC-AUC, confusion matrix, Evidently baseline report |
| ONNX export | ⬜ Not started | LightGBM → ONNX, validated via onnxruntime |
| Serving | ⬜ Not started | FastAPI + Docker (`/predict`, `/predict/batch`, `/health`, `/metrics`) |
| Monitoring | ⬜ Not started | Evidently drift, Prometheus + Grafana, SQLite prediction log |
| Data versioning | ⬜ Not started | DVC tracking of artifacts |

## 10. The 2-minute pitch

> "I built a production email-priority classifier end to end on the Enron corpus —
> about 517,000 real corporate emails from executives, which is the standard
> public benchmark for email. The hard part is there's no ground truth, so I
> defined a defensible heuristic label: priority if the sender is leadership or
> the subject has an urgency keyword. That gives 3,526 positives out of 517,401 —
> 0.68% — which is why I never use accuracy; a model that predicts 'not urgent'
> for everything scores 99.33% accuracy while catching nothing, so I evaluate on
> per-class precision, recall, and F1.
>
> The pipeline is real production engineering, not a notebook. I stream the
> archive in batches because my first naive version got OOM-killed — the machine
> has 5.6GB of RAM — and I fixed a crash inside CPython's own email parser caused
> by malformed Enron headers. Features are eight tabular signals plus TF-IDF on
> the subject line, where the vectorizer is fit on the training split only to
> avoid leakage. The model is LightGBM with class weights — 145.7 to one — chosen
> because it handles sparse features and is interpretable; a transformer is the
> stretch goal only if the baseline misses F1 0.70. Everything is orchestrated
> with Prefect, tracked in MLflow, and gated by CI — 71 tests passing. The
> serving and drift-monitoring layers are the next stage."

## 11. Six questions to answer cold

**Q1. Your accuracy is 99.33% — why isn't that good?**
Because the dataset is 513,875 negatives vs 3,526 positives (0.68%). A model that
predicts "not urgent" for every email gets 99.33% accuracy and finds zero urgent
emails — it's useless. I report precision, recall, and F1 on the positive class,
with class-level breakdowns for both. Accuracy appears in reports but is never
the decision metric.

**Q2. How do you train without ground-truth labels?**
Enron has no priority labels, so I defined a transparent heuristic: priority = 1
if the sender is on a known leadership list or the subject contains one of nine
urgency keywords. It's documented as a synthetic label in the architecture
decisions and single-sourced in `src/labels/labeling.py`. The model learns that
definition; it's an honest, defensible choice when no human labels exist.

**Q3. Walk me through the OOM you hit.**
My first ingestion design held three full copies of 517k emails in memory — raw,
validated, and a DataFrame — and the Linux OOM killer killed the run on my 5.6GB
machine. I switched to a generator that yields emails one at a time and a pyarrow
ParquetWriter that buffers 50,000 rows per write, so memory is bounded no matter
the dataset size. The full run then finished in ~22 minutes with flat memory and
zero dropped records.

**Q4. You mentioned crashing inside CPython — what happened?**
`email.policy.default` lazily applies Python's strict structured-address parser
when you read a header. A real Enron `To:` header like
`Sales Team@enron.com, .casaudoumecq@enron.com` is malformed — display name
followed by @ with no angle brackets — so the strict parse raises, and CPython
3.12 has a recovery bug that crashed the interpreter mid-run. I added
`_safe_header()`, which tries the strict path and falls back to raw header bytes
via `msg.raw_items()`, plus a decoder for RFC 2047 encoded subjects. The lesson:
real data breaks stdlib assumptions; tolerate malformed input by design.

**Q5. What's the TF-IDF leakage and what did you do?**
If you fit a TF-IDF vocabulary on the whole dataset, the test split's token
frequencies leak into training and inflate your scores. So the vectorizer is fit
only on the training split and saved as an artifact so serving applies the same
transform. I also store the output as scipy sparse — a dense 517k×100 matrix is
~200MB and would risk OOM on this machine.

**Q6. How do you handle the 0.68% class imbalance in the model?**
Two ways. Evaluation: precision/recall/F1 on class 1 instead of accuracy.
Training: LightGBM gets `scale_pos_weight=145.7` — the exact ratio of negatives
to positives (513,875 / 3,526) — so the model is penalized appropriately for
missing the rare class. Without it, LightGBM would learn to predict all-zeros.
The value is logged as a hyperparameter in MLflow so every run is reproducible.
