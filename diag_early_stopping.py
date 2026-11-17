"""Read-only early-stopping diagnostic. Run from the repo root. Touches no MLflow.

Reproduces the --limit path exactly (stratified sample, seed 42, topped up to
MIN_POSITIVE_IN_SAMPLE positives) and dumps everything needed to explain
best_iteration=1 on the full 362k-row run.

Sample size is controlled by the LIMIT env var:
    LIMIT=5000  python diag_early_stopping.py
    LIMIT=50000 python diag_early_stopping.py
"""

import os

import lightgbm as lgb
import numpy as np
import pandas as pd

from src.features.features import FEATURE_COLUMNS, fit_subject_vectorizer
from src.tasks.train_tasks import MIN_POSITIVE_IN_SAMPLE
from src.training.train import (
    SPLIT_SEED,
    SUBJECT,
    TARGET,
    VARIANTS,
    build_matrices,
    build_model,
    stratified_split,
)

INPUT = "data/features/features.parquet"
# Read from the environment so the sample size can change between runs. The old
# hardcoded 5000 made LIMIT=50000 a silent no-op: both runs printed 3500/750/750.
LIMIT = int(os.environ.get("LIMIT", 5000))
spec = VARIANTS[0]  # with-keyword-features

df = pd.read_parquet(INPUT, columns=[SUBJECT, TARGET, *FEATURE_COLUMNS])
print(f"[A] read {len(df):,} rows, {int((df[TARGET]==1).sum()):,} positives")
print(f"[A] LIMIT requested: {LIMIT:,}")

if LIMIT < 4 * MIN_POSITIVE_IN_SAMPLE:
    raise SystemExit(
        f"LIMIT={LIMIT} is too small to hold "
        f"{MIN_POSITIVE_IN_SAMPLE} positives in a 15% validation split. "
        f"Use at least {4 * MIN_POSITIVE_IN_SAMPLE}."
    )

# Same stratified subsample as load_features_task(limit=...).
# NOTE: LIMIT is NOT a parameter of stratified_split(); that function only takes
# (df, seed, fractions) and splits whatever frame it is given. LIMIT changes the
# size of `s` below, which IS the argument passed to it.
limit = min(LIMIT, len(df))
frac = limit / len(df)
parts = [
    g.sample(frac=frac, random_state=SPLIT_SEED)
    for _, g in df.groupby(TARGET, sort=True)
]
s = pd.concat(parts)
if int((s[TARGET] == 1).sum()) < MIN_POSITIVE_IN_SAMPLE:
    pos = df[df[TARGET] == 1]
    s = pd.concat(
        [
            s,
            pos.sample(
                n=MIN_POSITIVE_IN_SAMPLE - int((s[TARGET] == 1).sum()),
                random_state=SPLIT_SEED,
            ),
        ]
    )
s = s.sample(frac=1.0, random_state=SPLIT_SEED).reset_index(drop=True)
print(
    f"[A] sample drawn: {len(s):,} rows, {int((s[TARGET]==1).sum()):,} positives"
    f" ({100.0 * float((s[TARGET]==1).mean()):.3f}%)"
)

tr, va, te = stratified_split(s)
for name, part in (("train", tr), ("val", va), ("test", te)):
    print(
        f"[A] {name:5s} {len(part):6,} rows"
        f"  {int((part[TARGET] == 1).sum()):4d} positives"
    )

n_val_pos = int((va[TARGET] == 1).sum())
if n_val_pos == 0:
    print("[F] VERDICT: DEGENERATE VAL SPLIT - zero positives, F1 is 0.0 at every")
    print("    iteration, so best_iteration is pinned to 1. This is the bug.")

vec = fit_subject_vectorizer(tr[SUBJECT])
# build_matrices(df, ...) -> csr_matrix: ONE frame in, ONE matrix out. It is not a
# tuple-returning helper, so train and val need two separate calls.
Xtr = build_matrices(
    tr, vectorizer=vec, feature_columns=spec.feature_columns, use_tfidf=spec.use_tfidf
)
Xva = build_matrices(
    va, vectorizer=vec, feature_columns=spec.feature_columns, use_tfidf=spec.use_tfidf
)
ytr, yva = np.asarray(tr[TARGET]), np.asarray(va[TARGET])
print(f"[A] matrix train {Xtr.shape}  val {Xva.shape}")

trace = []


def feval(y_true, y_pred):
    y_pred = np.asarray(y_pred).ravel()
    y_true = np.asarray(y_true).ravel()
    val = float(
        2
        * np.sum((y_pred > 0.5) & (y_true == 1))
        / max(np.sum(y_pred > 0.5) + np.sum(y_true == 1), 1e-12)
    )
    trace.append((len(trace) + 1, val, float(y_pred.max()), int((y_pred > 0.5).sum())))
    return "f1_pos", val, True


m = build_model()
p = m.get_params()
print(
    f"[E] scale_pos_weight={p['scale_pos_weight']}"
    f"  min_child_samples={p['min_child_samples']}"
    f"  num_leaves={p['num_leaves']}  lr={p['learning_rate']}"
    f"  n_estimators={p['n_estimators']}"
    f"  metric={p.get('metric')}"
)

m.fit(
    Xtr,
    ytr,
    eval_X=Xva,
    eval_y=yva,
    eval_metric=feval,
    callbacks=[lgb.early_stopping(100, verbose=False)],
)

er = m.evals_result_
print(f"\n[B] feval invoked: {len(trace)} times (True = yes)")
print(f"[B] evals_result_ = dict with keys {list(er.keys())}")
print(f"[B]   inner metric keys: {list(er['valid_0'].keys())}")

traj = [t[1] for t in trace]
best_i = int(np.argmax(traj))
print(f"\n[C] trajectory length : {len(traj)}")
print(f"[C] first 15 f1_pos   : {[round(x,4) for x in traj[:15]]}")
print(f"[C] last 10 f1_pos    : {[round(x,4) for x in traj[-10:]]}")
print(f"[C] max f1_pos        : {max(traj):.6f} at iteration {best_i+1}")
print(f"[C] zeros in traj     : {sum(1 for x in traj if x == 0.0)}/{len(traj)}")
print(f"[C] best_iteration_   : {getattr(m,'best_iteration_',None)}")
print(f"[C] best_score_       : {dict(getattr(m,'best_score_',{}))}")
print(f"[C] iterations run    : {len(traj)} (patience 100 -> expect best_i+101)")

print(
    f"\n[D] it | f1_pos | max prob | #pred positive" f"   (val positives = {n_val_pos})"
)
for it, v, mp, npred in trace[:12]:
    print(f"      {it:3d} | {v:7.4f} | {mp:8.4f} | {npred:5d}")
first_pred = next((it for it, v, mp, npr in trace if npr > 0), None)
print(f"[D] first iteration with ANY predicted positive: {first_pred}")

print("\n[F] CLASSIFICATION")
if n_val_pos == 0:
    print(
        "  -> DEGENERATE VAL SPLIT (0 positives)."
        " Root cause, not a hyperparameter problem."
    )
elif max(traj) == 0.0:
    print("  -> MODEL NEVER CROSSES 0.5 ON VAL. F1 identically 0, best pinned to 1.")
elif best_i == 0:
    print("  -> ITERATION-1 PEAK THEN DECLINE. F1 is highest at iter 1 and never")
    print(
        "     recovers; raising patience will NOT help. Investigate threshold/overfit."
    )
else:
    print(f"  -> DOES NOT REPRODUCE at 5000 rows: best_iteration={best_i+1}.")
    print(
        "     Failure is scale-dependent -> re-run this same script with LIMIT=50000."
    )
