"""Isolation Forest anomaly detection on KDD Cup 1999: can a one-class model trained
on normal network connections reliably detect intrusion attacks?

Unlike the planets script, this dataset provides ground-truth attack labels.
The model is trained on normal connections only (one-class setup) and evaluated
with ROC-AUC, precision, and recall against known labels on a held-out test set."""

# %%
""" [1] Imports & config"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score, roc_curve,
    precision_recall_fscore_support,
    confusion_matrix, ConfusionMatrixDisplay,
)
from cases.sklearn_kddcup99.kdd_utils import load_kddcup99
from cases.sklearn_kddcup99.kdd_utils import NUMERIC_FEATURES as FEATURES

PLOT_DIR = Path(__file__).parent / "plots"
PLOT_DIR.mkdir(exist_ok=True)

RANDOM_STATE  = 42
TEST_SIZE     = 0.25
CONTAMINATION = 0.10   # only affects predict() threshold; evaluation uses score_samples()


# %%
""" [2] Load & prepare features
Same decode pipeline as the EDA script. After decoding and casting, three
highly skewed columns (src_bytes, dst_bytes, duration) are log₁₀-transformed
and stored as new columns — this was justified in EDA where linear axes showed
the distributions were dominated by a spike at zero.

Feature selection reduces 30+ numeric columns to 11 non-redundant features:
  - Traffic volume:      log_src_bytes, log_dst_bytes, log_duration
  - Connection freq:     count, srv_count, dst_host_count
  - Error rates:         serror_rate, rerror_rate (one per correlated family;
                         dropping srv_serror_rate, dst_host_serror_rate etc.
                         prevents over-weighting the SYN-flood signal)
  - Behavioral pattern:  same_srv_rate, diff_srv_rate
  - Authentication:      logged_in

Categorical features (protocol_type, service, flag) are excluded; the numeric
features already capture the relevant signal (flag=S0 → serror_rate=1.0).
"""
df = load_kddcup99(random_state=RANDOM_STATE)
for col in ["src_bytes", "dst_bytes", "duration"]:
    df[f"log_{col}"] = np.log10(df[col] + 1)

print(f"Dataset: {len(df):,} rows  |  "
      f"normal: {(df['is_attack']==0).sum():,}  "
      f"attack: {(df['is_attack']==1).sum():,}")
print(f"\nMissing in selected features:\n{df[FEATURES].isnull().sum()}")


# %%
""" [3] One-class train / test split
One-class anomaly detection setup:
  1. Stratified split on is_attack preserves the normal/attack ratio in both halves.
  2. IsolationForest is fit on NORMAL training rows only — it learns what 'normal'
     looks like without seeing any labeled attacks.
  3. At inference, score_samples() is called on the full test set (normal + attack);
     lower scores flag likely intrusions.

This mirrors real-world deployment: you have a baseline of clean traffic and
want to detect deviations from it, with no labeled attack examples during training.

Row indices are tracked so attack-type labels can be retrieved from df for the
post-hoc breakdown analysis.
"""
X   = df[FEATURES].values
y   = df["is_attack"].values
idx = np.arange(len(df))

X_tr_all, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
    X, y, idx, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE,
)

normal_mask  = y_train == 0
X_tr_normal  = X_tr_all[normal_mask]

scaler   = StandardScaler().fit(X_tr_normal)
X_tr_s   = scaler.transform(X_tr_normal)   # used for fitting only
X_test_s = scaler.transform(X_test)

df_test = df.iloc[idx_test].reset_index(drop=True)

print(f"\nFit on:  {len(X_tr_s):,} normal training connections")
print(f"Test set: {len(X_test):,} connections  "
      f"({(y_test==0).sum():,} normal, {(y_test==1).sum():,} attacks)")


# %%
""" [4] Baseline: single-feature threshold on serror_rate
Simplest possible baseline: flag as attack if serror_rate > 0.5.

serror_rate was the most discriminating feature in EDA:
  neptune drives it to 1.0 (every connection in the window times out via SYN error)
  normal traffic sits near 0.0 (connections complete successfully)

Threshold = 0.5 requires no training and no parameter tuning — it is a hard-coded
rule derived from domain knowledge. Its fundamental limitation: it catches only
attacks that produce SYN errors. Probe, R2L, and U2R attacks operate at normal
error rates and are invisible to this single-feature rule.
"""
serror_idx = FEATURES.index("serror_rate")
baseline_scores = X_test[:, serror_idx]
baseline_pred   = (baseline_scores > 0.5).astype(int)

bl_prec, bl_rec, bl_f1, _ = precision_recall_fscore_support(
    y_test, baseline_pred, average="binary", zero_division=0,
)
bl_auc = roc_auc_score(y_test, baseline_scores)

print(f"\nBaseline (serror_rate > 0.5):")
print(f"  Precision: {bl_prec:.3f}  Recall: {bl_rec:.3f}  F1: {bl_f1:.3f}  AUC: {bl_auc:.3f}")


# %%
""" [5] Isolation Forest
IsolationForest is fit on normal connections only.
score_samples() returns negative values — more negative means more anomalous.
Scores are negated when passed to roc_auc_score so that higher values indicate
attacks (matching the convention that the positive class has higher scores).

Key parameters:
  n_estimators=200   — doubled from default for stable scores on this dataset size
  max_samples='auto' — min(256, n_train) sub-samples per tree; fast and robust
  contamination      — sets the predict() threshold only; does NOT affect scores

For one-class models the contamination value is somewhat arbitrary because you
do not know the true attack fraction at training time. We set it to 0.10 as a
plausible prior and use score_samples() + ROC-AUC for calibration-free evaluation.
"""
iso = IsolationForest(
    n_estimators=200,
    contamination=CONTAMINATION,
    max_samples="auto",
    random_state=RANDOM_STATE,
)
iso.fit(X_tr_s)

scores_train = iso.score_samples(X_tr_s)
scores_test  = iso.score_samples(X_test_s)

threshold    = np.percentile(scores_train, CONTAMINATION * 100)
if_pred_test = (scores_test < threshold).astype(int)

if_prec, if_rec, if_f1, _ = precision_recall_fscore_support(
    y_test, if_pred_test, average="binary", zero_division=0,
)
if_auc = roc_auc_score(y_test, -scores_test)

print(f"\nIsolation Forest  (contamination={CONTAMINATION}):")
print(f"  Precision: {if_prec:.3f}  Recall: {if_rec:.3f}  F1: {if_f1:.3f}  AUC: {if_auc:.3f}")
print(f"\n  Threshold: {threshold:.4f}  (= {CONTAMINATION*100:.0f}th pct of normal train scores)")


# %%
""" [6] Score distribution — normal vs attack in test set
Score histograms for normal and attack connections in the test set.
A well-separated pair of distributions indicates the model can discriminate
between classes with a simple threshold; overlap reveals ambiguous connections.

The red threshold line is set at the CONTAMINATION percentile of training normal
scores — where we start calling test connections anomalous.

Note the bimodal structure in the attack distribution: neptune connections cluster
at the far-left (very easy to isolate via serror_rate), while other attack types
appear at intermediate scores, closer to — or overlapping with — the normal cluster.
"""
fig, ax = plt.subplots(figsize=(9, 4))
for label, mask, color in [
    ("normal (test)",  y_test == 0, "steelblue"),
    ("attack (test)",  y_test == 1, "tomato"),
]:
    ax.hist(scores_test[mask], bins=80, alpha=0.6,
            color=color, label=label, edgecolor="none")
ax.axvline(threshold, color="red", linestyle="--", linewidth=1.5,
           label=f"threshold (train {CONTAMINATION*100:.0f}th pct)")
ax.set_xlabel("Anomaly score  (lower = more anomalous)")
ax.set_ylabel("Number of connections")
ax.set_title("Isolation Forest — score distribution by true label (test set)")
ax.legend()
fig.tight_layout()
fig.savefig(PLOT_DIR / "kddcup99_if_score_dist.png")
plt.show()


# %%
""" [7] ROC curves — baseline vs Isolation Forest
ROC curves compare detectors across all thresholds simultaneously.
AUC (area under curve) summarises detection power independently of any
chosen operating threshold.

IF should outperform the single-feature baseline because it combines all 11
features and can isolate attacks that are not distinguishable by serror_rate alone
(probe scans, remote-to-local, user-to-root attacks).

Both curves are evaluated on the same held-out test set.
"""
bl_fpr, bl_tpr, _ = roc_curve(y_test, baseline_scores)
if_fpr, if_tpr, _ = roc_curve(y_test, -scores_test)

fig, ax = plt.subplots(figsize=(6, 5))
ax.plot(bl_fpr, bl_tpr, color="gray",
        label=f"Baseline — serror_rate > 0.5  (AUC={bl_auc:.3f})")
ax.plot(if_fpr, if_tpr, color="tomato",
        label=f"Isolation Forest              (AUC={if_auc:.3f})")
ax.plot([0, 1], [0, 1], "k--", linewidth=0.8, label="random")
ax.set_xlabel("False Positive Rate  (normal connections falsely flagged)")
ax.set_ylabel("True Positive Rate  (attacks caught)")
ax.set_title("ROC curves — attack detection, test set")
ax.legend(fontsize=8, loc="lower right")
fig.tight_layout()
fig.savefig(PLOT_DIR / "kddcup99_roc_curves.png")
plt.show()


# %%
""" [8] Confusion matrix and per-class summary
The confusion matrix at the chosen threshold translates continuous scores into
four concrete outcomes:
  TP — attack correctly flagged (security win)
  FP — normal connection falsely alarmed (analyst burden / alert fatigue)
  FN — attack missed (security risk)
  TN — normal connection correctly cleared

With 80%+ attacks in the test set, raw accuracy is a misleading metric —
even flagging everything gives >80% accuracy. Precision and recall on the
attack class (positive class) are the meaningful evaluation criteria.
"""
cm = confusion_matrix(y_test, if_pred_test)
disp = ConfusionMatrixDisplay(cm, display_labels=["normal", "attack"])

fig, ax = plt.subplots(figsize=(5, 4))
disp.plot(ax=ax, colorbar=False, cmap="Blues")
ax.set_title(
    f"Confusion matrix — IF at contamination={CONTAMINATION}\n"
    f"Precision={if_prec:.2f}  Recall={if_rec:.2f}  F1={if_f1:.2f}"
)
fig.tight_layout()
fig.savefig(PLOT_DIR / "kddcup99_confusion_matrix.png")
plt.show()

print(f"\nConfusion matrix breakdown at contamination={CONTAMINATION}:")
print(f"  TP — attacks caught   : {cm[1,1]:6,}  ({cm[1,1]/cm[1].sum()*100:.1f}% of attacks)")
print(f"  FN — attacks missed   : {cm[1,0]:6,}  ({cm[1,0]/cm[1].sum()*100:.1f}% of attacks)")
print(f"  FP — false alarms     : {cm[0,1]:6,}  ({cm[0,1]/cm[0].sum()*100:.1f}% of normals)")
print(f"  TN — normals cleared  : {cm[0,0]:6,}  ({cm[0,0]/cm[0].sum()*100:.1f}% of normals)")


# %%
""" [9] Per-attack-type recall
Breaking down recall by attack type reveals what IF catches vs misses.

Expected pattern:
  neptune (SYN flood)  — near-100% recall; serror_rate is a near-perfect discriminator
  smurf (ICMP flood)   — high recall via src_bytes=1,032 and protocol signature
  satan/ipsweep/nmap   — probe attacks; moderate recall via diff_srv_rate
  R2L/U2R attacks      — rarest and hardest: they mimic normal sessions and produce
                         no abnormal error rates or byte volumes

The attack types with low recall are where a more expressive model or additional
features (e.g., payload content, categorical flag/service) would be needed.
"""
df_test["if_flag"] = if_pred_test
attack_breakdown = (
    df_test[df_test["is_attack"] == 1]
    .groupby("labels")
    .agg(total=("is_attack", "count"), flagged=("if_flag", "sum"))
    .assign(recall=lambda d: d["flagged"] / d["total"])
    .sort_values("total", ascending=False)
)
print("\nPer-attack-type recall (test set):")
print(attack_breakdown.to_string())

fig, ax = plt.subplots(figsize=(9, 5))
colors_bar = ["tomato" if r >= 0.5 else "steelblue"
              for r in attack_breakdown["recall"]]
ax.barh(attack_breakdown.index[::-1], attack_breakdown["recall"][::-1],
        color=colors_bar[::-1])
ax.axvline(0.5, color="gray", linestyle="--", linewidth=1, label="50% recall")
ax.set_xlabel("Recall (fraction of attacks caught)")
ax.set_title("Isolation Forest — recall by attack type")
ax.set_xlim(0, 1.05)
ax.legend(fontsize=8)
for i, (_, row) in enumerate(attack_breakdown.iloc[::-1].iterrows()):
    ax.text(row["recall"] + 0.01, i,
            f"  {row['flagged']:.0f}/{row['total']:.0f}", va="center", fontsize=7)
fig.tight_layout()
fig.savefig(PLOT_DIR / "kddcup99_recall_by_attack_type.png")
plt.show()


# %%
""" [10] Per-feature AUC — interpretability
Which features contribute most to the IF's detection power?

For each feature in isolation: train a separate IF on that single feature
(normal training rows only) and measure ROC-AUC on the test set. Higher AUC
means the feature alone carries more attack signal.

This is a single-feature ablation rather than a true feature importance
(which would require permutation tests). It answers: "if we could only watch
one signal, which should it be?" The answer from EDA was serror_rate;
this plot confirms or challenges that prior.
"""
feature_aucs = {}
for i, feat in enumerate(FEATURES):
    iso_single = IsolationForest(n_estimators=100, random_state=RANDOM_STATE)
    iso_single.fit(X_tr_s[:, [i]])
    s = iso_single.score_samples(X_test_s[:, [i]])
    feature_aucs[feat] = roc_auc_score(y_test, -s)

feat_auc_series = pd.Series(feature_aucs).sort_values()

fig, ax = plt.subplots(figsize=(8, 5))
colors_bar = ["tomato" if v >= 0.7 else "steelblue" for v in feat_auc_series]
ax.barh(feat_auc_series.index, feat_auc_series.values, color=colors_bar)
ax.axvline(0.5, color="gray", linestyle="--", linewidth=1, label="random (AUC=0.5)")
ax.axvline(if_auc, color="red", linestyle=":", linewidth=1.2,
           label=f"full IF (AUC={if_auc:.3f})")
ax.set_xlabel("ROC-AUC (single-feature Isolation Forest)")
ax.set_title("Per-feature detection power — single-feature IF AUC")
ax.set_xlim(0.4, 1.05)
ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig(PLOT_DIR / "kddcup99_feature_auc.png")
plt.show()

print("\nPer-feature AUC (single-feature IF):")
for feat, auc in feat_auc_series.sort_values(ascending=False).items():
    print(f"  {feat:<22s}  {auc:.3f}")
print(f"\n  Full IF (all features):  {if_auc:.3f}")


# %%
""" [11] Contamination sensitivity
The contamination parameter sets where the IF threshold falls on the training
score distribution. Changing it does not retrain the model — only the operating
point changes, creating a precision / recall trade-off:
  low contamination  → strict threshold → high precision, lower recall
  high contamination → lenient threshold → higher recall, lower precision

This plot shows how precision and recall move as contamination is varied, and
marks the value that maximises F1 on this test set. In practice, the operating
point should be chosen using a validation set, not the test set.
"""
contamination_levels = np.linspace(0.001, 0.1, 100)
precisions, recalls, f1s = [], [], []

for c in contamination_levels:
    thr_c = np.percentile(scores_train, c * 100)
    pred_c = (scores_test < thr_c).astype(int)
    p, r, f, _ = precision_recall_fscore_support(
        y_test, pred_c, average="binary", zero_division=0,
    )
    precisions.append(p)
    recalls.append(r)
    f1s.append(f)

best_idx = int(np.argmax(f1s))
best_c   = contamination_levels[best_idx]

fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(contamination_levels * 100, precisions, label="Precision", color="steelblue")
ax.plot(contamination_levels * 100, recalls,    label="Recall",    color="tomato")
ax.plot(contamination_levels * 100, f1s,        label="F1",        color="purple", linewidth=1.5)
ax.axvline(CONTAMINATION * 100, color="gray",  linestyle="--", linewidth=1,
           label=f"default ({CONTAMINATION*100:.0f}%)")
ax.axvline(best_c * 100, color="purple", linestyle=":", linewidth=1.2,
           label=f"best F1={f1s[best_idx]:.3f} at {best_c*100:.0f}%")
ax.set_xlabel("Contamination (%)")
ax.set_ylabel("Score")
ax.set_title("Precision / Recall / F1 vs contamination threshold", fontsize=15)
ax.legend(fontsize=11)
fig.tight_layout()
fig.savefig(PLOT_DIR / "kddcup99_contamination_sensitivity.png")
plt.show()

print(f"\nContamination sweep:")
print(f"  Default  ({CONTAMINATION*100:.0f}%): "
      f"P={precisions[np.argmin(np.abs(contamination_levels - CONTAMINATION))]:.3f}  "
      f"R={recalls[np.argmin(np.abs(contamination_levels - CONTAMINATION))]:.3f}  "
      f"F1={f1s[np.argmin(np.abs(contamination_levels - CONTAMINATION))]:.3f}")
print(f"  Best F1  ({best_c*100:.0f}%):  "
      f"P={precisions[best_idx]:.3f}  R={recalls[best_idx]:.3f}  F1={f1s[best_idx]:.3f}")

# %% [12]
