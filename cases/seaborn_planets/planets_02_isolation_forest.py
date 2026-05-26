"""Isolation Forest anomaly detection: which confirmed exoplanets are statistical outliers in (log_orbital_period, log_distance) space?

Every row in the dataset is a confirmed discovery — there are no ground-truth anomaly labels. Isolation Forest is therefore used purely as a multivariate outlier detector:
it identifies points that are isolated in the joint (period, distance) feature space, not extreme on any single axis. Evaluation is qualitative — do the flagged planets match what domain knowledge says should be unusual? — rather than precision/recall against a labelled positive class."""

# %%
""" [1] Imports & config"""
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

PLOT_DIR = Path(__file__).parent / "plots"
PLOT_DIR.mkdir(exist_ok=True)

RANDOM_STATE  = 42
CONTAMINATION = 0.05    # assumed fraction of anomalies in the population
TEST_SIZE     = 0.20
FEATURES      = ["log_period", "log_distance"]


# %%
""" [2] Load & engineer features
Use orbital_period and distance — the two continuous quantities with the least
missingness (4% and 22%). Log₁₀-transform both to handle the 7-order-of-magnitude
skew identified in EDA. Drop rows missing either feature (~780 rows remain, 75%
of the de-duplicated dataset). mass is left out to preserve sample size; it can be
added as a third feature at the cost of halving the dataset to RV planets only.
"""
df_raw = sns.load_dataset("planets")
df = df_raw.drop_duplicates().copy()
df["log_period"]   = np.log10(df["orbital_period"])
df["log_distance"] = np.log10(df["distance"])

df_model = df.dropna(subset=FEATURES).reset_index(drop=True)
print(f"Rows available for modelling: {len(df_model)}  "
      f"({len(df_model) / len(df) * 100:.1f}% of de-duplicated dataset)")
print(df_model[FEATURES].describe().round(3))


# %%
""" [3] Train / test split
80/20 hold-out split. Isolation Forest is unsupervised — no target labels to leak.
The split serves two purposes:
  1. Fit the StandardScaler on training data only (no test information leaks in).
  2. Compare score distributions on train vs test to detect overfitting to
     training-set quirks.
Final anomaly scores are computed on the full dataset after training.
"""
X_all = df_model[FEATURES].values
idx_all = np.arange(len(df_model))
idx_train, idx_test = train_test_split(idx_all, test_size=TEST_SIZE, random_state=RANDOM_STATE)

scaler  = StandardScaler().fit(X_all[idx_train])
X_train = scaler.transform(X_all[idx_train])
X_test  = scaler.transform(X_all[idx_test])
X_all_s = scaler.transform(X_all)

print(f"Train: {len(X_train)} rows  |  Test: {len(X_test)} rows")


# %%
""" [4] Baseline: per-feature Z-score detector
Simplest anomaly detector: flag a point if its maximum per-feature |z-score|
exceeds Z_THRESH (univariate thresholding).

Strength  — fast, parameter-free, fully interpretable.
Weakness  — blind to the joint distribution: a planet with a moderately unusual
period AND a moderately unusual distance is missed even if that combination is rare.
This blind spot is exactly what Isolation Forest is designed to catch.

Z-scores are computed from training-set statistics (data is already standardised,
so |z| = |x_scaled|).
"""
Z_THRESH = 2.5

z_max_train = np.abs(X_train).max(axis=1)
z_max_all   = np.abs(X_all_s).max(axis=1)

baseline_flag_train = z_max_train > Z_THRESH
baseline_flag_all   = z_max_all   > Z_THRESH

print(f"\nBaseline Z-score  (|z| > {Z_THRESH}):")
print(f"  Train: {baseline_flag_train.sum():3d} / {len(X_train)} flagged  "
      f"({baseline_flag_train.mean() * 100:.1f}%)")
print(f"  All  : {baseline_flag_all.sum():3d} / {len(X_all_s)} flagged  "
      f"({baseline_flag_all.mean() * 100:.1f}%)")


# %%
""" [5] Isolation Forest
IsolationForest builds an ensemble of random trees that partition the feature
space with random splits. Anomalous points — in sparse, extreme regions — require
fewer splits to isolate and receive a lower anomaly score.

Key parameters:
  n_estimators=200: doubles the default for stable scores given the small n.
  contamination=CONTAMINATION: sets the score threshold that separates −1/+1
    labels. Does NOT change the raw scores; only affects predict().
  max_samples='auto': uses min(256, n_train) sub-samples per tree.

score_samples() convention: lower (more negative) = more anomalous.
predict() returns +1 (normal) or −1 (anomaly).
"""
iso = IsolationForest(
    n_estimators=200,
    contamination=CONTAMINATION,
    max_samples="auto",
    random_state=RANDOM_STATE,
)
iso.fit(X_train)

scores_train = iso.score_samples(X_train)
scores_test  = iso.score_samples(X_test)
scores_all   = iso.score_samples(X_all_s)
pred_all     = iso.predict(X_all_s)

n_anomaly_train = (iso.predict(X_train) == -1).sum()
n_anomaly_test  = (iso.predict(X_test)  == -1).sum()

print(f"\nIsolation Forest  (contamination={CONTAMINATION}):")
print(f"  Train: {n_anomaly_train:3d} / {len(X_train)} flagged  "
      f"({n_anomaly_train / len(X_train) * 100:.1f}%)")
print(f"  Test : {n_anomaly_test:3d} / {len(X_test)} flagged  "
      f"({n_anomaly_test  / len(X_test)  * 100:.1f}%)")


# %%
""" [6] Score distribution — train vs test consistency
A stable model should produce similar score distributions on both splits.
The decision threshold (red line) is set at the contamination percentile of the
training scores and applied to the test set — if the distributions diverge, the
threshold would misclassify a different fraction on the test set.
"""
threshold = np.percentile(scores_train, CONTAMINATION * 100)

fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True, sharex=True)
for ax, scores, label in zip(
        axes, [scores_train, scores_test], ["Train", "Test"]):
    ax.hist(scores, bins=40, edgecolor="none")
    ax.axvline(threshold, color="red", linestyle="--", linewidth=1.2,
               label=f"threshold (train {CONTAMINATION*100:.0f}th pct)")
    ax.set_xlabel("Anomaly score  (lower = more anomalous)")
    ax.set_title(f"{label}  (n={len(scores)})")
    ax.legend(fontsize=8)

fig.suptitle("Isolation Forest — score distribution, train vs test")
fig.tight_layout()
fig.savefig(PLOT_DIR / "planets_if_score_dist.png")
plt.show()

above_train = (scores_train >= threshold).mean() * 100
above_test  = (scores_test  >= threshold).mean() * 100
print(f"\nFraction above threshold — train: {above_train:.1f}%  test: {above_test:.1f}%")
print("(Should both be close to 1 − contamination = "
      f"{(1 - CONTAMINATION) * 100:.0f}%)")


# %%
""" [7] Baseline vs Isolation Forest — agreement
Compare which planets each detector flags, using the same contamination fraction
as the threshold for both. This 2×2 breakdown exposes the key difference:

  IF-only detections  ← the multivariate catch: not extreme in any single feature
                        but isolated in the joint (period, distance) space
  Z-only detections   ← extreme on one axis but inside the IF anomaly-score cloud
  Both                ← highest-confidence anomalies; both methods agree

IF-only planets are the most pedagogically interesting — they motivate why
multivariate anomaly detection outperforms per-feature thresholds.
"""
z_thresh_matched = np.percentile(z_max_train, (1 - CONTAMINATION) * 100)
baseline_flag_train_matched = z_max_train > z_thresh_matched
iso_flag_train = iso.predict(X_train) == -1

agree_normal = (~iso_flag_train & ~baseline_flag_train_matched).sum()
if_only      = ( iso_flag_train & ~baseline_flag_train_matched).sum()
z_only       = (~iso_flag_train &  baseline_flag_train_matched).sum()
both         = ( iso_flag_train &  baseline_flag_train_matched).sum()
n_tr = len(X_train)

print(f"\nDetector agreement on training set  "
      f"(both thresholds matched to {CONTAMINATION*100:.0f}% contamination):")
print(f"  Both normal     : {agree_normal:4d}  ({agree_normal / n_tr * 100:.1f}%)")
print(f"  IF only         : {if_only:4d}  ({if_only      / n_tr * 100:.1f}%)  "
      "← multivariate-only catch")
print(f"  Z-score only    : {z_only:4d}  ({z_only        / n_tr * 100:.1f}%)")
print(f"  Both flag       : {both:4d}  ({both            / n_tr * 100:.1f}%)  "
      "← highest-confidence anomalies")


# %%
""" [8] Anomaly map — scores in feature space
All planets coloured by their Isolation Forest anomaly score (blue = normal,
red = anomalous). Planets flagged as anomalies are additionally marked with ×.
The colour gradient shows the continuous scoring surface — not a hard boundary —
and highlights how sparse regions at extreme period or distance receive low scores
even when a single-axis z-score would not catch them.
"""
anomaly_mask = pred_all == -1

fig, ax = plt.subplots(figsize=(9, 6))
sc = ax.scatter(
    df_model["log_period"], df_model["log_distance"],
    c=scores_all, cmap="RdYlBu",
    vmin=scores_all.min(), vmax=scores_all.max(),
    s=15, alpha=0.75,
)
plt.colorbar(sc, ax=ax, label="Anomaly score  (lower = more anomalous)")
ax.scatter(
    df_model.loc[anomaly_mask, "log_period"],
    df_model.loc[anomaly_mask, "log_distance"],
    marker="x", color="black", s=40, linewidths=0.8,
    label=f"Flagged  (n={anomaly_mask.sum()}, contamination={CONTAMINATION})",
)
ax.set_xlabel("log₁₀(orbital period, days)")
ax.set_ylabel("log₁₀(distance, parsecs)")
ax.set_title("Exoplanet anomaly map — Isolation Forest")
ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig(PLOT_DIR / "planets_if_anomaly_map.png")
plt.show()


# %%
""" [9] Top anomalies — who got flagged?
The most anomalous planets ranked by raw anomaly score.
Cross-referenced with detection method and original physical values to validate
that the algorithm flags scientifically meaningful outliers, not artefacts.

Expected top candidates from EDA:
  Imaging planets      — extreme orbital period (HR 8799 system, ~730,000 days)
  Microlensing planets — extreme distance (galactic bulge, ~8,500 pc)
  RV outliers          — unusually long period given their nearby host star

A detection dominated by rare-method planets confirms the model is capturing
genuine physical isolation, not a numeric artefact of the log transform.
"""
df_model["if_score"] = scores_all
df_model["if_flag"]  = pred_all

top = (
    df_model[df_model["if_flag"] == -1]
    .sort_values("if_score")
    [["if_score", "method", "orbital_period", "distance", "year", "number"]]
    .head(15)
)
print("\nTop 15 anomalies by Isolation Forest score:")
print(top.round(3).to_string())

print("\nAnomalies by detection method:")
method_breakdown = (
    df_model[df_model["if_flag"] == -1]["method"].value_counts()
)
total_by_method = df_model["method"].value_counts()
for method, n_flagged in method_breakdown.items():
    n_total = total_by_method[method]
    print(f"  {method:<35s} {n_flagged:3d} / {n_total:3d} "
          f"({n_flagged / n_total * 100:.1f}%)")


# %%
""" [10] Sensitivity: contamination parameter
The contamination parameter only shifts the score threshold — it does NOT retrain
the model. This analysis answers two questions:
  1. How many planets get flagged as we relax the threshold?  (count curve)
  2. Which planets are 'core anomalies' flagged even at the most stringent level?

Core anomalies (flagged at 1% contamination) are the most robustly anomalous;
borderline anomalies only appear at lenient thresholds and warrant extra scrutiny.
"""
contamination_levels = [0.01, 0.02, 0.05, 0.10, 0.15]
flagged_counts = []
for c in contamination_levels:
    thr_c = np.percentile(scores_train, c * 100)
    flagged_counts.append((scores_all < thr_c).sum())

print("\nFlagged count by contamination level:")
for c, n in zip(contamination_levels, flagged_counts):
    print(f"  {c*100:5.1f}%  →  {n:3d} planets")

# Core anomalies: flagged even at the strictest level (1%)
core_threshold = np.percentile(scores_train, contamination_levels[0] * 100)
core_mask = scores_all < core_threshold
print(f"\nCore anomalies (top {contamination_levels[0]*100:.0f}%,"
      f" n={core_mask.sum()}):")
print(
    df_model[core_mask]
    .sort_values("if_score")
    [["if_score", "method", "orbital_period", "distance", "year"]]
    .round(3).to_string()
)

fig, ax = plt.subplots(figsize=(7, 4))
ax.plot([c * 100 for c in contamination_levels], flagged_counts, marker="o")
for c, n in zip(contamination_levels, flagged_counts):
    ax.annotate(str(n), (c * 100, n), textcoords="offset points",
                xytext=(4, 4), fontsize=8)
ax.set_xlabel("Contamination (%)")
ax.set_ylabel("Planets flagged as anomalies")
ax.set_title("Sensitivity to contamination threshold")
ax.axvline(CONTAMINATION * 100, color="red", linestyle="--",
           linewidth=1, label=f"default ({CONTAMINATION*100:.0f}%)")
ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig(PLOT_DIR / "planets_if_contamination_sensitivity.png")
plt.show()
