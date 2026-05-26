"""Which class-imbalance strategy best recovers recall for rare attack categories
when classifying network connections into specific attack types?

Five-class problem: Normal / DoS / Probe / R2L / U2R (official KDD competition taxonomy).
Class sizes span four orders of magnitude — DoS connections outnumber R2L by ~60:1.
Four strategies are compared: no correction, cost-sensitive weights, random
undersampling, and random oversampling (bootstrap). An optional cell demonstrates
SMOTE synthetic oversampling (requires imbalanced-learn; the rest of the script runs
without it).
"""

# %%
""" [1] Imports & config"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from cases.sklearn_kddcup99.kdd_utils import (
    NUMERIC_FEATURES, ATTACK_CATEGORY, load_kddcup99, make_pipe,
)
from cases.utils.imbalance import random_undersample, random_oversample

try:
    from imblearn.over_sampling import SMOTE
    SMOTE_AVAILABLE = True
except ImportError:
    SMOTE_AVAILABLE = False

PLOT_DIR = Path(__file__).parent / "plots"
PLOT_DIR.mkdir(exist_ok=True)

RANDOM_STATE     = 42
TEST_SIZE        = 0.25
VAL_SIZE         = 0.20   # fraction of non-test rows held out for strategy selection
RESAMPLE_TARGET  = 5_000  # per-class target size for both under- and oversampling

CAT_FEATURES = ["protocol_type", "flag", "service"]


# %%
""" [2] Load, decode, feature engineering, and attack category mapping
Same decode pipeline as previous KDD scripts. After decoding and casting:
  - Three skewed columns receive log₁₀(x+1) transforms (see kdd_01 EDA)
  - Each raw label is mapped to its attack category; rows with no matching
    label (unmapped attacks) are dropped to keep classes clean

The five resulting categories encode the competition's official threat taxonomy:
  Normal — legitimate traffic
  DoS    — flood attacks (neptune/SYN, smurf/ICMP, teardrop/fragmentation)
  Probe  — reconnaissance scans (ipsweep, portsweep, satan, nmap)
  R2L    — remote-to-local exploits (warezclient, guess_passwd, imap, …)
  U2R    — user-to-root privilege escalation (buffer_overflow, rootkit, …)
"""
df = load_kddcup99(add_is_attack=False, random_state=RANDOM_STATE)
for col in ["src_bytes", "dst_bytes", "duration"]:
    df[f"log_{col}"] = np.log10(df[col] + 1)

df["attack_category"] = df["labels"].map(ATTACK_CATEGORY)
n_unmapped = df["attack_category"].isna().sum()
if n_unmapped:
    print(f"Unmapped labels dropped: {n_unmapped}  "
          f"({df.loc[df['attack_category'].isna(), 'labels'].value_counts().to_dict()})")
df = df.dropna(subset=["attack_category"]).copy()

# Drop categories with fewer than 2 samples — too sparse for any stratified split.
# U2R has exactly 1 sample (buffer_overflow.) in the SA subset.
class_counts_raw = df["attack_category"].value_counts()
singleton_classes = class_counts_raw[class_counts_raw < 2].index.tolist()
if singleton_classes:
    n_dropped = df["attack_category"].isin(singleton_classes).sum()
    print(f"Singleton classes dropped ({n_dropped} rows): {singleton_classes}")
    df = df[~df["attack_category"].isin(singleton_classes)].copy()

print(f"Dataset after mapping: {len(df):,} rows")
print(df["attack_category"].value_counts().to_string())


# %%
""" [3] Category distribution — the four-order-of-magnitude imbalance
Bar chart on a log scale makes all five categories readable at once.

DoS connections outnumber R2L by roughly 60:1. U2R may be entirely absent from
the SA subset — the evaluation partition used here did not guarantee all attack
types are represented. Strategies that ignore this imbalance will produce models
with strong DoS/Probe recall and near-zero R2L/U2R recall.
"""
cat_counts = df["attack_category"].value_counts()

fig, ax = plt.subplots(figsize=(8, 4))
bar_colors = {
    "Normal": "steelblue", "DoS": "tomato", "Probe": "darkorange",
    "R2L": "purple", "U2R": "crimson",
}
ax.bar(cat_counts.index, cat_counts.values,
       color=[bar_colors.get(c, "gray") for c in cat_counts.index])
ax.set_yscale("log")
ax.set_ylabel("Number of connections (log scale)")
ax.set_title("Attack category distribution — SA subset\n(log scale; DoS/R2L ratio ≈ 60:1)")
for i, (cat, n) in enumerate(cat_counts.items()):
    ax.text(i, n * 1.2, f"{n:,}", ha="center", va="bottom", fontsize=8)
fig.tight_layout()
path = PLOT_DIR / "kddcup99_category_distribution.png"
fig.savefig(path)
plt.show()
print(f"Saved: {path}")
print(f"\nSamllest : largest class ratio = 1 : {cat_counts.max() // cat_counts.min()}")


# %%
""" [4] Three-way train / val / test split
Test set is held out completely until [8]. Validation set is used in [7] to
compare strategies and select the best. Only the winning strategy is then
evaluated on the test set exactly once — mirroring the pattern in kdd_04.

First split is stratified on attack_category. The second (train vs val) is not:
R2L has only 8 total samples, too sparse for two consecutive stratified splits.
"""
CATEGORIES = cat_counts.index.tolist()

X_raw = df[NUMERIC_FEATURES + CAT_FEATURES]
y_cat = df["attack_category"]

X_tv, X_test, y_tv, y_test = train_test_split(
    X_raw, y_cat, test_size=TEST_SIZE, stratify=y_cat, random_state=RANDOM_STATE,
)
X_tr_raw, X_val, y_train, y_val = train_test_split(
    X_tv, y_tv, test_size=VAL_SIZE, random_state=RANDOM_STATE,
)

df_train = X_tr_raw.copy()
df_train["attack_category"] = y_train.values

df_tv = X_tv.copy()
df_tv["attack_category"] = y_tv.values

print(f"Train: {len(df_train):,}  |  Val: {len(X_val):,}  |  Test: {len(X_test):,}")
print("Category counts — train:")
print(y_train.value_counts().to_string())


# %%
""" [5] Preprocessing pipeline factory + baseline classifier
A factory function returns a fresh Pipeline (independent preprocessor per model),
preventing state sharing across strategies.

The baseline DummyClassifier(stratified) draws predictions from the training class
distribution. Its macro F1 approaches zero on rare classes — a concrete lower bound
that highlights what each resampling strategy must beat.

results dict collects each strategy's predictions and report for comparison in [10].
"""
results = {}

dummy = make_pipe(DummyClassifier(strategy="stratified", random_state=RANDOM_STATE),
                  cat_features=CAT_FEATURES)
dummy.fit(X_tr_raw, y_train)
y_pred_dummy = dummy.predict(X_val)
rep_dummy = classification_report(
    y_val, y_pred_dummy, labels=CATEGORIES, output_dict=True, zero_division=0,
)
results["Dummy (stratified)"] = {"y_pred": y_pred_dummy, "report": rep_dummy,
                                  "n_train": len(y_train)}
print(f"Dummy (stratified):  macro F1={rep_dummy['macro avg']['f1-score']:.3f}")
print(classification_report(y_val, y_pred_dummy, labels=CATEGORIES, zero_division=0))


# %%
""" [6] Strategy comparison — fit all resampling approaches and collect results
Each strategy is a (X_train, y_train, pipeline) tuple; one loop does fit/predict/store.

  RF (balanced)  — no resampling; class_weight='balanced' upweights rare-class splits
                   inside the Gini criterion (~60× for R2L vs DoS). Fastest, leak-free.
  Undersampling  — majority classes trimmed to RESAMPLE_TARGET; minority kept as-is.
                   Risk: discards real DoS diversity (neptune variants may vanish).
  Oversampling   — minority classes bootstrapped up to RESAMPLE_TARGET; majority kept.
                   Risk: duplicating R2L rows causes memorisation over generalisation.

SMOTE (optional, requires imbalanced-learn) appends to the registry before the loop:
  new minority examples are interpolated between real neighbours rather than copied,
  but sparse classes like U2R may lack enough points for meaningful interpolation.
"""
RF = lambda **kw: RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE,
                                          n_jobs=-1, **kw)

strategies: dict[str, tuple] = {
    "RF (class_weight=balanced)": (
        X_tr_raw, y_train,
        make_pipe(RF(class_weight="balanced"), cat_features=CAT_FEATURES),
    ),
    "Undersampling + RF": (
        *random_undersample(X_tr_raw, y_train, RESAMPLE_TARGET, RANDOM_STATE),
        make_pipe(RF(), cat_features=CAT_FEATURES),
    ),
    "Oversampling + RF": (
        *random_oversample(X_tr_raw, y_train, RESAMPLE_TARGET, RANDOM_STATE),
        make_pipe(RF(), cat_features=CAT_FEATURES),
    ),
}

if SMOTE_AVAILABLE:
    from imblearn.pipeline import Pipeline as ImbPipeline
    k_neighbors = min(5, y_train.value_counts().min() - 1)
    strategies["SMOTE + RF"] = (
        X_tr_raw, y_train,
        ImbPipeline([
            ("pre", ColumnTransformer([
                ("num", StandardScaler(), NUMERIC_FEATURES),
                ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                 CAT_FEATURES),
            ], remainder="drop")),
            ("smote", SMOTE(random_state=RANDOM_STATE, k_neighbors=k_neighbors)),
            ("clf",   RF()),
        ]),
    )
else:
    print("imbalanced-learn not installed — skipping SMOTE.  pip install imbalanced-learn")

for name, (X_tr, y_tr, pipe) in strategies.items():
    pipe.fit(X_tr, y_tr)
    y_pred = pipe.predict(X_val)
    rep = classification_report(y_val, y_pred, labels=CATEGORIES,
                                output_dict=True, zero_division=0)
    results[name] = {"y_pred": y_pred, "report": rep, "n_train": len(y_tr)}
    print(f"{name}:  n_train={len(y_tr):,}  "
          f"macro F1={rep['macro avg']['f1-score']:.3f}  "
          f"R2L recall={rep.get('R2L', {}).get('recall', 0):.3f}")


# %%
""" [7] Strategy comparison — macro F1 table + per-category recall heatmap
Two views of the same results, all measured on the validation set:
  Table — macro F1 and per-class recall for every strategy side-by-side
  Heatmap — per-category recall as a colour grid; green = good recall, red = zero

The heatmap directly answers the research question: which strategy most evenly
distributes recall across all categories, not just the majority DoS class.
The best strategy by macro F1 is then refitted and evaluated on test in [8].
"""
present_cats = [c for c in CATEGORIES if c in y_val.unique()]

recall_df = pd.DataFrame(
    {name: {cat: res["report"].get(cat, {}).get("recall", 0) for cat in present_cats}
     for name, res in results.items()}
)
macro_f1s = pd.Series(
    {name: res["report"]["macro avg"]["f1-score"] for name, res in results.items()},
    name="macro F1",
)

print("Per-category recall by strategy:")
print(recall_df.round(3).to_string())
print(f"\nMacro F1:\n{macro_f1s.round(3).to_string()}")

fig, ax = plt.subplots(figsize=(max(8, len(results) * 2), 4))
im = ax.imshow(recall_df.values, vmin=0, vmax=1, cmap="RdYlGn", aspect="auto")
ax.set_xticks(range(len(recall_df.columns)))
ax.set_xticklabels(recall_df.columns, rotation=20, ha="right", fontsize=9)
ax.set_yticks(range(len(recall_df.index)))
ax.set_yticklabels(recall_df.index, fontsize=9)
plt.colorbar(im, ax=ax, label="Recall")
for i in range(len(recall_df.index)):
    for j in range(len(recall_df.columns)):
        ax.text(j, i, f"{recall_df.iloc[i, j]:.2f}",
                ha="center", va="center", fontsize=9,
                color="black" if recall_df.iloc[i, j] > 0.3 else "white")
ax.set_title("Per-category recall by resampling strategy (validation set)")
fig.tight_layout()
path = PLOT_DIR / "kddcup99_multiclass_recall_heatmap.png"
fig.savefig(path)
plt.show()
print(f"Saved: {path}")


# %%
""" [8] Final evaluation — best strategy refitted on train+val, evaluated on test
Validation comparison in [7] selected the best strategy by macro F1. That strategy
is now refitted on the combined train+val pool and evaluated exactly once on the
held-out test set — the only time test labels are examined.

Off-diagonal cells reveal which categories are most confused:
  Probe / Normal — occasional; port scans can resemble normal browsing patterns
  R2L → Normal — the hardest failure: R2L sessions mimic interactive sessions with
                  no abnormal error rates or byte volumes
"""
best_name = macro_f1s.idxmax()
print(f"Best strategy by val macro F1: {best_name}  ({macro_f1s[best_name]:.3f})")
print("Refitting on full train+val, then evaluating once on held-out test set...")

if best_name == "RF (class_weight=balanced)":
    pipe_final = make_pipe(
        RandomForestClassifier(n_estimators=200, class_weight="balanced",
                               random_state=RANDOM_STATE, n_jobs=-1),
        cat_features=CAT_FEATURES,
    )
    pipe_final.fit(X_tv, y_tv)
elif best_name == "Undersampling + RF":
    parts_final = []
    for cat, sub in df_tv.groupby("attack_category"):
        parts_final.append(sub.sample(n=min(len(sub), RESAMPLE_TARGET),
                                      random_state=RANDOM_STATE))
    df_final = pd.concat(parts_final).sample(frac=1, random_state=RANDOM_STATE)
    pipe_final = make_pipe(
        RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE, n_jobs=-1),
        cat_features=CAT_FEATURES,
    )
    pipe_final.fit(df_final[NUMERIC_FEATURES + CAT_FEATURES], df_final["attack_category"])
elif best_name == "Oversampling + RF":
    parts_final = []
    for cat, sub in df_tv.groupby("attack_category"):
        parts_final.append(sub.sample(n=max(len(sub), RESAMPLE_TARGET),
                                      replace=True, random_state=RANDOM_STATE))
    df_final = pd.concat(parts_final).sample(frac=1, random_state=RANDOM_STATE)
    pipe_final = make_pipe(
        RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE, n_jobs=-1),
        cat_features=CAT_FEATURES,
    )
    pipe_final.fit(df_final[NUMERIC_FEATURES + CAT_FEATURES], df_final["attack_category"])
elif best_name == "SMOTE + RF":
    from imblearn.pipeline import Pipeline as ImbPipeline
    k_neighbors_tv = min(5, y_tv.value_counts().min() - 1)
    pipe_final = ImbPipeline([
        ("pre", ColumnTransformer([
            ("num", StandardScaler(), NUMERIC_FEATURES),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False),
             CAT_FEATURES),
        ], remainder="drop")),
        ("smote", SMOTE(random_state=RANDOM_STATE, k_neighbors=k_neighbors_tv)),
        ("clf",   RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE,
                                         n_jobs=-1)),
    ])
    pipe_final.fit(X_tv, y_tv)
else:  # Dummy
    pipe_final = make_pipe(
        DummyClassifier(strategy="stratified", random_state=RANDOM_STATE),
        cat_features=CAT_FEATURES,
    )
    pipe_final.fit(X_tv, y_tv)

y_pred_final = pipe_final.predict(X_test)
present_test_cats = [c for c in CATEGORIES if c in y_test.unique()]

fig, ax = plt.subplots(figsize=(7, 6))
ConfusionMatrixDisplay.from_predictions(
    y_test, y_pred_final,
    labels=present_test_cats,
    ax=ax, colorbar=False, cmap="Blues",
    xticks_rotation=30,
)
ax.set_title(f"Confusion matrix — {best_name}\n(held-out test set)")
fig.tight_layout()
path = PLOT_DIR / "kddcup99_multiclass_confusion.png"
fig.savefig(path)
plt.show()
rep_final = classification_report(
    y_test, y_pred_final, labels=present_test_cats, output_dict=True, zero_division=0,
)
print(f"Saved: {path}")
print(f"\nBest strategy: {best_name}")
print(f"  macro F1 — val: {macro_f1s[best_name]:.3f}  "
      f"test: {rep_final['macro avg']['f1-score']:.3f}")
print(classification_report(y_test, y_pred_final, labels=present_test_cats,
                             zero_division=0))

# %%
