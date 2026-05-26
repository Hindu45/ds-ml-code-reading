"""Research question: How much does each added measurement improve species classification?
purpose: comparison | style: library-optimal | flags: docstring-depth: minimal

Expected: each feature addition improves accuracy; bill_length alone is already well above dummy.
Surprising: bill_depth (not flipper) being the most informative single add — see EDA Simpson's paradox.
"""

# %%
""" [1] Imports & config"""
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

RANDOM_STATE = 42
PLOT_DIR = Path(__file__).parent / "plots"
PLOT_DIR.mkdir(exist_ok=True)

NUMERIC_COLS = ["bill_length_mm", "bill_depth_mm", "flipper_length_mm", "body_mass_g"]

# Comparison criterion: macro F1 on the held-out test set.
# Macro F1 weights all three species equally regardless of class frequency.
def _logreg_pipeline() -> Pipeline:
    return Pipeline([
        ("sc", StandardScaler()),
        ("clf", LogisticRegression(max_iter=500, random_state=RANDOM_STATE))
        ])

CONFIGS = [
    ("Dummy (majority)", 
     ["bill_length_mm"],
     DummyClassifier(strategy="most_frequent", random_state=RANDOM_STATE)),
    ("LogReg — bill_length",
     ["bill_length_mm"],
     _logreg_pipeline()),
    ("LogReg — +flipper",
     ["bill_length_mm", "flipper_length_mm"],_logreg_pipeline()),
    ("LogReg — +bill_depth",
     ["bill_length_mm", "flipper_length_mm", "bill_depth_mm"], 
     _logreg_pipeline()),
    ("LogReg — all 4",
      NUMERIC_COLS,
      _logreg_pipeline()),
]

# %%
""" [2] Load & split
Drop the 2 rows missing all numeric measurements; retain the 9 sex-missing rows
(sex is not used here). Same 60/20/20 split for every configuration — the only
thing changing across configs is which features are passed to the model.
"""
df = sns.load_dataset("penguins").dropna(subset=NUMERIC_COLS).copy()
y = df["species"]
X = df[NUMERIC_COLS]

X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.4, stratify=y, random_state=RANDOM_STATE
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=RANDOM_STATE
)
print(f"Train: {len(X_train)}  Val: {len(X_val)}  Test: {len(X_test)}")
print("Class distribution (val):\n", y_val.value_counts())

# %%
""" [3] Run all configurations
Each model is fit on the training columns it declares, then evaluated on the
validation set. StandardScaler is always fit on train and applied to val —
no leakage even though the feature subsets differ.
"""
results = []
for label, features, model in CONFIGS:
    model.fit(X_train[features], y_train)
    y_pred = model.predict(X_val[features])
    results.append({
        "config": label,
        "n_features": len(features),
        "accuracy": accuracy_score(y_val, y_pred),
        "macro_f1": f1_score(y_val, y_pred, average="macro"),
    })

results_df = pd.DataFrame(results)
print("\n" + results_df.to_string(index=False, float_format="{:.3f}".format))

# %%
""" [4] Metric comparison plot
Bar chart juxtaposes accuracy and macro F1 for every configuration so the
marginal gain of each added feature is immediately visible.
"""
fig, ax = plt.subplots(figsize=(9, 4))

x = range(len(results_df))
width = 0.35
ax.bar([i - width / 2 for i in x], results_df["accuracy"], width, label="Accuracy")
ax.bar([i + width / 2 for i in x], results_df["macro_f1"], width, label="Macro F1")
ax.set_xticks(list(x))
ax.set_xticklabels(results_df["config"], rotation=20, ha="right", fontsize=8)
ax.set_ylabel("Score")
ax.set_title("Accuracy and macro F1 as features are added (validation set)")
ax.set_ylim(0, 1.05)
ax.legend()
fig.tight_layout()
fig.savefig(PLOT_DIR / "penguins_baseline_feature_progression.png")
plt.show()
print(f"Saved: {PLOT_DIR / 'penguins_baseline_feature_progression.png'}")

# %%
""" [5] Confusion matrix — best configuration (test set)
Test set is touched exactly once here. Confusion matrix for the full 4-feature
LogReg model; Adelie/Chinstrap misclassifications are expected given their
overlapping bill dimensions (see EDA).
"""
_, features_best, model_best = CONFIGS[-1]
model_best.fit(X_train[features_best], y_train)
y_pred_best = model_best.predict(X_test[features_best])

fig, ax = plt.subplots(figsize=(5, 4))
ConfusionMatrixDisplay.from_predictions(y_test, y_pred_best, ax=ax, colorbar=False, cmap="Blues")
ax.set_title("Confusion matrix — LogReg all 4 features")
fig.tight_layout()
fig.savefig(PLOT_DIR / "penguins_baseline_confusion.png")
plt.show()
print(f"Best model (all 4 features) — accuracy: {accuracy_score(y_test, y_pred_best):.3f}  macro F1: {f1_score(y_test, y_pred_best, average='macro'):.3f}")
print(f"Saved: {PLOT_DIR / 'penguins_baseline_confusion.png'}")
