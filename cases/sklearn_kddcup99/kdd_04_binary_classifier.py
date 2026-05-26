"""Supervised binary attack detection on KDD Cup 1999: does using ground-truth labels
at training time produce a better intrusion detector than the unsupervised Isolation Forest?

kdd_02 trained on normal connections only. Here the full labeled dataset is used,
turning detection into standard binary classification. The class imbalance
(normals ≈97%, attacks ≈3%) means a classifier that predicts 'normal' for every
row achieves ~97% accuracy — making accuracy a misleading metric. Attack recall
and ROC-AUC guide model selection: missing an attack is the primary failure mode.
"""

# %%
""" [1] Imports & config"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report, roc_auc_score, roc_curve,
    ConfusionMatrixDisplay,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split
from cases.sklearn_kddcup99.kdd_utils import NUMERIC_FEATURES, load_kddcup99, make_pipe

PLOT_DIR = Path(__file__).parent / "plots"
PLOT_DIR.mkdir(exist_ok=True)

RANDOM_STATE = 42
TEST_SIZE    = 0.25

# Same 11 numeric features as kdd_02, plus two categorical features decoded to OHE
CAT_FEATURES = ["protocol_type", "flag"]


# %%
""" [2] Load & decode + feature engineering
Same decode pipeline as kdd_01/kdd_02: every column arrives as bytes objects.
Three right-skewed columns receive log₁₀(x+1) transforms — justified in EDA
where linear-scale histograms showed distributions dominated by a zero spike.

Categorical features (protocol_type, flag) are kept as decoded strings; the
Pipeline preprocessor will one-hot encode them at fit time.
"""
df = load_kddcup99(random_state=RANDOM_STATE)
for col in ["src_bytes", "dst_bytes", "duration"]:
    df[f"log_{col}"] = np.log10(df[col] + 1)

print(f"Dataset: {len(df):,} rows  |  "
      f"normal: {(df['is_attack']==0).sum():,}  "
      f"attack: {(df['is_attack']==1).sum():,}  "
      f"({df['is_attack'].mean()*100:.1f}% attacks)")


# %%
""" [3] Stratified train / test split
Stratified on is_attack to preserve the ~97/3 normal/attack ratio in both halves.
All models are trained on X_train and evaluated once on X_test at the end.

The majority-class baseline accuracy is set by the normal fraction (~97%): any model
predicting 'normal' for every row achieves that accuracy while missing every attack.
"""
X = df[NUMERIC_FEATURES + CAT_FEATURES]
y = df["is_attack"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE,
)

print(f"Train: {len(X_train):,}  |  Test: {len(X_test):,}")
print(f"Attack rate — train: {y_train.mean()*100:.1f}%  test: {y_test.mean()*100:.1f}%")
print(f"Majority-class accuracy ceiling: {max(y_test.mean(), 1-y_test.mean())*100:.1f}%")


# %%
""" [4] Baseline: majority-class dummy classifier
DummyClassifier(most_frequent) always predicts 'normal' — the majority class (~97%).
It achieves ~97% accuracy while producing zero recall on the attack class.

This anchors the evaluation frame: high accuracy on imbalanced data is not evidence
of a useful detector. A model that misses every attack is operationally useless.
"""
dummy_pipe = make_pipe(
    DummyClassifier(strategy="most_frequent", random_state=RANDOM_STATE),
    cat_features=CAT_FEATURES,
)
dummy_pipe.fit(X_train, y_train)
y_pred_dummy = dummy_pipe.predict(X_test)

print("Dummy (most_frequent) on test set:")
print(classification_report(y_test, y_pred_dummy,
                             target_names=["normal", "attack"], digits=3,
                             zero_division=0))


# %%
""" [5] Compare classifiers with 5-fold stratified cross-validation
`cross_val_predict` returns out-of-fold (OOF) predictions — each training row is
scored by a fold that never saw it during training, so model selection does not
touch the held-out test set. The test set is used only once, in cell 8.

The three models use the same folds, making the comparison apples-to-apples.
"""
models = {
    "Logistic Regression": make_pipe(
        LogisticRegression(max_iter=1_000, random_state=RANDOM_STATE),
        cat_features=CAT_FEATURES,
    ),
    "Random Forest": make_pipe(
        RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE, n_jobs=-1),
        cat_features=CAT_FEATURES,
    ),
    "RF (balanced)": make_pipe(
        RandomForestClassifier(n_estimators=200, class_weight="balanced",
                               random_state=RANDOM_STATE, n_jobs=-1),
        cat_features=CAT_FEATURES,
    ),
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

cv_probs:   dict[str, np.ndarray] = {}
cv_preds:   dict[str, np.ndarray] = {}
cv_reports: dict[str, dict]       = {}

for name, pipe in models.items():
    y_prob = cross_val_predict(pipe, X_train, y_train, cv=cv,
                               method="predict_proba", n_jobs=-1)[:, 1]
    y_pred = (y_prob >= 0.5).astype(int)
    rep    = classification_report(y_train, y_pred,
                                   target_names=["normal", "attack"],
                                   output_dict=True)
    cv_probs[name]   = y_prob
    cv_preds[name]   = y_pred
    cv_reports[name] = rep
    print(f"{name:<28s}  acc={rep['accuracy']:.3f}  "
          f"normal-rec={rep['normal']['recall']:.3f}  "
          f"attack-rec={rep['attack']['recall']:.3f}  "
          f"macro-F1={rep['macro avg']['f1-score']:.3f}")


# %%
""" [6] ROC curves — all three classifiers
ROC-AUC measures detection power independent of any fixed threshold.
The dummy classifier outputs a single class, so no ROC curve exists for it.

LR, RF, and RF-balanced should all achieve very high AUC because the numeric
features (serror_rate, flag encoding) are near-perfect discriminators for the
dominant attack type (neptune). The curves mainly differ near the top-left corner,
where a fraction of hard-to-separate normal/attack pairs live.
"""
fig, ax = plt.subplots(figsize=(7, 5))
for name, y_prob in cv_probs.items():
    fpr, tpr, _ = roc_curve(y_train, y_prob)
    auc = roc_auc_score(y_train, y_prob)
    ax.plot(fpr, tpr, label=f"{name}  (AUC={auc:.4f})")
ax.plot([0, 1], [0, 1], "k--", linewidth=0.8, label="random")
ax.set_xlabel("False Positive Rate  (normal connections flagged as attack)")
ax.set_ylabel("True Positive Rate  (attacks caught)")
ax.set_title("ROC curves — out-of-fold predictions on train set")
ax.legend(fontsize=8, loc="lower right")
fig.tight_layout()
path = PLOT_DIR / "kddcup99_binary_roc.png"
fig.savefig(path)
plt.show()
print(f"Saved: {path}")


# %%
""" [7] Metric comparison — accuracy vs macro F1 vs normal recall
Three metrics side-by-side expose the accuracy illusion:
  accuracy      — dominated by the majority attack class; looks good for all models
  macro F1      — averages F1 equally across both classes; penalises models that
                  ignore the minority normal class
  normal recall — directly measures how many normal connections are correctly
                  identified; zero for the dummy despite its high accuracy

The comparison motivates choosing a model and metric that reflect operational goals,
not just overall accuracy.
"""
dummy_report = classification_report(
    y_test, y_pred_dummy, target_names=["normal", "attack"],
    output_dict=True, zero_division=0
)
summary = pd.DataFrame({
    "Dummy": {
        "accuracy":      dummy_report["accuracy"],
        "macro F1":      dummy_report["macro avg"]["f1-score"],
        "normal recall": dummy_report["normal"]["recall"],
    },
    **{
        name: {
            "accuracy":      rep["accuracy"],
            "macro F1":      rep["macro avg"]["f1-score"],
            "normal recall": rep["normal"]["recall"],
        }
        for name, rep in cv_reports.items()
    }
}).T

metrics = ["accuracy", "macro F1", "normal recall"]
x = np.arange(len(summary))
width = 0.25
colors = ["steelblue", "tomato", "seagreen"]

fig, ax = plt.subplots(figsize=(10, 4))
for i, (metric, color) in enumerate(zip(metrics, colors)):
    ax.bar(x + i * width, summary[metric], width, label=metric, color=color, alpha=0.85)
ax.set_xticks(x + width)
ax.set_xticklabels(summary.index, rotation=15, ha="right", fontsize=9)
ax.set_ylabel("Score")
ax.set_ylim(0, 1.08)
ax.set_title("Accuracy vs macro F1 vs normal recall — binary classifier comparison")
ax.legend(fontsize=9)
fig.tight_layout()
path = PLOT_DIR / "kddcup99_binary_metrics.png"
fig.savefig(path)
plt.show()
print(f"Saved: {path}")
print(f"\n{summary.round(3).to_string()}")


# %%
""" [8] Final evaluation — RF (balanced) refitted on full train, evaluated on test
CV in cell 5 selected RF (balanced) as the best model. It is now refitted on the
entire training set (all 5 folds combined) and evaluated exactly once on the
held-out test set — the only time test labels are used for a selected model.
"""
best_name = "RF (balanced)"
best_pipe = models[best_name]
best_pipe.fit(X_train, y_train)
y_pred_test = best_pipe.predict(X_test)

fig, ax = plt.subplots(figsize=(5, 4))
ConfusionMatrixDisplay.from_predictions(
    y_test, y_pred_test,
    display_labels=["normal", "attack"],
    ax=ax, colorbar=False, cmap="Blues",
)
ax.set_title(f"Confusion matrix — {best_name} (held-out test set)")
fig.tight_layout()
path = PLOT_DIR / "kddcup99_binary_confusion.png"
fig.savefig(path)
plt.show()
print(f"Saved: {path}")

rep_test = classification_report(y_test, y_pred_test,
                                  target_names=["normal", "attack"],
                                  output_dict=True)
print(f"\n{best_name} [test]: acc={rep_test['accuracy']:.3f}  "
      f"normal-rec={rep_test['normal']['recall']:.3f}  "
      f"attack-rec={rep_test['attack']['recall']:.3f}  "
      f"macro-F1={rep_test['macro avg']['f1-score']:.3f}")


# %%
""" [9] Feature importances — RF (balanced)
Impurity-based importance from the random forest: the mean decrease in Gini
impurity across all trees when a feature is used for a split.

One-hot encoded flag values appear as individual columns (e.g. flag_S0, flag_SF).
flag_S0 is the near-perfect neptune discriminator (SYN sent, no response received);
expect it to dominate the ranking alongside serror_rate — which captures the same
flood signal continuously on a 0–1 scale.
"""
rf_clf = best_pipe.named_steps["clf"]
ohe    = best_pipe.named_steps["pre"].named_transformers_["cat"]
all_names = NUMERIC_FEATURES + ohe.get_feature_names_out(CAT_FEATURES).tolist()

importances = (
    pd.Series(rf_clf.feature_importances_, index=all_names)
    .sort_values(ascending=False)
)
top20 = importances.head(20)

fig, ax = plt.subplots(figsize=(8, 6))
ax.barh(top20.index[::-1], top20.values[::-1], color="steelblue")
ax.set_xlabel("Mean decrease in impurity (Gini importance)")
ax.set_title(f"Top-20 feature importances — {best_name}")
fig.tight_layout()
path = PLOT_DIR / "kddcup99_binary_importances.png"
fig.savefig(path)
plt.show()
print(f"Saved: {path}")
print("\nTop-10 features:")
for feat, imp in importances.head(10).items():
    print(f"  {feat:<30s}  {imp:.4f}")

# %%
