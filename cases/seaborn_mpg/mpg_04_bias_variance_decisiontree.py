"""Research question: How do model complexity and regularisation affect the bias-variance tradeoff?"""

# %% [1] Imports & config
import numpy as np
import pandas as pd
import seaborn as sns
from pathlib import Path
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import (
    learning_curve, validation_curve,
    cross_val_score, train_test_split, KFold,
)
from sklearn.tree import DecisionTreeRegressor

from cases.utils.model_eval_plots import (
    plot_learning_curve_panels, plot_validation_curve, plot_cv_stability,
)

PLOT_DIR = Path(__file__).parent / "plots"
PLOT_DIR.mkdir(exist_ok=True)

# ── Dataset configuration ─────────────────────────────────────────────────────
# To swap datasets, edit this block only.
# Requirements: seaborn dataset name, numeric feature list, string target column.
# Suggested alternatives: "diamonds" (target="price"), "taxis" (target="fare"),
#   "healthexp" (features=["Year","Spending_USD"], target="Life_Expectancy")
DATASET_SLUG = "mpg"
FEATURES = ["cylinders", "displacement", "horsepower", "weight",
            "acceleration", "model_year"]
TARGET = "mpg"
DROP_NA_COLS = ["horsepower"]   # columns where NaN rows should be dropped
DEPTH_RANGE  = np.arange(1, 16) # max_depth values for the validation curve
# ─────────────────────────────────────────────────────────────────────────────


# %%
""" [2] Load & prepare
Load the seaborn dataset, drop rows with missing values in key columns,
and split into feature matrix X and target vector y.
All downstream cells consume (X, y) — swapping dataset = swapping this cell.
"""
df_raw = sns.load_dataset(DATASET_SLUG)
df = df_raw.dropna(subset=DROP_NA_COLS).copy() if DROP_NA_COLS else df_raw.copy()
X = df[FEATURES].values
y = df[TARGET].values
print(f"Dataset : {DATASET_SLUG}  —  {X.shape[0]} rows × {X.shape[1]} features")
print(f"Target  : {TARGET}  (mean={y.mean():.2f}, std={y.std():.2f})")


# %%
""" [3] Figure 1 — Learning curves: high bias vs high variance
Two panels, each plotting train RMSE and CV RMSE vs. training set size.
  • High-bias panel (max_depth=2): train RMSE rises as n grows (the shallow tree
    cannot memorise a larger set); CV RMSE falls modestly but stays high; the gap
    narrows yet remains elevated (~1.5 at n=250). Adding more data yields only
    modest improvement — the bottleneck is model capacity, not data volume.
  • High-variance panel (fully grown tree): train ≈ 0 throughout, CV much higher,
    gap persists. The model memorises whatever it sees but cannot generalise.

x-axis = training set size (independent of the complexity dial).
This figure diagnoses WHAT is wrong. Figure 2 shows WHERE to set complexity.
"""
cv_folds    = 5
tree_shallow = DecisionTreeRegressor(max_depth=2,    random_state=0)
tree_deep    = DecisionTreeRegressor(max_depth=None, random_state=0)

configs_lc = [
    (tree_shallow, "High bias\n(max_depth=2)"),
    (tree_deep,    "High variance\n(fully grown tree)"),
]

panels_lc = []
for model, label in configs_lc:
    sizes, tr_scores, cv_scores = learning_curve(
        model, X, y,
        train_sizes=np.linspace(0.10, 0.80, 8),
        cv=cv_folds,
        scoring="neg_root_mean_squared_error",
        n_jobs=-1,
    )
    panels_lc.append((sizes, -tr_scores.mean(axis=1), tr_scores.std(axis=1),
                      -cv_scores.mean(axis=1), cv_scores.std(axis=1), label))

results_lc = plot_learning_curve_panels(
    panels_lc,
    suptitle=f"Learning curves: {DATASET_SLUG} dataset, predicting {TARGET}",
    save_path=PLOT_DIR / f"{DATASET_SLUG}_learning_curves.png",
)
for lbl, tr_f, cv_f, n_f in results_lc:
    print(f"  {lbl}: train={tr_f:.3f}  CV={cv_f:.3f}  gap={cv_f - tr_f:.3f}  (n={n_f})")
print(f"Saved: {PLOT_DIR / f'{DATASET_SLUG}_learning_curves.png'}")


# %%
""" [4] Figure 2 — Validation curve: the complexity dial (max_depth)
Fix the dataset (all samples, 5-fold CV) and sweep max_depth from 1 to 15.
max_depth is the "complexity dial": turning it up adds capacity but risks overfitting.

Reading the chart:
  • Left  (small depth) : both train and CV high → underfitting / high bias.
  • Right (large depth) : train ≈ 0, CV rises   → overfitting / high variance.
  • Sweet spot           : the depth where CV RMSE is minimised.

The validation curve answers "where on the complexity axis should I sit?"
The learning curve (Fig 1) answers "do I have enough data at that complexity?"
"""
train_vc, cv_vc = validation_curve(
    DecisionTreeRegressor(random_state=0),
    X, y,
    param_name="max_depth",
    param_range=DEPTH_RANGE,
    cv=cv_folds,
    scoring="neg_root_mean_squared_error",
    n_jobs=-1,
)
tr_mean_vc = -train_vc.mean(axis=1)
tr_std_vc  =  train_vc.std(axis=1)
cv_mean_vc = -cv_vc.mean(axis=1)
cv_std_vc  =  cv_vc.std(axis=1)

best_depth = int(DEPTH_RANGE[cv_mean_vc.argmin()])
best_cv    = cv_mean_vc.min()

plot_validation_curve(
    DEPTH_RANGE, tr_mean_vc, tr_std_vc, cv_mean_vc, cv_std_vc,
    best_param=best_depth,
    best_label=f"Best CV depth = {best_depth}",
    xlabel="max_depth  (complexity →)",
    ylabel="RMSE",
    title=f"Validation curve: Decision Tree on {DATASET_SLUG} dataset, predicting {TARGET}",
    save_path=PLOT_DIR / f"{DATASET_SLUG}_validation_curve.png",
)
print(f"Best max_depth={best_depth}  CV RMSE={best_cv:.3f}  (sweep: depth 1–{DEPTH_RANGE[-1]})")
print(f"Saved: {PLOT_DIR / f'{DATASET_SLUG}_validation_curve.png'}")


# %%
""" [5] Figure 3 — CV stability vs single validation split
How much does the evaluation score vary depending on which 20% of the data
ends up in the validation set?

Procedure (30 random seeds):
  • Single split : one train_test_split(test_size=0.2), record RMSE on the held-out 20%.
  • 5-fold CV    : cross_val_score(cv=5), record the mean RMSE of the 5 folds.

A wider box = more sensitivity to which samples happen to land in validation.
CV boxes should be narrower because each point is an average of 5 folds,
not a single lucky/unlucky draw.

Uses the best-depth tree from cell [4] (no scaler — trees are scale-invariant)
so the stability result is for the model this script is actually about.
"""
tree_best = DecisionTreeRegressor(max_depth=best_depth, random_state=0)

N_SEEDS = 30
single_scores, cv_scores_stability = [], []

for seed in range(N_SEEDS):
    X_tr, X_va, y_tr, y_va = train_test_split(X, y, test_size=0.2, random_state=seed)
    tree_best.fit(X_tr, y_tr)
    single_scores.append(np.sqrt(mean_squared_error(y_va, tree_best.predict(X_va))))

    cv_fold_obj = KFold(n_splits=cv_folds, shuffle=True, random_state=seed)
    cv_run = cross_val_score(tree_best, X, y, cv=cv_fold_obj,
                             scoring="neg_root_mean_squared_error")
    cv_scores_stability.append(-cv_run.mean())

single_s = pd.Series(single_scores,        name="Single 80/20")
cv_s     = pd.Series(cv_scores_stability,  name=f"{cv_folds}-fold CV (mean)")

plot_cv_stability(
    single_scores, cv_scores_stability, cv_folds,
    ylabel=f"RMSE  (Decision Tree, max_depth={best_depth})",
    title=f"Evaluation stability ({N_SEEDS} random seeds)\n{DATASET_SLUG} dataset, predicting {TARGET}",
    save_path=PLOT_DIR / f"{DATASET_SLUG}_cv_stability.png",
)
print(f"Single 80/20: mean={single_s.mean():.3f}  std={single_s.std():.3f}")
print(f"{cv_folds}-fold CV:  mean={cv_s.mean():.3f}  std={cv_s.std():.3f}  ({N_SEEDS} seeds)")
print(f"Saved: {PLOT_DIR / f'{DATASET_SLUG}_cv_stability.png'}")