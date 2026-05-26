"""Research question: How do model complexity and regularisation affect the bias-variance tradeoff?"""

# %% [1] Imports & config
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import (
    learning_curve, validation_curve,
    cross_val_score, train_test_split, KFold,
)
from sklearn.tree import DecisionTreeRegressor

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

fig, axes = plt.subplots(1, 2, figsize=(11, 5), sharey=True)

for ax, (model, label) in zip(axes, configs_lc):
    sizes, tr_scores, cv_scores = learning_curve(
        model, X, y,
        train_sizes=np.linspace(0.10, 0.80, 8),
        cv=cv_folds,
        scoring="neg_root_mean_squared_error",
        n_jobs=-1,
    )
    tr_mean = -tr_scores.mean(axis=1)
    tr_std  =  tr_scores.std(axis=1)
    cv_mean = -cv_scores.mean(axis=1)
    cv_std  =  cv_scores.std(axis=1)

    ax.plot(sizes, tr_mean, "o-",  color="C0", label="Train RMSE")
    ax.fill_between(sizes, tr_mean - tr_std, tr_mean + tr_std, alpha=0.15, color="C0")
    ax.plot(sizes, cv_mean, "s--", color="C1", label="CV RMSE")
    ax.fill_between(sizes, cv_mean - cv_std, cv_mean + cv_std, alpha=0.10, color="C1")

    ax.set_xlabel("Training set size")
    ax.set_title(label, fontsize=11)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    title_plain = label.replace("\n", " ")
    lc_df = pd.DataFrame({
        "n_train":   sizes.astype(int),
        "train_mean": tr_mean,
        "train_std":  tr_std,
        "cv_mean":    cv_mean,
        "cv_std":     cv_std,
        "gap":        cv_mean - tr_mean,
    })
    print(f"\n── Learning curve: {title_plain} ──")
    print(f"   {cv_folds}-fold CV at {len(sizes)} training sizes "
          f"({int(sizes[0])}–{int(sizes[-1])} of {X.shape[0]} samples).")
    print(lc_df.to_string(index=False, float_format="{:.3f}".format))

axes[0].set_ylabel("RMSE")
fig.suptitle(f"Learning curves: {DATASET_SLUG} dataset, predicting {TARGET}", fontsize=13)
fig.tight_layout()
fig.savefig(PLOT_DIR / f"{DATASET_SLUG}_learning_curves.png")
plt.show()


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

vc_df = pd.DataFrame({
    "max_depth":  DEPTH_RANGE,
    "train_mean": tr_mean_vc,
    "cv_mean":    cv_mean_vc,
    "gap":        cv_mean_vc - tr_mean_vc,
})
print(f"\n── Validation curve (max_depth sweep) ──")
print(f"   Best CV RMSE = {best_cv:.3f} at max_depth = {best_depth}")
print(vc_df.to_string(index=False, float_format="{:.3f}".format))

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(DEPTH_RANGE, tr_mean_vc, "o-",  color="C0", label="Train RMSE")
ax.fill_between(DEPTH_RANGE, tr_mean_vc - tr_std_vc, tr_mean_vc + tr_std_vc,
                alpha=0.15, color="C0")
ax.plot(DEPTH_RANGE, cv_mean_vc, "s--", color="C1", label="CV RMSE")
ax.fill_between(DEPTH_RANGE, cv_mean_vc - cv_std_vc, cv_mean_vc + cv_std_vc,
                alpha=0.10, color="C1")
ax.axvline(best_depth, color="gray", linestyle=":", linewidth=1.5,
           label=f"Best CV depth = {best_depth}")
ax.set_xlabel("max_depth  (complexity →)")
ax.set_ylabel("RMSE")
ax.set_title(f"Validation curve: Decision Tree on {DATASET_SLUG} dataset, predicting {TARGET}", fontsize=12)
ax.legend()
ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig(PLOT_DIR / f"{DATASET_SLUG}_validation_curve.png")
plt.show()


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
"""
pipe_linear = Pipeline([
    ("scaler", StandardScaler()),
    ("model",  LinearRegression()),
])

N_SEEDS = 30
single_scores, cv_scores_stability = [], []

for seed in range(N_SEEDS):
    X_tr, X_va, y_tr, y_va = train_test_split(X, y, test_size=0.2, random_state=seed)
    pipe_linear.fit(X_tr, y_tr)
    single_scores.append(np.sqrt(mean_squared_error(y_va, pipe_linear.predict(X_va))))

    cv_fold_obj = KFold(n_splits=cv_folds, shuffle=True, random_state=seed)
    cv_run = cross_val_score(pipe_linear, X, y, cv=cv_fold_obj,
                             scoring="neg_root_mean_squared_error")
    cv_scores_stability.append(-cv_run.mean())

single_s = pd.Series(single_scores,        name="Single 80/20")
cv_s     = pd.Series(cv_scores_stability,  name=f"{cv_folds}-fold CV (mean)")
stability_df = pd.concat([single_s.describe(), cv_s.describe()], axis=1).round(3)
print(f"\n── CV stability — {N_SEEDS} random seeds ──")
print(f"   Single 80/20: train_test_split(test_size=0.2) per seed "
      f"→ 1 RMSE on ~{X.shape[0] - int(X.shape[0]*0.8)} samples.")
print(f"   {cv_folds}-fold CV: KFold(shuffle=True) per seed "
      f"→ mean RMSE over {cv_folds} folds of ~{X.shape[0]//cv_folds} samples each.")
print(f"   Box = IQR; whiskers = 1.5×IQR.")
print(stability_df.to_string())

fig, ax = plt.subplots(figsize=(6, 5))
bp = ax.boxplot(
    [single_scores, cv_scores_stability],
    tick_labels=["Single 80/20 split", f"{cv_folds}-fold CV (mean)"],
    patch_artist=True,
    widths=0.4,
)
bp["boxes"][0].set_facecolor("tab:orange")
bp["boxes"][1].set_facecolor("tab:blue")
for patch in bp["boxes"]:
    patch.set_alpha(0.6)

ax.set_ylabel("RMSE  (linear model on full feature set)")
ax.set_title(
    f"Evaluation stability ({N_SEEDS} random seeds)\n"
    f"{DATASET_SLUG} dataset, predicting {TARGET}"
)
fig.tight_layout()
fig.savefig(PLOT_DIR / f"{DATASET_SLUG}_cv_stability.png")
plt.show()

# %% [6]
