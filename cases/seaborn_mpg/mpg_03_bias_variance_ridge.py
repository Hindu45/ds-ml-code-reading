"""Research question: Does Ridge regression produce cleaner learning-curve bands than decision trees for the same bias-variance patterns?"""

# %% [1] Imports & config
import numpy as np
import pandas as pd
import seaborn as sns
from pathlib import Path
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import (
    learning_curve, validation_curve,
    cross_val_score, train_test_split, KFold,
)

from cases.utils.model_eval_plots import (
    plot_learning_curve_panels, plot_validation_curve, plot_cv_stability,
)

PLOT_DIR = Path(__file__).parent / "plots"
PLOT_DIR.mkdir(exist_ok=True)

# ── Dataset configuration ─────────────────────────────────────────────────────
DATASET_SLUG = "mpg"
FEATURES = ["cylinders", "displacement", "horsepower", "weight",
            "acceleration", "model_year"]
TARGET = "mpg"
DROP_NA_COLS = ["horsepower"]
RIDGE_ALPHA_HIGH  = 1000.0                    # heavily regularised → high bias
RIDGE_ALPHA_LOW   = 0.01                      # lightly regularised → close to OLS
RIDGE_ALPHA_RANGE = np.logspace(-2, 4, 25)   # alpha sweep for validation curve
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
""" [3] Figure 1 — Learning curves: high bias vs low-regularisation Ridge
Both panels use Ridge regression (StandardScaler + Ridge) on the same 6 features.
The only difference is the regularisation strength α.

  • High-bias panel  (α=1000): coefficients heavily shrunk → model underfits.
    Expected signature: train ≈ CV, both plateau high; gap stays small.
  • Low-reg panel    (α=0.01): barely regularised, close to OLS.
    With n/p ≈ 65 (392 samples, 6 features) OLS does not dramatically overfit,
    so this panel shows the 'well-specified but limited' baseline rather than
    classic high-variance. The learning curve tells us whether we need more data.

Key diagnostic question: are the CV std bands (shaded regions) visibly narrower
than in the decision-tree version (bias_variance_tradeoff.py)?
"""
cv_folds = 5

def ridge_pipe(alpha: float) -> Pipeline:
    return Pipeline([
        ("scaler", StandardScaler()),
        ("model",  Ridge(alpha=alpha)),
    ])

configs_lc = [
    (ridge_pipe(RIDGE_ALPHA_HIGH), f"High bias  (α={RIDGE_ALPHA_HIGH:.0f})"),
    (ridge_pipe(RIDGE_ALPHA_LOW),  f"Low regularisation  (α={RIDGE_ALPHA_LOW})"),
]

panels_lc = []
for pipe, label in configs_lc:
    sizes, tr_scores, cv_scores = learning_curve(
        pipe, X, y,
        train_sizes=np.linspace(0.10, 0.80, 8),
        cv=cv_folds,
        scoring="neg_root_mean_squared_error",
        n_jobs=-1,
    )
    panels_lc.append((sizes, -tr_scores.mean(axis=1), tr_scores.std(axis=1),
                      -cv_scores.mean(axis=1), cv_scores.std(axis=1), label))

results_lc = plot_learning_curve_panels(
    panels_lc,
    suptitle=f"Learning curves (Ridge) — {DATASET_SLUG} → {TARGET}",
    save_path=PLOT_DIR / f"{DATASET_SLUG}_ridge_learning_curves.png",
)
for lbl, tr_f, cv_f, n_f in results_lc:
    print(f"  {lbl}: train={tr_f:.3f}  CV={cv_f:.3f}  gap={cv_f - tr_f:.3f}  (n={n_f})")
print(f"Saved: {PLOT_DIR / f'{DATASET_SLUG}_ridge_learning_curves.png'}")


# %%
""" [4] Figure 2 — Validation curve: regularisation strength as complexity dial
Fix the dataset (all samples, 5-fold CV) and sweep α across 25 log-spaced values
from 0.01 to 10 000.  α is the regularisation dial — the inverse of complexity:

  • Left  (small α): less shrinkage, closer to OLS → potentially high variance.
  • Right (large α): heavy shrinkage, coefficients → 0 → high bias.
  • Sweet spot       : the α where CV RMSE is minimised.

"""
train_vc, cv_vc = validation_curve(
    Pipeline([("scaler", StandardScaler()), ("model", Ridge())]),
    X, y,
    param_name="model__alpha",
    param_range=RIDGE_ALPHA_RANGE,
    cv=cv_folds,
    scoring="neg_root_mean_squared_error",
    n_jobs=-1,
)
tr_mean_vc = -train_vc.mean(axis=1)
tr_std_vc  =  train_vc.std(axis=1)
cv_mean_vc = -cv_vc.mean(axis=1)
cv_std_vc  =  cv_vc.std(axis=1)

best_alpha = RIDGE_ALPHA_RANGE[cv_mean_vc.argmin()]
best_cv    = cv_mean_vc.min()

plot_validation_curve(
    RIDGE_ALPHA_RANGE, tr_mean_vc, tr_std_vc, cv_mean_vc, cv_std_vc,
    best_param=best_alpha,
    best_label=f"Best α = {best_alpha:.3f}",
    xlabel="α  (← less regularisation   |   more regularisation →)",
    ylabel="RMSE",
    title=f"Validation curve — Ridge on {DATASET_SLUG} → {TARGET}",
    save_path=PLOT_DIR / f"{DATASET_SLUG}_ridge_validation_curve.png",
    logx=True,
)
print(f"Best α={best_alpha:.4f}  CV RMSE={best_cv:.3f}  (sweep: {RIDGE_ALPHA_RANGE[0]:.2f}–{RIDGE_ALPHA_RANGE[-1]:.0f})")
print(f"Saved: {PLOT_DIR / f'{DATASET_SLUG}_ridge_validation_curve.png'}")


# %%
""" [5] Figure 3 — CV stability vs single validation split
How much does the evaluation score vary depending on which 20% of the data
ends up in the validation set?

Procedure (30 random seeds):
  • Single split : one train_test_split(test_size=0.2), record RMSE on the held-out 20%.
  • 5-fold CV    : cross_val_score(cv=5), record the mean RMSE of the 5 folds.

Uses the best-α Ridge pipe so the stability result is for the tuned model.
"""
pipe_best = ridge_pipe(best_alpha)

N_SEEDS = 30
single_scores, cv_scores_stability = [], []

for seed in range(N_SEEDS):
    X_tr, X_va, y_tr, y_va = train_test_split(X, y, test_size=0.2, random_state=seed)
    pipe_best.fit(X_tr, y_tr)
    single_scores.append(np.sqrt(mean_squared_error(y_va, pipe_best.predict(X_va))))

    cv_fold_obj = KFold(n_splits=cv_folds, shuffle=True, random_state=seed)
    cv_run = cross_val_score(pipe_best, X, y, cv=cv_fold_obj,
                             scoring="neg_root_mean_squared_error")
    cv_scores_stability.append(-cv_run.mean())

single_s = pd.Series(single_scores,       name="Single 80/20")
cv_s     = pd.Series(cv_scores_stability, name=f"{cv_folds}-fold CV (mean)")

plot_cv_stability(
    single_scores, cv_scores_stability, cv_folds,
    ylabel=f"RMSE  (Ridge α={best_alpha:.4f})",
    title=f"Evaluation stability — {N_SEEDS} random seeds\n{DATASET_SLUG} → {TARGET}",
    save_path=PLOT_DIR / f"{DATASET_SLUG}_ridge_cv_stability.png",
)
print(f"Single 80/20: mean={single_s.mean():.3f}  std={single_s.std():.3f}")
print(f"{cv_folds}-fold CV:  mean={cv_s.mean():.3f}  std={cv_s.std():.3f}  ({N_SEEDS} seeds)")
print(f"Saved: {PLOT_DIR / f'{DATASET_SLUG}_ridge_cv_stability.png'}")