"""Research question: Can we predict diamond price from cut, color, clarity, and physical dimensions?"""

# %% Imports & config
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.linear_model import Ridge, Lasso
from sklearn.model_selection import train_test_split
from tqdm import tqdm

PLOT_DIR = Path(__file__).parent / "plots"
PLOT_DIR.mkdir(exist_ok=True)

NUMERIC_COLS = ["carat", "depth", "table", "x", "y", "z"]
CAT_COLS     = ["cut", "color", "clarity"]

ALPHA_MAX = 1   # upper bound of the α search range; lower if optimum is still near 0

# --- Lasso compute budget -------------------------------------------------
# Reduce these to trade accuracy for speed; restore to defaults for a full run.
LASSO_TRAIN_FRAC = 0.30    # fraction of train set used for Lasso (1.0 = all ~32k rows)
LASSO_MAX_ITER   = 1_000   # coordinate-descent iterations per α; raise for convergence
LASSO_TOL        = 1e-3    # convergence tolerance (default 1e-4; loosen to go faster)
LASSO_N_FINE     = 51      # how many points in the fine α grid for Lasso (max 101)
# -------------------------------------------------------------------------

# %% Load & inspect
"""
53,940 rows — one per diamond.
Numeric: carat, depth (%), table (%), x/y/z (mm dimensions).
Categorical: cut (5 grades), color (7 grades D–J), clarity (8 grades).
Target: price in USD.
A small number of rows have x=y=z=0 (data entry errors) — we drop them.
"""
df = sns.load_dataset("diamonds")

df = df[(df[["x", "y", "z"]] > 0).all(axis=1)].copy()
print(f"After cleanup: {len(df):,} rows")


# %% Encode features
"""
One-hot encode cut/color/clarity (drop_first=True avoids dummy trap).
Result: 6 numeric + 4 cut + 6 color + 7 clarity = 23 features.
With 23 features and high multicollinearity, regularisation is well-motivated.
"""
X_df: pd.DataFrame = pd.get_dummies(
    df[NUMERIC_COLS + CAT_COLS],
    columns=CAT_COLS,
    drop_first=True,
    dtype=float,
)
y              = df["price"].values.astype(float)
feature_names  = X_df.columns.tolist()

print(f"\nFeature matrix: {X_df.shape}")
print("Features:", feature_names)

# %% Train / val / test split  (60 / 20 / 20)
"""
Three-way split is required whenever we search over a hyperparameter (α):
  train  (60 %) — fit model weights
  val    (20 %) — select best α  (tuning set)
  test   (20 %) — final unbiased evaluation; never used during grid search
"""
X_np = X_df.values

X_trainval, X_test,  y_trainval, y_test  = train_test_split(
    X_np, y, test_size=0.20, random_state=42
)
X_train, X_val, y_train, y_val = train_test_split(
    X_trainval, y_trainval, test_size=0.25, random_state=42   # 0.25 × 0.80 = 0.20
)

print(f"Train {len(y_train):,}  |  Val {len(y_val):,}  |  Test {len(y_test):,}")

# %% Standardise — fit on train only; apply to val and test
"""
All numeric inputs and the target are z-scored using train statistics only.
Dummies stay in {0, 1} — scaling them is harmless but unnecessary.
Z-scoring y is essential: it brings the target onto the same O(1) scale as the
features, so the regularisation strength α is comparable across both Ridge and
Lasso and meaningful in [0, ALPHA_MAX] regardless of the original price range.
"""
_num_idx = [feature_names.index(c) for c in NUMERIC_COLS]
_mu      = X_train[:, _num_idx].mean(axis=0)
_sigma   = X_train[:, _num_idx].std(axis=0)

_y_mean = float(y_train.mean())
_y_std  = float(y_train.std())


def scale(X: np.ndarray) -> np.ndarray:
    """Z-score numeric columns with train statistics; leave dummies unchanged."""
    X_s = X.copy()
    X_s[:, _num_idx] = (X[:, _num_idx] - _mu) / _sigma
    return X_s


def unscale_y(y_s: np.ndarray) -> np.ndarray:
    """Convert z-scored predictions back to USD."""
    return y_s * _y_std + _y_mean


X_train_s    = scale(X_train)
X_val_s      = scale(X_val)
X_test_s     = scale(X_test)
X_trainval_s = scale(X_trainval)

y_train_s    = (y_train    - _y_mean) / _y_std
y_val_s      = (y_val      - _y_mean) / _y_std
y_test_s     = (y_test     - _y_mean) / _y_std
y_trainval_s = (y_trainval - _y_mean) / _y_std

# %% Metric helpers
def mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """MSE = (1/n) Σ(y − ŷ)²"""
    return float(np.mean((y_true - y_pred) ** 2))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """RMSE in the same unit as the target (USD for price)."""
    return float(np.sqrt(mse(y_true, y_pred)))


def r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """R² = 1 − SS_res / SS_tot"""
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    return 1.0 - ss_res / ss_tot


# %% Ridge: sklearn with n-normalised α
"""
sklearn Ridge minimises ||y − Xw||² + α_skl · ||w||²  (sum, not mean).
We pass α_skl = user_α × n so the effective per-sample penalty equals user_α,
matching sklearn Lasso's (1/2n) convention and making α comparable across both.

With y and X z-scored, α ≈ 0 recovers OLS; α ≈ 1 applies strong shrinkage.
"""
def _ridge(alpha: float, n: int) -> Ridge:
    """Return a Ridge instance whose α is scaled by n for per-sample comparability."""
    return Ridge(alpha=max(alpha, 1e-10) * n)


# OLS baseline (α ≈ 0)
_ols = _ridge(0.0, len(y_train_s))
_ols.fit(X_train_s, y_train_s)
print(f"\nOLS  train RMSE = {rmse(unscale_y(_ols.predict(X_train_s)), y_train):,.0f} USD")
print(f"OLS  val   RMSE = {rmse(unscale_y(_ols.predict(X_val_s)),   y_val):,.0f} USD")

# %% Ridge grid search — coarse (Δα = 0.1) and fine (Δα = 0.01)
"""
Sweep α across [0, 1] at two resolutions.
Coarse (11 points) may straddle the optimum; fine (101 points) resolves it.
"""
ALPHAS_COARSE = np.linspace(0.0, ALPHA_MAX, 11)    # 11 points  (Δα = ALPHA_MAX / 10)
ALPHAS_FINE   = np.linspace(0.0, ALPHA_MAX, 101)   # 101 points (Δα = ALPHA_MAX / 100)


def ridge_grid(
    alphas: np.ndarray,
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    X_va: np.ndarray,
    y_va: np.ndarray,
) -> tuple[list[float], list[float]]:
    """Evaluate Ridge train/val RMSE for each α value.

    Args:
        alphas: Sequence of regularisation strengths.
        X_tr, y_tr: Training split.
        X_va, y_va: Validation split.

    Returns:
        Tuple (train_rmse_list, val_rmse_list).
    """
    n = len(y_tr)
    train_errs, val_errs = [], []
    for a in tqdm(alphas, desc="Ridge", ncols=70):
        m = _ridge(a, n)
        m.fit(X_tr, y_tr)
        train_errs.append(rmse(y_tr, m.predict(X_tr)))
        val_errs.append(rmse(y_va, m.predict(X_va)))
    return train_errs, val_errs


ridge_train_c, ridge_val_c = ridge_grid(ALPHAS_COARSE, X_train_s, y_train_s, X_val_s, y_val_s)
ridge_train_f, ridge_val_f = ridge_grid(ALPHAS_FINE,   X_train_s, y_train_s, X_val_s, y_val_s)

best_ridge_c = ALPHAS_COARSE[int(np.argmin(ridge_val_c))]
best_ridge_f = ALPHAS_FINE[int(np.argmin(ridge_val_f))]
print(f"\nRidge coarse  best α={best_ridge_c:.3f}  val RMSE={min(ridge_val_c) * _y_std:,.0f} USD")
print(f"Ridge fine    best α={best_ridge_f:.3f}  val RMSE={min(ridge_val_f) * _y_std:,.0f} USD")

# %% Lasso grid search — coarse and fine
"""
Lasso minimises ||y − Xθ||² + α · ||θ_features||₁  (L1 penalty).
No closed-form → sklearn uses coordinate descent.
α=0 is a degenerate limit; we substitute 1e-6 as a proxy for OLS.
Key difference from Ridge: Lasso drives some coefficients to exactly zero,
performing implicit feature selection.
"""
def lasso_grid(
    alphas: np.ndarray,
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    X_va: np.ndarray,
    y_va: np.ndarray,
    desc: str = "Lasso",
) -> tuple[list[float], list[float]]:
    """Evaluate Lasso train/val RMSE for each α value.

    Uses a single warm-started model so each fit continues from the previous
    solution rather than re-initialising from scratch — much faster on a path.

    Args:
        alphas: Sequence of regularisation strengths (low → high recommended).
        X_tr, y_tr: Training split.
        X_va, y_va: Validation split.
        desc: Label shown in the progress bar.

    Returns:
        Tuple (train_rmse_list, val_rmse_list).
    """
    train_errs, val_errs = [], []
    m = Lasso(alpha=1.0, warm_start=True, max_iter=LASSO_MAX_ITER, tol=LASSO_TOL)
    for a in tqdm(alphas, desc=desc, ncols=70):
        m.alpha = max(float(a), 1e-6)
        m.fit(X_tr, y_tr)
        train_errs.append(rmse(y_tr, m.predict(X_tr)))
        val_errs.append(rmse(y_va, m.predict(X_va)))
    return train_errs, val_errs


# Lasso uses a subsampled train set and a capped fine grid for speed.
# Increase LASSO_TRAIN_FRAC → 1.0 and LASSO_N_FINE → 101 for full accuracy.
rng = np.random.default_rng(42)
_lasso_idx = rng.choice(len(y_train_s), size=int(len(y_train_s) * LASSO_TRAIN_FRAC), replace=False)
X_lasso_tr, y_lasso_tr = X_train_s[_lasso_idx], y_train_s[_lasso_idx]

ALPHAS_FINE_LASSO = np.linspace(0.0, ALPHA_MAX, LASSO_N_FINE)

lasso_train_c, lasso_val_c = lasso_grid(ALPHAS_COARSE,      X_lasso_tr, y_lasso_tr, X_val_s, y_val_s, desc="Lasso coarse")
lasso_train_f, lasso_val_f = lasso_grid(ALPHAS_FINE_LASSO,  X_lasso_tr, y_lasso_tr, X_val_s, y_val_s, desc="Lasso fine  ")

best_lasso_c = ALPHAS_COARSE[int(np.argmin(lasso_val_c))]
best_lasso_f = ALPHAS_FINE_LASSO[int(np.argmin(lasso_val_f))]
print(f"Lasso coarse  best α={best_lasso_c:.3f}  val RMSE={min(lasso_val_c) * _y_std:,.0f} USD")
print(f"Lasso fine    best α={best_lasso_f:.3f}  val RMSE={min(lasso_val_f) * _y_std:,.0f} USD")

# %% Error curves — 2×2: (Ridge / Lasso) × (coarse / fine)
"""
Each subplot: blue = train RMSE, orange = val RMSE, green dashed = best α.
Reading across columns shows how the coarse grid can misplace the optimum.
Reading across rows shows Ridge vs Lasso bias-variance profiles.
"""
specs = [
    (ALPHAS_COARSE,     ridge_train_c, ridge_val_c, "Ridge (L2)", "Δα = 0.1",  best_ridge_c),
    (ALPHAS_FINE,       ridge_train_f, ridge_val_f, "Ridge (L2)", "Δα = 0.01", best_ridge_f),
    (ALPHAS_COARSE,     lasso_train_c, lasso_val_c, "Lasso (L1)", "Δα = 0.1",  best_lasso_c),
    (ALPHAS_FINE_LASSO, lasso_train_f, lasso_val_f, "Lasso (L1)", f"N={LASSO_N_FINE} pts", best_lasso_f),
]

fig, axes = plt.subplots(2, 2, figsize=(12, 8))

for ax, (alphas, tr, va, model_name, step, best_a) in zip(axes.flat, specs):
    ax.plot(alphas, [v * _y_std for v in tr], label="train", color="tab:blue")
    ax.plot(alphas, [v * _y_std for v in va], label="val",   color="tab:orange")
    ax.axvline(best_a, color="tab:green", linestyle="--", linewidth=1.2,
               label=f"best α = {best_a:.3f}")
    ax.set_xlabel("α (regularisation strength)")
    ax.set_ylabel("RMSE (USD)")
    ax.set_title(f"{model_name} — {step}")
    ax.legend(fontsize=8)

fig.suptitle("Train / Val error curves: coarse vs. fine grid search")
fig.tight_layout()
fig.savefig(PLOT_DIR / "diamonds_error_curves.png")
plt.show()

# %% Coefficient paths
"""
Plot how each feature weight changes as α increases from 0 to 1.
Ridge (L2): all weights shrink smoothly; none reach exactly zero.
Lasso (L1): weights are pulled to zero one by one — implicit feature selection.
The contrast illustrates the geometric difference between L2 (sphere) and L1
(diamond) constraint regions.
"""
ridge_paths = np.array([
    _ridge(a, len(y_train_s)).fit(X_train_s, y_train_s).coef_
    for a in tqdm(ALPHAS_FINE, desc="Ridge path", ncols=70)
])

lasso_paths: list[np.ndarray] = []
m_path = Lasso(alpha=1.0, warm_start=True, max_iter=LASSO_MAX_ITER, tol=LASSO_TOL)
for a in tqdm(ALPHAS_FINE_LASSO, desc="Lasso path", ncols=70):
    m_path.alpha = max(float(a), 1e-6)
    m_path.fit(X_lasso_tr, y_lasso_tr)
    lasso_paths.append(m_path.coef_.copy())
lasso_paths_arr = np.array(lasso_paths)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
for ax, alphas_plot, paths, title in zip(
    axes,
    [ALPHAS_FINE,       ALPHAS_FINE_LASSO],
    [ridge_paths,       lasso_paths_arr],
    ["Ridge (L2) — smooth shrinkage", "Lasso (L1) — sparsity-inducing"],
):
    for j, feat in enumerate(feature_names):
        ax.plot(alphas_plot, paths[:, j], label=feat, linewidth=1)
    ax.axhline(0, color="black", linewidth=0.7)
    ax.set_xlabel("α")
    ax.set_ylabel("Coefficient")
    ax.set_title(title)
    ax.legend(fontsize=6, ncol=2, loc="upper right")

fig.tight_layout()
fig.savefig(PLOT_DIR / "diamonds_coef_paths.png")
plt.show()

# %% Final evaluation on test set
"""
Refit the best Ridge and Lasso on the combined train+val set, then evaluate
once on the held-out test set.  Using train+val for the final fit squeezes
more signal from the available data without inflating the test estimate.
Predictions are produced in z-scored space then unscaled back to USD.
"""
model_ols_final = _ridge(0.0, len(y_trainval_s))
model_ols_final.fit(X_trainval_s, y_trainval_s)

model_ridge_final = _ridge(best_ridge_f, len(y_trainval_s))
model_ridge_final.fit(X_trainval_s, y_trainval_s)

model_lasso_final = Lasso(alpha=max(float(best_lasso_f), 1e-6), max_iter=LASSO_MAX_ITER, tol=LASSO_TOL)
model_lasso_final.fit(X_trainval_s, y_trainval_s)

print("\n--- Test-set evaluation ---")
for label, model in [("OLS  ", model_ols_final),
                      ("Ridge", model_ridge_final),
                      ("Lasso", model_lasso_final)]:
    y_pred = unscale_y(model.predict(X_test_s))
    print(f"{label}  RMSE = {rmse(y_test, y_pred):,.0f} USD   R² = {r2(y_test, y_pred):.4f}")

n_zeros = int(np.sum(model_lasso_final.coef_ == 0))
print(f"\nLasso zeroed {n_zeros}/{len(feature_names)} features at α = {best_lasso_f:.3f}")

# %% Predicted vs actual (test set)
fig, axes = plt.subplots(1, 2, figsize=(11, 5))
for ax, model, label in zip(axes,
                             [model_ridge_final, model_lasso_final],
                             ["Ridge", "Lasso"]):
    y_pred = unscale_y(model.predict(X_test_s))
    ax.scatter(y_test, y_pred, alpha=0.05, s=5)
    lo = min(float(y_test.min()), float(y_pred.min()))
    hi = max(float(y_test.max()), float(y_pred.max()))
    ax.plot([lo, hi], [lo, hi], "r--", linewidth=1)
    ax.set_xlabel("Actual price (USD)")
    ax.set_ylabel("Predicted price (USD)")
    ax.set_title(f"{label} — predicted vs. actual (test)")
fig.tight_layout()
fig.savefig(PLOT_DIR / "diamonds_pred_vs_actual.png")
plt.show()

# %%