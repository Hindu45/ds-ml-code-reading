"""Research question: Can we predict diamond price from cut, color, clarity, and physical dimensions?"""

import numpy as np
import pandas as pd
import seaborn as sns
from pathlib import Path
from sklearn.linear_model import Lasso
from sklearn.metrics import root_mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from linear_regression.utils import (
    make_ridge,
    ridge_grid, lasso_grid,
    plot_error_curves, plot_coef_paths, plot_pred_vs_actual,
)

PLOT_DIR = Path(__file__).parent / "plots"
PLOT_DIR.mkdir(exist_ok=True)

NUMERIC_COLS = ["carat", "depth", "table", "x", "y", "z"]
CAT_COLS     = ["cut", "color", "clarity"]

ALPHA_MAX        = 1
LASSO_TRAIN_FRAC = 0.30
LASSO_MAX_ITER   = 1_000
LASSO_TOL        = 1e-3
LASSO_N_FINE     = 51

# --- Load & clean -----------------------------------------------------------
df = sns.load_dataset("diamonds")
df = df[(df[["x", "y", "z"]] > 0).all(axis=1)].copy()
print(f"After cleanup: {len(df):,} rows")

# --- Encode features --------------------------------------------------------
X_df: pd.DataFrame = pd.get_dummies(
    df[NUMERIC_COLS + CAT_COLS],
    columns=CAT_COLS,
    drop_first=True,
    dtype=float,
)
y             = df["price"].values.astype(float)
feature_names = X_df.columns.tolist()
X_np          = X_df.values

# --- Three-way split (60 / 20 / 20) ----------------------------------------
X_trainval, X_test,  y_trainval, y_test  = train_test_split(X_np, y, test_size=0.20, random_state=42)
X_train,    X_val,   y_train,    y_val   = train_test_split(X_trainval, y_trainval, test_size=0.25, random_state=42)
print(f"Train {len(y_train):,}  |  Val {len(y_val):,}  |  Test {len(y_test):,}")

# --- Scale (fit on train only) ----------------------------------------------
num_idx  = [feature_names.index(c) for c in NUMERIC_COLS]
x_scaler = StandardScaler().fit(X_train[:, num_idx])
y_scaler = StandardScaler().fit(y_train.reshape(-1, 1))


def _scale_X(X: np.ndarray) -> np.ndarray:
    """Z-score numeric columns with train statistics; leave dummies unchanged."""
    X_s = X.copy()
    X_s[:, num_idx] = x_scaler.transform(X[:, num_idx])
    return X_s


def _unscale_y(y_s: np.ndarray) -> np.ndarray:
    return y_scaler.inverse_transform(y_s.reshape(-1, 1)).ravel()


X_train_s    = _scale_X(X_train)
X_val_s      = _scale_X(X_val)
X_test_s     = _scale_X(X_test)
X_trainval_s = _scale_X(X_trainval)

y_train_s    = y_scaler.transform(y_train.reshape(-1, 1)).ravel()
y_val_s      = y_scaler.transform(y_val.reshape(-1, 1)).ravel()
y_trainval_s = y_scaler.transform(y_trainval.reshape(-1, 1)).ravel()

# --- OLS baseline -----------------------------------------------------------
ols = make_ridge(0.0, len(y_train_s)).fit(X_train_s, y_train_s)
print(f"\nOLS  train RMSE = {root_mean_squared_error(y_train, _unscale_y(ols.predict(X_train_s))):,.0f}")
print(f"OLS  val   RMSE = {root_mean_squared_error(y_val,   _unscale_y(ols.predict(X_val_s))):,.0f}")

# --- Ridge grid search ------------------------------------------------------
ALPHAS_COARSE = np.linspace(0.0, ALPHA_MAX, 11)
ALPHAS_FINE   = np.linspace(0.0, ALPHA_MAX, 101)

ridge_train_c, ridge_val_c = ridge_grid(ALPHAS_COARSE, X_train_s, y_train_s, X_val_s, y_val_s, desc="Ridge coarse")
ridge_train_f, ridge_val_f = ridge_grid(ALPHAS_FINE,   X_train_s, y_train_s, X_val_s, y_val_s, desc="Ridge fine  ")

best_ridge_c = ALPHAS_COARSE[int(np.argmin(ridge_val_c))]
best_ridge_f = ALPHAS_FINE[int(np.argmin(ridge_val_f))]
y_std = float(y_scaler.scale_[0])
print(f"\nRidge coarse  best α={best_ridge_c:.3f}  val RMSE={min(ridge_val_c) * y_std:,.0f}")
print(f"Ridge fine    best α={best_ridge_f:.3f}  val RMSE={min(ridge_val_f) * y_std:,.0f}")

# --- Lasso grid search ------------------------------------------------------
ALPHAS_FINE_LASSO = np.linspace(0.0, ALPHA_MAX, LASSO_N_FINE)

rng = np.random.default_rng(42)
lasso_idx  = rng.choice(len(y_train_s), size=int(len(y_train_s) * LASSO_TRAIN_FRAC), replace=False)
X_lasso_tr = X_train_s[lasso_idx]
y_lasso_tr = y_train_s[lasso_idx]

lasso_train_c, lasso_val_c = lasso_grid(ALPHAS_COARSE,      X_lasso_tr, y_lasso_tr, X_val_s, y_val_s, max_iter=LASSO_MAX_ITER, tol=LASSO_TOL, desc="Lasso coarse")
lasso_train_f, lasso_val_f = lasso_grid(ALPHAS_FINE_LASSO,  X_lasso_tr, y_lasso_tr, X_val_s, y_val_s, max_iter=LASSO_MAX_ITER, tol=LASSO_TOL, desc="Lasso fine  ")

best_lasso_c = ALPHAS_COARSE[int(np.argmin(lasso_val_c))]
best_lasso_f = ALPHAS_FINE_LASSO[int(np.argmin(lasso_val_f))]
print(f"Lasso coarse  best α={best_lasso_c:.3f}  val RMSE={min(lasso_val_c) * y_std:,.0f}")
print(f"Lasso fine    best α={best_lasso_f:.3f}  val RMSE={min(lasso_val_f) * y_std:,.0f}")

# --- Error curves plot ------------------------------------------------------
plot_error_curves(
    specs=[
        (ALPHAS_COARSE,     [v * y_std for v in ridge_train_c], [v * y_std for v in ridge_val_c], "Ridge (L2)", "Δα = 0.1",             best_ridge_c),
        (ALPHAS_FINE,       [v * y_std for v in ridge_train_f], [v * y_std for v in ridge_val_f], "Ridge (L2)", "Δα = 0.01",            best_ridge_f),
        (ALPHAS_COARSE,     [v * y_std for v in lasso_train_c], [v * y_std for v in lasso_val_c], "Lasso (L1)", "Δα = 0.1",             best_lasso_c),
        (ALPHAS_FINE_LASSO, [v * y_std for v in lasso_train_f], [v * y_std for v in lasso_val_f], "Lasso (L1)", f"N={LASSO_N_FINE} pts", best_lasso_f),
    ],
    save_path=PLOT_DIR / "diamonds_error_curves.png",
    ylabel="RMSE (USD)",
)

# --- Coefficient paths ------------------------------------------------------
ridge_paths = np.array([
    make_ridge(a, len(y_train_s)).fit(X_train_s, y_train_s).coef_
    for a in tqdm(ALPHAS_FINE, desc="Ridge path", ncols=70)
])

lasso_paths: list[np.ndarray] = []
m_path = Lasso(alpha=1.0, warm_start=True, max_iter=LASSO_MAX_ITER, tol=LASSO_TOL)
for a in tqdm(ALPHAS_FINE_LASSO, desc="Lasso path", ncols=70):
    m_path.alpha = max(float(a), 1e-6)
    m_path.fit(X_lasso_tr, y_lasso_tr)
    lasso_paths.append(m_path.coef_.copy())

plot_coef_paths(
    alphas_ridge=ALPHAS_FINE,
    ridge_paths=ridge_paths,
    alphas_lasso=ALPHAS_FINE_LASSO,
    lasso_paths=np.array(lasso_paths),
    feature_names=feature_names,
    save_path=PLOT_DIR / "diamonds_coef_paths.png",
)

# --- Final evaluation on test set -------------------------------------------
model_ols_final   = make_ridge(0.0,          len(y_trainval_s)).fit(X_trainval_s, y_trainval_s)
model_ridge_final = make_ridge(best_ridge_f, len(y_trainval_s)).fit(X_trainval_s, y_trainval_s)
model_lasso_final = Lasso(alpha=max(float(best_lasso_f), 1e-6), max_iter=LASSO_MAX_ITER, tol=LASSO_TOL)
model_lasso_final.fit(X_trainval_s, y_trainval_s)

print("\n--- Test-set evaluation ---")
for label, model in [("OLS  ", model_ols_final), ("Ridge", model_ridge_final), ("Lasso", model_lasso_final)]:
    y_pred = _unscale_y(model.predict(X_test_s))
    print(f"{label}  RMSE = {root_mean_squared_error(y_test, y_pred):,.0f}   R² = {r2_score(y_test, y_pred):.4f}")

n_zeros = int(np.sum(model_lasso_final.coef_ == 0))
print(f"\nLasso zeroed {n_zeros}/{len(feature_names)} features at α = {best_lasso_f:.3f}")

# --- Predicted vs actual ----------------------------------------------------
plot_pred_vs_actual(
    predictions=[
        ("Ridge", _unscale_y(model_ridge_final.predict(X_test_s))),
        ("Lasso", _unscale_y(model_lasso_final.predict(X_test_s))),
    ],
    y_test=y_test,
    save_path=PLOT_DIR / "diamonds_pred_vs_actual.png",
)
