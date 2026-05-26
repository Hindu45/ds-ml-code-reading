"""How well can linear regression predict diamond price from quality grades and physical features?"""

# %%
""" [1] Imports & config"""
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.linear_model import LinearRegression
from sklearn.metrics import root_mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from cases.utils.regression_helpers import plot_pred_vs_actual, plot_prediction_distributions

PLOT_DIR     = Path(__file__).parent / "plots"
PLOT_DIR.mkdir(exist_ok=True)

RANDOM_STATE = 42
NUMERIC_COLS = ["carat", "depth", "table", "x", "y", "z"]
CAT_COLS     = ["cut", "color", "clarity"]

# %%
""" [2] Load & clean
53,940 rows, one per diamond. Drop 20 rows where x = y = z = 0 (measurement errors).
"""
df = sns.load_dataset("diamonds")
df = df[(df[["x", "y", "z"]] > 0).all(axis=1)].copy()
print(f"Rows after cleanup: {len(df):,}")

# %%
""" [3] Encode features
One-hot encode cut/color/clarity. drop_first removes one level per group to avoid
perfect multicollinearity in the design matrix (the dummy trap).
Result: 6 numeric + 4 cut + 6 color + 7 clarity = 23 features.
"""
X_df = pd.get_dummies(
    df[NUMERIC_COLS + CAT_COLS],
    columns=CAT_COLS,
    drop_first=True,
    dtype=float,
)
y             = df["price"].values.astype(float)
feature_names = X_df.columns.tolist()
print(f"Feature matrix: {X_df.shape}")

# %%
""" [4] Train / test split  (80 / 20)
Simple two-way hold-out. No validation split needed because we are not searching
over any hyperparameter.
"""
X_np = X_df.values
X_train, X_test, y_train, y_test = train_test_split(
    X_np, y, test_size=0.20, random_state=RANDOM_STATE
)
print(f"Train: {len(y_train):,}  |  Test: {len(y_test):,}")

# %%
""" [5] Standardise — fit on train only
Z-score numeric columns using train statistics only; dummies stay in {0, 1}.
Fitting the scaler on train data only prevents leakage of test-set statistics
into the model. After standardisation, numeric coefficients are on a common
scale and directly comparable in magnitude.
"""
num_idx = [feature_names.index(c) for c in NUMERIC_COLS]
scaler  = StandardScaler().fit(X_train[:, num_idx])


def scale(X: np.ndarray) -> np.ndarray:
    """Z-score numeric columns with train statistics; leave dummies unchanged."""
    X_s = X.copy()
    X_s[:, num_idx] = scaler.transform(X[:, num_idx])
    return X_s


X_train_s = scale(X_train)
X_test_s  = scale(X_test)
stats = pd.DataFrame({"mean": scaler.mean_, "std": scaler.scale_}, index=NUMERIC_COLS)
print(stats.round(2).to_string())

# %%
""" [6] Baseline model — carat only
Single-feature OLS on carat alone. Sets a performance floor that the full model
must beat, and makes explicit how much price variation carat explains by itself.
"""
carat_idx   = feature_names.index("carat")
lr_base     = LinearRegression().fit(X_train_s[:, [carat_idx]], y_train)
y_pred_base = lr_base.predict(X_test_s[:, [carat_idx]])
rmse_base   = root_mean_squared_error(y_test, y_pred_base)
r2_base     = r2_score(y_test, y_pred_base)
print(f"Baseline (carat only)  RMSE: ${rmse_base:,.0f}   R²: {r2_base:.3f}")

# %%
""" [7] Full model — all 23 features
Adding quality grades and all dimension features. The improvement over the carat
baseline shows the incremental value of cut/color/clarity grading.
Coefficients reveal whether a grade adds or subtracts price relative to its
dropped reference level (e.g. cut_Good is relative to cut_Fair).
"""
lr_full   = LinearRegression().fit(X_train_s, y_train)
y_pred    = lr_full.predict(X_test_s)
rmse_full = root_mean_squared_error(y_test, y_pred)
r2_full   = r2_score(y_test, y_pred)
print(f"Full model             RMSE: ${rmse_full:,.0f}   R²: {r2_full:.3f}")
print(f"RMSE reduction: {(rmse_base - rmse_full) / rmse_base * 100:.1f}%")

# %%
""" [8] Prediction distributions — train vs. test
OLS is unconstrained and can predict negative prices for cheap diamonds.
The red region marks the impossible domain (price < 0); comparing train and test
distributions confirms the violation is not a data-leakage artefact but a model limitation.
"""
y_pred_train = lr_full.predict(X_train_s)
n_neg_tr = (y_pred_train < 0).sum()
n_neg_te = (y_pred < 0).sum()
plot_prediction_distributions(
    y_pred_train,
    y_pred,
    save_path=PLOT_DIR / "diamonds_pred_distributions.png",
    xlabel="Predicted price (USD)",
)

print(f"Negative predictions — train: {n_neg_tr:,} ({n_neg_tr/len(y_pred_train):.1%})  |  test: {n_neg_te:,} ({n_neg_te/len(y_pred):.1%})")
print(f"Saved: {PLOT_DIR / 'diamonds_pred_distributions.png'}")

# %%
""" [9] Predicted vs actual
Points on the dashed diagonal = perfect predictions. Fanning at high prices
reveals heteroscedasticity — the linear model underestimates variance for
expensive diamonds (typically large-carat stones with non-linear price curves).
"""
plot_pred_vs_actual(
    [("OLS full model", y_pred)],
    y_test,
    save_path=PLOT_DIR / "diamonds_pred_vs_actual.png",
    xlabel="Actual price (USD)",
    ylabel="Predicted price (USD)",
)
print(f"Saved: {PLOT_DIR / 'diamonds_pred_vs_actual.png'}")

# %%
""" [10] Coefficient plot
Sorted horizontal bar chart. Numeric features are z-scored so bar lengths are
directly comparable: the longest bar is the most influential feature.
The sign of quality-grade dummies shows whether the grade adds or subtracts
price relative to the dropped reference level.
"""
coef = pd.Series(lr_full.coef_, index=feature_names).sort_values()

fig, ax = plt.subplots(figsize=(8, 6))
coef.plot.barh(ax=ax)
ax.axvline(0, color="black", linewidth=0.8)
ax.set_xlabel("Coefficient (USD per standardised unit)")
ax.set_title("OLS coefficients — full model")
fig.tight_layout()
fig.savefig(PLOT_DIR / "diamonds_coefficients.png")
print(f"Saved: {PLOT_DIR / 'diamonds_coefficients.png'}")
plt.show()

print(f"Largest positive:\n{coef.tail(3).round(0).to_string()}")
print(f"Largest negative:\n{coef.head(3).round(0).to_string()}")

