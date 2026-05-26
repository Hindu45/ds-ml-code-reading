"""Research question: Does healthcare spending predict life expectancy across six countries?"""

# %%
""" [1] Imports & config"""
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

PLOT_DIR = Path(__file__).parent / "plots"
PLOT_DIR.mkdir(exist_ok=True)

CATEGORICAL = ["Country"]
NUMERIC     = ["Year", "Spending_USD"]
N_REPEATS   = 50
RNG         = np.random.default_rng(0)

# %%
""" [2] Load & explore
274 rows, one per (country, year) combination spanning 1970–2020.
Features: Year (int), Country (6 categories), Spending_USD (float).
Target:   Life_Expectancy (float, years).
"""
df = sns.load_dataset("healthexp")
print(df.head())
print(df[["Spending_USD", "Life_Expectancy"]].describe().round(2))
print("Countries:", sorted(df["Country"].unique()))

# %% [3] Train / test split  (80 / 20)  — single feature first
X = df[["Spending_USD"]].values   # shape (n, 1)
y = df["Life_Expectancy"].values  # shape (n,)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# %% [4] Standardise — fit statistics on train only, apply to both splits
mu: float    = float(X_train.mean())
sigma: float = float(X_train.std())

X_train_s = (X_train - mu) / sigma
X_test_s  = (X_test  - mu) / sigma


def add_bias(X: np.ndarray) -> np.ndarray:
    """Prepend a column of ones so the first weight becomes the intercept."""
    return np.column_stack([np.ones(len(X)), X])


X_train_b = add_bias(X_train_s)   # shape (n_train, 2)
X_test_b  = add_bias(X_test_s)    # shape (n_test,  2)

# %% [5] Loss function: Mean Squared Error
def mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """MSE = (1/n) Σ(y - ŷ)²  — average squared prediction error."""
    return float(np.mean((y_true - y_pred) ** 2))


# %% [6] Analytical solution — Normal equation: θ = (XᵀX)⁻¹ Xᵀy
theta = np.linalg.lstsq(X_train_b, y_train, rcond=None)[0]

print(f"\nOLS  intercept={theta[0]:.4f}  slope={theta[1]:.4f}")
print(f"Train MSE = {mse(y_train, X_train_b @ theta):.4f}")

# %% [7] Metrics by hand: RMSE and R²
def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Root MSE — same unit as the target (years of life expectancy)."""
    return float(np.sqrt(mse(y_true, y_pred)))


def r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """R² = 1 - SS_res / SS_tot.  1.0 is perfect; 0.0 means predicting the mean."""
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    return 1.0 - ss_res / ss_tot


y_pred_test = X_test_b @ theta
print(f"\n--- Single feature (Spending_USD) ---")
print(f"Test  RMSE = {rmse(y_test, y_pred_test):.3f} years")
print(f"Test  R²   = {r2(y_test, y_pred_test):.3f}")

# %% [8] Plot regression line on original (unstandardised) scale
x_orig = np.linspace(X_test.min(), X_test.max(), 200).reshape(-1, 1)
x_orig_b = add_bias((x_orig - mu) / sigma)

fig, ax = plt.subplots(figsize=(7, 5))
ax.scatter(X_test, y_test, alpha=0.6, label="test data")
ax.plot(x_orig, x_orig_b @ theta, color="tab:red", label="OLS fit")
ax.set_xlabel("Healthcare spending (USD)")
ax.set_ylabel("Life expectancy (years)")
ax.set_title("Linear regression: life expectancy ~ spending")
ax.legend()
fig.tight_layout()
fig.savefig(PLOT_DIR / "linreg_healthexp_single.png")
plt.show()

# ── ALL FEATURES ──────────────────────────────────────────────────────────────

# %%
""" [9] Prepare features: one-hot encode Country, keep Year + Spending_USD
Adding Year captures the global upward trend in life expectancy independent of
spending. Country dummies absorb between-country baseline differences.
drop_first=True avoids the dummy trap (perfect multicollinearity).
"""
X_all: pd.DataFrame = pd.get_dummies(
    df.drop(columns="Life_Expectancy"),
    columns=CATEGORICAL,
    drop_first=True,
    dtype=float,
)
y_all = df["Life_Expectancy"].values

print("\nFeatures after encoding:")
print(X_all.columns.tolist())

X_all_train, X_all_test, y_all_train, y_all_test = train_test_split(
    X_all.values, y_all, test_size=0.2, random_state=42
)

# %% [10] Standardise only the numeric columns on train; dummies stay 0/1
col_list = list(X_all.columns)
num_idx  = [col_list.index(c) for c in NUMERIC]

mu_all    = X_all_train[:, num_idx].mean(axis=0)
sigma_all = X_all_train[:, num_idx].std(axis=0)

X_all_train_s = X_all_train.copy()
X_all_test_s  = X_all_test.copy()
X_all_train_s[:, num_idx] = (X_all_train[:, num_idx] - mu_all) / sigma_all
X_all_test_s[:, num_idx]  = (X_all_test[:, num_idx]  - mu_all) / sigma_all

# %% [11] Fit & compare R²
model = LinearRegression()
model.fit(X_all_train_s, y_all_train)

r2_train = r2_score(y_all_train, model.predict(X_all_train_s))
r2_test  = r2_score(y_all_test,  model.predict(X_all_test_s))

print(f"\n--- All features ---")
print(f"R² train = {r2_train:.3f}")
print(f"R² test  = {r2_test:.3f}")
print(f"(single-feature test R² was {r2(y_test, y_pred_test):.3f})")

# %%
""" [12] Permutation importance
For each feature: shuffle its values in the test set and measure how much R²
drops. Larger drop → the model relied on that feature more.
No refitting needed — purely a test-time diagnostic.
"""
importances: dict[str, float] = {}

for i, feat in enumerate(col_list):
    drops: list[float] = []
    for _ in range(N_REPEATS):
        X_perm = X_all_test_s.copy()
        X_perm[:, i] = RNG.permutation(X_perm[:, i])
        drops.append(r2_test - r2_score(y_all_test, model.predict(X_perm)))
    importances[feat] = float(np.mean(drops))

print("\nPermutation importance (mean R² drop):")
for feat, imp in sorted(importances.items(), key=lambda x: -x[1]):
    print(f"  {feat:<25} {imp:+.4f}")

# %% [13] Plot feature importance
order  = sorted(importances, key=importances.get)   # ascending → bottom = least
values = [importances[f] for f in order]

fig, ax = plt.subplots(figsize=(7, 5))
ax.barh(order, values)
ax.axvline(0, color="black", linewidth=0.8)
ax.set_xlabel("Mean R² drop when feature is permuted")
ax.set_title(f"Permutation importance  (test R² = {r2_test:.3f})")
fig.tight_layout()
fig.savefig(PLOT_DIR / "linreg_healthexp_importance.png")
plt.show()

# %%
""" [14] Residuals vs fitted values
Homoscedasticity check: if the linear model's assumptions hold, residuals
should scatter evenly around zero across all fitted values (no funnel shape).
"""
y_hat     = model.predict(X_all_train_s)
residuals = y_all_train - y_hat

fig, ax = plt.subplots(figsize=(7, 4))
ax.scatter(y_hat, residuals, alpha=0.5, s=18)
ax.axhline(0, color="black", linewidth=0.8)
ax.set_xlabel("Fitted value (predicted life expectancy, years)")
ax.set_ylabel("Residual (actual − predicted)")
ax.set_title("Residuals vs. fitted — homoscedasticity check")
fig.tight_layout()
fig.savefig(PLOT_DIR / "linreg_healthexp_residuals.png")
plt.show()

# %%
""" [15] Standardised coefficients
β_std_j = β_j × (σ_xj / σ_y)
Converts each coefficient to units of std(y) per std(x_j), making magnitudes
directly comparable across features with different scales.
"""
std_X    = X_all_train_s.std(axis=0)
std_y    = float(y_all_train.std())
beta_std = model.coef_ * std_X / std_y

print("\nStandardised coefficients:")
for name, b in sorted(zip(col_list, beta_std), key=lambda x: -abs(x[1])):
    print(f"  {name:<25} {b:+.4f}")
