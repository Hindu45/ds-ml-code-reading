"""Research question: How much does each feature contribute to predicting tip,
and what R² can we achieve with all available predictors?"""

# %% Imports & config
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

CATEGORICAL = ["sex", "smoker", "day", "time"]
NUMERIC     = ["total_bill", "size"]
N_REPEATS   = 50
RNG         = np.random.default_rng(0)

# %% Load & one-hot encode (drop_first → identifiable, avoids dummy trap)
df = sns.load_dataset("tips")

X: pd.DataFrame = pd.get_dummies(
    df.drop(columns="tip"),
    columns=CATEGORICAL,
    drop_first=True,
    dtype=float,
)
y = df["tip"].values
# y = df["tip"].values / X["total_bill"].values

print("Features after encoding:")
print(X.columns.tolist())

# %% Train / test split
X_train, X_test, y_train, y_test = train_test_split(
    X.values, y, test_size=0.2, random_state=42
)

# %% Fit & baseline R²
model = LinearRegression()
model.fit(X_train, y_train)

r2_train = r2_score(y_train, model.predict(X_train))
r2_test  = r2_score(y_test,  model.predict(X_test))

print(f"\nR² train = {r2_train:.3f}")
print(f"R² test  = {r2_test:.3f}")

# %% Regression coefficients — OLS solution
# Features kept in dataset order so encoded dummies of the same original variable stay adjacent.
feat_names = X.columns.tolist()
coefs = model.coef_

fig, ax = plt.subplots(figsize=(13, 4))
colors = ["tab:red" if c < 0 else "tab:blue" for c in coefs]
ax.bar(feat_names, coefs, color=colors)
ax.axhline(0, color="black", linewidth=0.8)
ax.set_ylabel("Coefficient", fontsize = 15)
ax.set_title("OLS regression coefficients (raw scale)", fontsize=20)
ax.tick_params(axis="x", rotation=40, labelsize=15)
fig.tight_layout()
fig.savefig(PLOT_DIR / "linreg_tips_coefficients.png")
plt.show()

# %% Map each original feature → its column indices in X
# Numeric: direct name match.  Categorical: all dummies share the prefix "{name}_".
feature_groups: dict[str, list[int]] = {}
col_list = list(X.columns)

for feat in NUMERIC:
    feature_groups[feat] = [col_list.index(feat)]

for feat in CATEGORICAL:
    feature_groups[feat] = [i for i, c in enumerate(col_list) if c.startswith(f"{feat}_")]

# %% Permutation importance
# For each feature: shuffle its columns in X_test N_REPEATS times, average the R² drop.
# Larger drop → feature was more important.  Does not require refitting.
importances: dict[str, float] = {}

for feat, col_idx in feature_groups.items():
    drops: list[float] = []
    for _ in range(N_REPEATS):
        X_perm = X_test.copy()
        perm = RNG.permutation(len(X_perm))
        X_perm[:, col_idx] = X_perm[perm][:, col_idx]
        drops.append(r2_test - r2_score(y_test, model.predict(X_perm)))
    importances[feat] = float(np.mean(drops))

print("\nPermutation importance (mean R² drop):")
for feat, imp in sorted(importances.items(), key=lambda x: -x[1]):
    print(f"  {feat:<12} {imp:+.4f}")

# %% Plot feature importance
order = sorted(importances, key=importances.get)        # ascending → bottom = least important
values = [importances[f] for f in order]

fig, ax = plt.subplots(figsize=(7, 4))
ax.barh(order, values)
ax.axvline(0, color="black", linewidth=0.8)
ax.set_xlabel("Mean R² drop when feature is permuted")
ax.set_title(f"Permutation importance  (test R² = {r2_test:.3f})")
fig.tight_layout()
fig.savefig(PLOT_DIR / "linreg_tips_importance.png")
plt.show()

# %% Heteroscedasticity — residuals vs. fitted values (full training set)
# If variance is constant (homoscedastic), points scatter evenly around zero.
# A funnel shape means larger predictions have larger errors → assumption violated.
y_hat = model.predict(X_train)
residuals = y_train - y_hat

fig, ax = plt.subplots(figsize=(7, 4))
ax.scatter(y_hat, residuals, alpha=0.5, s=18)
ax.axhline(0, color="black", linewidth=0.8)
ax.set_xlabel("Fitted value (predicted tip $)")
ax.set_ylabel("Residual (actual − predicted)")
ax.set_title("Residuals vs. fitted — heteroscedasticity check")
fig.tight_layout()
fig.savefig(PLOT_DIR / "linreg_tips_residuals.png")
plt.show()

# %% Partial regression plot — size controlling for all other features (Frisch-Waugh-Lovell)
"""
To isolate the effect of `size` we residualise both variables against
*all other regressors* (not just total_bill — that was the bug):

  1. Regress size ~ all other features  → residuals e_size
     (the part of party size *not* explained by anything else)
  2. Regress tip  ~ all other features  → residuals e_tip
     (the part of the tip *not* explained by anything else)
  3. Plot e_tip vs e_size — the slope now equals the `size` coefficient
     in the full multiple regression exactly (Frisch-Waugh-Lovell theorem).

The question becomes: for parties of *unexpected* size given everything else,
do they tip more or less than expected?
"""
size_idx       = col_list.index("size")
X_train_others = np.delete(X_train, size_idx, axis=1)   # all columns except size, train only

e_size = X_train[:, size_idx] - LinearRegression().fit(X_train_others, X_train[:, size_idx]).predict(X_train_others)
e_tip  = y_train - LinearRegression().fit(X_train_others, y_train).predict(X_train_others)

slope_fwl = LinearRegression().fit(e_size.reshape(-1, 1), e_tip).coef_[0]
print(f"\nFWL slope for size : {slope_fwl:.4f}")
print(f"Full-model coef    : {model.coef_[size_idx]:.4f}  (should match)")

fig, ax = plt.subplots(figsize=(6, 5))
ax.scatter(e_size, e_tip, alpha=0.5, s=18)
ax.axhline(0, color="black", linewidth=0.6)
ax.axvline(0, color="black", linewidth=0.6)
x_line = np.linspace(e_size.min(), e_size.max(), 100)
ax.plot(x_line, slope_fwl * x_line, color="tab:red")
ax.set_xlabel("size residual  (unexplained by all other features)")
ax.set_ylabel("tip residual   (unexplained by all other features)")
ax.set_title("Partial regression: size | all controls")
fig.tight_layout()
fig.savefig(PLOT_DIR / "linreg_tips_partial_size.png")
plt.show()

# %% Standardised coefficients (post-hoc, no refit needed)
# β_std_j = β_j × (σ_xj / σ_y)  →  units: std(y) per std(x_j)
# Allows direct magnitude comparison across features.
std_X = X_train.std(axis=0)
std_y = float(y_train.std())
beta_std = model.coef_ * std_X / std_y

print("\nStandardised coefficients:")
for name, b in sorted(zip(col_list, beta_std), key=lambda x: -abs(x[1])):
    print(f"  {name:<20} {b:+.4f}")
