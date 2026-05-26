"""Research question: How much does each feature contribute to predicting tip,
and what R² can we achieve with all available predictors?"""

# %%
""" [1] Imports & config
Import NumPy, pandas, seaborn, matplotlib, pathlib, and sklearn.
Define feature lists, permutation repeat count, and the RNG seed.
"""
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

# %%
""" [2] Load & one-hot encode
Load tips, drop the raw tip column, and one-hot encode all categorical predictors.
drop_first=True makes the design matrix identifiable by removing one redundant dummy per group.
"""
df = sns.load_dataset("tips")

X: pd.DataFrame = pd.get_dummies(
    df.drop(columns="tip"),
    columns=CATEGORICAL,
    drop_first=True,
    dtype=float,
)
y = df["tip"].values
# y = df["tip"].values / X["total_bill"].values * 100 # tip_pct

print("Features after encoding:")
print(X.columns.tolist())

# %%
""" [3] Train / test split
Split the encoded feature matrix and tip target 80/20.
The test set is held out until final evaluation so reported metrics are honest.
"""
X_train, X_test, y_train, y_test = train_test_split(
    X.values, y, test_size=0.2, random_state=42
)
print(f"train={X_train.shape}  test={X_test.shape}")

# %%
""" [4] Fit & baseline R²
Fit sklearn LinearRegression on the training set and compute R² on both splits.
The train/test gap is one overfitting signal, but on small datasets it can be misleadingly small; comparing test R² against a simpler single-feature baseline is an equally important check.
"""
model = LinearRegression()
model.fit(X_train, y_train)

r2_train = r2_score(y_train, model.predict(X_train))
r2_test  = r2_score(y_test,  model.predict(X_test))

print(f"\nR² train = {r2_train:.3f}")
print(f"R² test  = {r2_test:.3f}")

# %%
""" [5] Regression coefficients — OLS solution
Extract and plot raw OLS coefficients — one bar per encoded column.
Raw-scale coefficients show direction but are not comparable across features
with different units (total_bill in $ vs. binary dummies).
"""
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
print(f"Saved: {PLOT_DIR / 'linreg_tips_coefficients.png'}")
top3 = sorted(zip(feat_names, coefs), key=lambda x: -abs(x[1]))[:3]
print("Top-3 by magnitude: " + "  ".join(f"{n}={c:+.3f}" for n, c in top3))

# %%
""" [6] Map each feature → column indices in X
Build a lookup from each original feature name to its column indices in X.
Numeric features map to a single index; categorical features map to all their dummy columns.
Used in the permutation importance loop to shuffle all dummies of a feature together.
"""
feature_groups: dict[str, list[int]] = {}
col_list = list(X.columns)

for feat in NUMERIC:
    feature_groups[feat] = [col_list.index(feat)]

for feat in CATEGORICAL:
    feature_groups[feat] = [i for i, c in enumerate(col_list) if c.startswith(f"{feat}_")]

# %%
""" [7] Permutation importance
For each feature, shuffle its columns in X_test N_REPEATS times and average the R² drop.
A large drop means the model relied on that feature; near-zero means it was irrelevant.
Does not require refitting — the permutation is applied to the test set only.
"""
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

# %%
""" [8] Plot feature importance
Horizontal bar chart of permutation importance, sorted ascending (least important at bottom).
The title shows the baseline test R² for reference.
"""
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
print(f"Saved: {PLOT_DIR / 'linreg_tips_importance.png'}")

# %%
""" [9] Heteroscedasticity — residuals vs. fitted values (full training set)
Plot residuals (actual − predicted) against fitted values on the training set.
Constant vertical spread (homoscedasticity) is a key OLS assumption.
A funnel shape means variance grows with the fitted value, violating the assumption.
"""
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
print(f"residuals: mean={residuals.mean():.3f}  std={residuals.std():.3f}  |max|={np.abs(residuals).max():.3f}")
print(f"Saved: {PLOT_DIR / 'linreg_tips_residuals.png'}")

# %%
""" [10] Partial regression: size | all controls (Frisch-Waugh-Lovell)
Residualise both size and tip against all other features, then regress the residuals.
By Frisch-Waugh-Lovell, the slope equals the size coefficient in the full model exactly.
"""
size_idx       = col_list.index("size")
X_train_others = np.delete(X_train, size_idx, axis=1)   # all columns except size, train only

e_size = X_train[:, size_idx] - LinearRegression().fit(X_train_others, X_train[:, size_idx]).predict(X_train_others)
e_tip  = y_train - LinearRegression().fit(X_train_others, y_train).predict(X_train_others)

slope_fwl = LinearRegression().fit(e_size.reshape(-1, 1), e_tip).coef_[0]

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
print(f"FWL slope for size: {slope_fwl:.4f}  (full-model coef: {model.coef_[size_idx]:.4f})")
print(f"Saved: {PLOT_DIR / 'linreg_tips_partial_size.png'}")

# %%
""" [11] Standardised coefficients (post-hoc, no refit needed)
Multiply each raw coefficient by σ_xj / σ_y to put all predictors on a common scale.
β_std_j is interpretable as: a one-std change in x_j shifts tip by β_std_j standard deviations.
"""
std_X = X_train.std(axis=0)
std_y = float(y_train.std())
beta_std = model.coef_ * std_X / std_y

print("\nStandardised coefficients (top 4 by magnitude):")
for name, b in sorted(zip(col_list, beta_std), key=lambda x: -abs(x[1]))[:4]:
    print(f"  {name:<20} {b:+.4f}")
