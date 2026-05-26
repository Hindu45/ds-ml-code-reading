"""Modeling: does adding degree-2 polynomial features improve diabetes progression prediction over linear OLS?
Purpose: modeling | Style: library-optimal | Flags: docstring-depth: lab

Expected: poly(2) gains several R² points over linear OLS; both beat dummy comfortably.
Surprising: if the gain is negligible, it implies the feature-target relationships are already well-captured by the linear terms alone.
"""

# %%
""" [1] Imports & config"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.datasets import load_diabetes
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures

RANDOM_STATE = 42
PLOT_DIR = Path(__file__).parent / "plots"
PLOT_DIR.mkdir(exist_ok=True)


# %%
""" [2] Load & split
Three-way split: 60% train / 20% val / 20% test.
Val is used for model selection; test is touched exactly once for final reporting.
All 10 features are pre-standardized (mean ~0, std ~0.048) -- no additional scaling needed.
"""
bunch = load_diabetes(as_frame=True)
X = bunch.data
y = bunch.target

X_trainval, X_test, y_trainval, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE
)
X_train, X_val, y_train, y_val = train_test_split(
    X_trainval, y_trainval, test_size=0.25, random_state=RANDOM_STATE  # 0.25 * 0.80 = 0.20
)
print(f"Train: {len(X_train)}  Val: {len(X_val)}  Test: {len(X_test)}")
print(f"Poly(2) feature count: 10 linear + 10 squared + 45 interactions = "
      f"{PolynomialFeatures(degree=2, include_bias=False).fit(X_train).n_output_features_} total")


# %%
""" [3] Define model ladder
Four rungs; the only change across rungs is the feature representation fed to OLS.
Dummy (predict-mean) sets the floor at R²=0. OLS(bmi) is a single-feature sanity check.
Linear OLS is the full-feature reference. Poly(2)+OLS adds all squared and pairwise
interaction terms: 10 -> 65 features.
"""
MODELS = [
    ("Dummy (mean)", DummyRegressor(strategy="mean")),
    ("OLS (bmi only)", Pipeline([
        ("select", ColumnTransformer([("bmi", "passthrough", ["bmi"])])),
        ("ols",    LinearRegression()),
    ])),
    ("Linear OLS",   LinearRegression()),
    ("Poly(2) + OLS", Pipeline([
        ("poly", PolynomialFeatures(degree=2, include_bias=False)),
        ("ols",  LinearRegression()),
    ])),
]


# %%
""" [4] Train & compare on val
All three models are fit on the training set and scored on the validation set.
Val scores determine which model is selected -- the test set is not touched here.
RMSE is in original target units (disease score points).
"""
results = []
for name, model in MODELS:
    model.fit(X_train, y_train)
    y_pred_val = model.predict(X_val)
    rmse = np.sqrt(mean_squared_error(y_val, y_pred_val))
    r2 = model.score(X_val, y_val)
    results.append({"model": name, "val_RMSE": rmse, "val_R2": r2})

results_df = pd.DataFrame(results)
print("Validation scores:")
print(results_df.to_string(index=False, float_format="{:.3f}".format))


# %%
""" [5] Val results chart
Two side-by-side bar charts: R² (higher is better) and RMSE (lower is better).
Keeping metrics in separate panels avoids mixing axes and makes each comparison self-contained.
"""
colors = ["#888888", "steelblue", "tomato"]
fig, (ax_r2, ax_rmse) = plt.subplots(1, 2, figsize=(11, 4))

ax_r2.bar(results_df["model"], results_df["val_R2"], color=colors)
ax_r2.set_ylabel("R² (validation set)")
ax_r2.set_ylim(0, results_df["val_R2"].max() + 0.1)
ax_r2.set_title("R² — higher is better")

ax_rmse.bar(results_df["model"], results_df["val_RMSE"], color=colors)
ax_rmse.set_ylabel("RMSE (validation set)")
ax_rmse.set_ylim(0, results_df["val_RMSE"].max() * 1.15)
ax_rmse.set_title("RMSE — lower is better")

fig.suptitle("Diabetes progression: Dummy vs OLS vs Poly(2)+OLS — validation")
fig.tight_layout()
fig.savefig(PLOT_DIR / "diabetes_model_ladder.png")
plt.show()
print(f"Saved: {PLOT_DIR / 'diabetes_model_ladder.png'}")


# %%
""" [6] Final evaluation on test set
The model with the highest val R² is selected. It is refit on the full train+val
set before being evaluated on the held-out test set -- test data is used exactly once.
"""
winner_name = results_df.loc[results_df["val_R2"].idxmax(), "model"]
winner_model = dict(MODELS)[winner_name]
winner_model.fit(X_trainval, y_trainval)

test_rmse = np.sqrt(mean_squared_error(y_test, winner_model.predict(X_test)))
test_r2   = winner_model.score(X_test, y_test)
print(f"Selected model: {winner_name}")
print(f"Test RMSE: {test_rmse:.2f}  |  Test R²: {test_r2:.3f}")


# %%
""" [7] Coefficient interpretation -- poly(2) model
Top 15 features by |coefficient| in the poly(2) model (fitted on train+val).
Signed values preserve direction: positive = higher value -> higher progression.
Large interaction terms (e.g. bmi s5) reveal non-linear signals absent from linear OLS.
"""
poly_pipe = dict(MODELS)["Poly(2) + OLS"]
if winner_name != "Poly(2) + OLS":
    poly_pipe.fit(X_trainval, y_trainval)
feature_names = poly_pipe.named_steps["poly"].get_feature_names_out(X.columns)
coefs = pd.Series(poly_pipe.named_steps["ols"].coef_, index=feature_names)
top15 = coefs.abs().sort_values(ascending=False).head(15)
top15_signed = coefs.loc[top15.index].sort_values()

fig, ax = plt.subplots(figsize=(7, 5))
bar_colors = ["tomato" if v > 0 else "steelblue" for v in top15_signed]
ax.barh(top15_signed.index, top15_signed.values, color=bar_colors)
ax.axvline(0, color="black", linewidth=0.8)
ax.set_xlabel("Coefficient (signed)")
ax.set_title("Top 15 poly(2) features by |coefficient|")
fig.tight_layout()
fig.savefig(PLOT_DIR / "diabetes_poly_coefficients.png")
plt.show()
print(f"Saved: {PLOT_DIR / 'diabetes_poly_coefficients.png'}")
top3_idx = coefs.abs().sort_values(ascending=False).head(3).index
print(f"Top 3 poly features by |coef|:\n{coefs.loc[top3_idx].round(1).to_string()}")


# %%
""" [8] Residual plot -- linear OLS vs poly(2) on test set
Residuals for both non-trivial models on the held-out test set.
The winner was refit on train+val; linear OLS is refit here for a fair side-by-side.
A random horizontal band around zero indicates a well-specified model.
"""
linear_model = dict(MODELS)["Linear OLS"]
linear_model.fit(X_trainval, y_trainval)

pred_linear = linear_model.predict(X_test)
pred_poly   = poly_pipe.predict(X_test)
resid_linear = y_test.values - pred_linear
resid_poly   = y_test.values - pred_poly

fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
for ax, preds, resids, label in [
    (axes[0], pred_linear, resid_linear, "Linear OLS"),
    (axes[1], pred_poly,   resid_poly,   "Poly(2) + OLS"),
]:
    ax.scatter(preds, resids, alpha=0.4, s=18, color="steelblue")
    ax.axhline(0, color="tomato", linewidth=1)
    ax.set_xlabel("Predicted progression score")
    ax.set_ylabel("Residual")
    ax.set_title(label)

fig.suptitle("Residuals vs predicted -- test set")
fig.tight_layout()
fig.savefig(PLOT_DIR / "diabetes_residuals.png")
plt.show()
print(f"Saved: {PLOT_DIR / 'diabetes_residuals.png'}")
rmse_lin = np.sqrt(mean_squared_error(y_test, pred_linear))
rmse_pol = np.sqrt(mean_squared_error(y_test, pred_poly))
print(f"Test RMSE — Linear OLS: {rmse_lin:.2f}  |  Poly(2)+OLS: {rmse_pol:.2f}")
