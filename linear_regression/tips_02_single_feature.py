"""Research question: Can we predict the tip amount from the restaurant bill total?"""

# %% Imports & config
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.model_selection import train_test_split

PLOT_DIR = Path(__file__).parent / "plots"
PLOT_DIR.mkdir(exist_ok=True)

# %% Load data
df = sns.load_dataset("tips")
print(df.head())
print(df[["total_bill", "tip"]].describe().round(2))

# %% Train / test split  (80 / 20)
X = df["total_bill"].values.reshape(-1, 1)   # shape (n, 1)
y = df["tip"].values                          # shape (n,)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# %% Standardise features — fit statistics on train only
mu: float = float(X_train.mean())
sigma: float = float(X_train.std())

X_train_s = (X_train - mu) / sigma
X_test_s  = (X_test  - mu) / sigma


def add_bias(X: np.ndarray) -> np.ndarray:
    """Prepend a column of ones for the intercept term."""
    return np.column_stack([np.ones(len(X)), X])


X_train_b = add_bias(X_train_s)   # shape (n_train, 2)
X_test_b  = add_bias(X_test_s)    # shape (n_test,  2)

# %% Loss function: Mean Squared Error
def mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """MSE = (1/n) * sum((y - ŷ)²)"""
    return float(np.mean((y_true - y_pred) ** 2))


# %% Analytical solution — Normal equation: θ = (XᵀX)⁻¹ Xᵀy
theta_ols = np.linalg.lstsq(X_train_b, y_train, rcond=None)[0]

print(f"\nOLS  intercept={theta_ols[0]:.4f}  slope={theta_ols[1]:.4f}")
print(f"OLS  train MSE={mse(y_train, X_train_b @ theta_ols):.4f}")

# %% Gradient descent from scratch
def gradient_descent(
    X: np.ndarray,
    y: np.ndarray,
    alpha: float = 0.1,
    n_iter: int = 300,
) -> tuple[np.ndarray, list[float]]:
    """Batch gradient descent minimising MSE.

    Args:
        X: Design matrix with bias column, shape (n, p).
        y: Target vector, shape (n,).
        alpha: Learning rate.
        n_iter: Number of full-batch iterations.

    Returns:
        Tuple of (final theta vector, MSE history per iteration).
    """
    n = len(y)
    theta = np.zeros(X.shape[1])
    history: list[float] = []

    for _ in range(n_iter):
        residual = X @ theta - y            # (n,)
        grad = (2 / n) * X.T @ residual    # ∂MSE/∂θ = (2/n) Xᵀ(Xθ − y)
        theta = theta - alpha * grad
        history.append(mse(y, X @ theta))

    return theta, history


theta_gd, loss_history = gradient_descent(X_train_b, y_train, alpha=0.1, n_iter=300)

print(f"\nGD   intercept={theta_gd[0]:.4f}  slope={theta_gd[1]:.4f}")
print(f"GD   train MSE={loss_history[-1]:.4f}")

# %% Visualise GD convergence — loss curve
fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(loss_history)
ax.axhline(mse(y_train, X_train_b @ theta_ols), color="tab:red", linestyle="--", label="OLS optimum")
ax.set_xlabel("Iteration")
ax.set_ylabel("MSE (train)")
ax.set_title("Gradient descent convergence")
ax.legend()
fig.tight_layout()
fig.savefig(PLOT_DIR / "linreg_tips_gd_convergence.png")
plt.show()

# %% Visualise GD steps — regression line at selected iterations
SNAPSHOTS = [0, 5, 20, 80, 300]

fig, axes = plt.subplots(1, len(SNAPSHOTS), figsize=(14, 3), sharey=True)

x_range = np.linspace(X_train_s.min(), X_train_s.max(), 100).reshape(-1, 1)
x_range_b = add_bias(x_range)

theta_snap = np.zeros(2)
snap_idx = 0

for i in range(max(SNAPSHOTS) + 1):
    residual = X_train_b @ theta_snap - y_train
    grad = (2 / len(y_train)) * X_train_b.T @ residual
    theta_snap = theta_snap - 0.1 * grad

    if i + 1 in SNAPSHOTS:
        ax = axes[SNAPSHOTS.index(i + 1)]
        ax.scatter(X_train_s, y_train, alpha=0.3, s=10)
        ax.plot(x_range, x_range_b @ theta_snap, color="tab:orange")
        ax.set_title(f"iter {i + 1}")
        ax.set_xlabel("bill (std)")

axes[0].set_ylabel("tip ($)")
fig.suptitle("GD — regression line evolution")
fig.tight_layout()
fig.savefig(PLOT_DIR / "linreg_tips_gd_steps.png")
plt.show()

# %% Metrics by hand: RMSE and R²
def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Root Mean Squared Error — same units as the target."""
    return float(np.sqrt(mse(y_true, y_pred)))


def r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """R² = 1 − SS_res / SS_tot.  1.0 is perfect; 0.0 means predicting the mean."""
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    return 1.0 - ss_res / ss_tot


print("\n--- Test-set evaluation ---")
for label, theta in [("OLS", theta_ols), ("GD ", theta_gd)]:
    y_pred = X_test_b @ theta
    print(f"{label}  RMSE={rmse(y_test, y_pred):.3f}  R²={r2(y_test, y_pred):.3f}")

# %% Regression line on test data
fig, ax = plt.subplots(figsize=(7, 5))
ax.scatter(X_test, y_test, alpha=0.6, label="test data")

x_orig = np.linspace(X_test.min(), X_test.max(), 100).reshape(-1, 1)
x_orig_b = add_bias((x_orig - mu) / sigma)

ax.plot(x_orig, x_orig_b @ theta_ols, color="tab:red",    label="OLS")
# ax.plot(x_orig, x_orig_b @ theta_gd,  color="tab:orange", linestyle="--", label="GD")
ax.set_xlabel("Total bill ($)")
ax.set_ylabel("Tip ($)")
ax.set_title("Linear regression: tip ~ total_bill")
ax.legend()
fig.tight_layout()
fig.savefig(PLOT_DIR / "linreg_tips_fit.png")
plt.show()

# %%
