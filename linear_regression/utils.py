"""Shared utilities for regularised linear regression analyses."""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.linear_model import Ridge, Lasso
from sklearn.metrics import root_mean_squared_error
from tqdm import tqdm


def make_ridge(alpha: float, n: int) -> Ridge:
    """Ridge regressor with per-sample α normalisation.

    sklearn Ridge minimises ||y − Xw||² + α_skl · ||w||²  (sum, not mean).
    Multiplying by n converts user_α to a per-sample penalty, matching
    Lasso's (1/2n) convention so α is comparable across both models.

    Args:
        alpha: Regularisation strength on a per-sample basis.
        n: Number of training samples.

    Returns:
        Configured Ridge instance (unfitted).
    """
    return Ridge(alpha=max(alpha, 1e-10) * n)


def ridge_grid(
    alphas: np.ndarray,
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    X_va: np.ndarray,
    y_va: np.ndarray,
    desc: str = "Ridge",
) -> tuple[list[float], list[float]]:
    """Sweep Ridge over an α grid; return train and val RMSE lists.

    Args:
        alphas: Sequence of regularisation strengths.
        X_tr, y_tr: Training split (z-scored).
        X_va, y_va: Validation split (z-scored).
        desc: Progress bar label.

    Returns:
        Tuple (train_rmse, val_rmse), each a list aligned with alphas.
    """
    n = len(y_tr)
    train_errs, val_errs = [], []
    for a in tqdm(alphas, desc=desc, ncols=70):
        m = make_ridge(a, n)
        m.fit(X_tr, y_tr)
        train_errs.append(root_mean_squared_error(y_tr, m.predict(X_tr)))
        val_errs.append(root_mean_squared_error(y_va, m.predict(X_va)))
    return train_errs, val_errs


def lasso_grid(
    alphas: np.ndarray,
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    X_va: np.ndarray,
    y_va: np.ndarray,
    max_iter: int = 1_000,
    tol: float = 1e-3,
    desc: str = "Lasso",
) -> tuple[list[float], list[float]]:
    """Sweep Lasso over an α grid using a warm-started model.

    Warm-starting means each fit continues from the previous solution rather
    than cold-initialising — much faster when stepping along a regularisation
    path (low → high α recommended).

    Args:
        alphas: Sequence of regularisation strengths (low → high).
        X_tr, y_tr: Training split (z-scored).
        X_va, y_va: Validation split (z-scored).
        max_iter: Coordinate-descent iterations per α.
        tol: Convergence tolerance.
        desc: Progress bar label.

    Returns:
        Tuple (train_rmse, val_rmse), each a list aligned with alphas.
    """
    train_errs, val_errs = [], []
    m = Lasso(alpha=1.0, warm_start=True, max_iter=max_iter, tol=tol)
    for a in tqdm(alphas, desc=desc, ncols=70):
        m.alpha = max(float(a), 1e-6)
        m.fit(X_tr, y_tr)
        train_errs.append(root_mean_squared_error(y_tr, m.predict(X_tr)))
        val_errs.append(root_mean_squared_error(y_va, m.predict(X_va)))
    return train_errs, val_errs


def plot_error_curves(
    specs: list[tuple],
    save_path: Path,
    ylabel: str = "RMSE",
    title: str = "Train / Val error curves: coarse vs. fine grid search",
) -> None:
    """2×2 grid of train/val RMSE curves for Ridge and Lasso.

    Args:
        specs: List of 4 tuples (alphas, train_errs, val_errs, model_name,
               step_label, best_alpha). Values should already be in the
               desired display units (unscale before passing if needed).
        save_path: File path for the saved PNG.
        ylabel: Y-axis label (e.g. "RMSE (USD)").
        title: Figure suptitle.
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for ax, (alphas, tr, va, model_name, step, best_a) in zip(axes.flat, specs):
        ax.plot(alphas, tr, label="train", color="tab:blue")
        ax.plot(alphas, va, label="val",   color="tab:orange")
        ax.axvline(best_a, color="tab:green", linestyle="--", linewidth=1.2,
                   label=f"best α = {best_a:.3f}")
        ax.set_xlabel("α (regularisation strength)")
        ax.set_ylabel(ylabel)
        ax.set_title(f"{model_name} — {step}")
        ax.legend(fontsize=8)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(save_path)
    plt.show()


def plot_coef_paths(
    alphas_ridge: np.ndarray,
    ridge_paths: np.ndarray,
    alphas_lasso: np.ndarray,
    lasso_paths: np.ndarray,
    feature_names: list[str],
    save_path: Path,
) -> None:
    """Side-by-side coefficient path plots for Ridge and Lasso.

    Args:
        alphas_ridge: α grid used for Ridge paths.
        ridge_paths: Array of shape (n_alphas, n_features) for Ridge.
        alphas_lasso: α grid used for Lasso paths.
        lasso_paths: Array of shape (n_alphas, n_features) for Lasso.
        feature_names: Feature labels for the legend.
        save_path: File path for the saved PNG.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, alphas_plot, paths, title in zip(
        axes,
        [alphas_ridge,                    alphas_lasso],
        [ridge_paths,                     lasso_paths],
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
    fig.savefig(save_path)
    plt.show()


def plot_pred_vs_actual(
    predictions: list[tuple[str, np.ndarray]],
    y_test: np.ndarray,
    save_path: Path,
    alpha: float=0.05
) -> None:
    """Scatter plot of predicted vs. actual values for each model.

    Args:
        predictions: List of (label, y_pred) tuples in original (unscaled) units.
        y_test: Ground-truth test targets in original units.
        save_path: File path for the saved PNG.
    """
    fig, axes = plt.subplots(1, len(predictions), figsize=(5.5 * len(predictions), 5))
    if len(predictions) == 1:
        axes = [axes]
    for ax, (label, y_pred) in zip(axes, predictions):
        lo = min(float(y_test.min()), float(y_pred.min()))
        hi = max(float(y_test.max()), float(y_pred.max()))
        ax.scatter(y_test, y_pred, alpha=alpha, s=5)
        ax.plot([lo, hi], [lo, hi], "r--", linewidth=1)
        ax.set_xlabel("Actual")
        ax.set_ylabel("Predicted")
        ax.set_title(f"{label} — predicted vs. actual (test)")
    fig.tight_layout()
    fig.savefig(save_path)
    plt.show()
