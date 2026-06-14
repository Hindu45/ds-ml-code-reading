"""Shared plotting helpers for model-evaluation diagnostics (learning/validation curves, split stability).

Pure plotting functions: callers compute scores (e.g. via sklearn's
learning_curve / validation_curve / cross_val_score) and pass the aggregated
arrays in. Nothing here is specific to a model family or dataset.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def plot_learning_curve_panels(
    panels: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, str]],
    suptitle: str,
    save_path: Path,
    ylabel: str = "RMSE",
) -> list[tuple[str, float, float, int]]:
    """Side-by-side train/CV learning-curve bands (±1 std), one panel per config.

    Args:
        panels: List of (sizes, tr_mean, tr_std, cv_mean, cv_std, panel_title)
                tuples, already aggregated and sign-flipped to positive error units.
        suptitle: Figure-level title.
        save_path: File path for the saved PNG.
        ylabel: Y-axis label (shown on the leftmost panel only).

    Returns:
        List of (panel_title, train_final, cv_final, n_final) summaries, one per
        panel — the final-training-size values, for the caller to print.
    """
    fig, axes = plt.subplots(1, len(panels), figsize=(5.5 * len(panels), 5), sharey=True)
    if len(panels) == 1:
        axes = [axes]
    summaries = []
    for ax, (sizes, tr_mean, tr_std, cv_mean, cv_std, label) in zip(axes, panels):
        ax.plot(sizes, tr_mean, "o-",  color="C0", label="Train RMSE")
        ax.fill_between(sizes, tr_mean - tr_std, tr_mean + tr_std, alpha=0.15, color="C0")
        ax.plot(sizes, cv_mean, "s--", color="C1", label="CV RMSE")
        ax.fill_between(sizes, cv_mean - cv_std, cv_mean + cv_std, alpha=0.10, color="C1")

        ax.set_xlabel("Training set size")
        ax.set_title(label, fontsize=11)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)

        summaries.append((label.replace("\n", " "), float(tr_mean[-1]), float(cv_mean[-1]), int(sizes[-1])))

    axes[0].set_ylabel(ylabel)
    fig.suptitle(suptitle, fontsize=13)
    fig.tight_layout()
    fig.savefig(save_path)
    plt.show()
    return summaries


def plot_validation_curve(
    param_range: np.ndarray,
    tr_mean: np.ndarray,
    tr_std: np.ndarray,
    cv_mean: np.ndarray,
    cv_std: np.ndarray,
    best_param: float,
    best_label: str,
    xlabel: str,
    ylabel: str,
    title: str,
    save_path: Path,
    logx: bool = False,
) -> None:
    """Single train/CV validation-curve plot with ±1 std bands and a best-value marker.

    Args:
        param_range: Swept hyperparameter values (x-axis).
        tr_mean, tr_std, cv_mean, cv_std: Aggregated train/CV scores, sign-flipped
            to positive error units.
        best_param: Hyperparameter value to mark with a vertical reference line.
        best_label: Legend label for that reference line (e.g. "Best α = 0.010").
        xlabel, ylabel, title: Axis and figure labels.
        save_path: File path for the saved PNG.
        logx: Plot the x-axis on a log scale (e.g. for a wide α sweep).
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    plot_fn = ax.semilogx if logx else ax.plot
    plot_fn(param_range, tr_mean, "o-",  color="C0", label="Train RMSE")
    ax.fill_between(param_range, tr_mean - tr_std, tr_mean + tr_std, alpha=0.15, color="C0")
    plot_fn(param_range, cv_mean, "s--", color="C1", label="CV RMSE")
    ax.fill_between(param_range, cv_mean - cv_std, cv_mean + cv_std, alpha=0.10, color="C1")
    ax.axvline(best_param, color="gray", linestyle=":", linewidth=1.5, label=best_label)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=12)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path)
    plt.show()


def plot_cv_stability(
    single_scores: list[float],
    cv_scores: list[float],
    cv_folds: int,
    ylabel: str,
    title: str,
    save_path: Path,
) -> None:
    """Boxplot comparing single-split vs. k-fold-CV evaluation stability across seeds.

    Args:
        single_scores: RMSE from one random 80/20 split, one value per seed.
        cv_scores: Mean RMSE across the k folds, one value per seed.
        cv_folds: Number of CV folds (used for the box label).
        ylabel, title: Axis and figure labels.
        save_path: File path for the saved PNG.
    """
    fig, ax = plt.subplots(figsize=(6, 5))
    bp = ax.boxplot(
        [single_scores, cv_scores],
        tick_labels=["Single 80/20 split", f"{cv_folds}-fold CV (mean)"],
        patch_artist=True,
        widths=0.4,
    )
    bp["boxes"][0].set_facecolor("tab:orange")
    bp["boxes"][1].set_facecolor("tab:blue")
    for patch in bp["boxes"]:
        patch.set_alpha(0.6)

    ax.set_ylabel(ylabel)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(save_path)
    plt.show()
