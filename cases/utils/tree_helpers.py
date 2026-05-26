"""Visualization helpers for decision tree classifiers."""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from matplotlib.figure import Figure
from sklearn.tree import DecisionTreeClassifier


def plot_dt_boundary_sequence(
    clf: DecisionTreeClassifier,
    X_train_2d: np.ndarray,
    y_train: np.ndarray,
    X_2d: np.ndarray,
    y: np.ndarray,
    feature_names: list[str],
    save_path: Path | None = None,
) -> Figure:
    """Plot a 3-panel sequential decision boundary for a depth-2 classifier.

    Each panel adds one split rule, matching the order in export_text output:
    root only → root + left subtree split → root + both subtree splits.
    Assumes max_depth=2, exactly 2 features, and the root splits on feature_names[0].

    Args:
        clf: Fitted DecisionTreeClassifier with max_depth=2.
        X_train_2d: (n_train, 2) training features; used to fit the depth-1 baseline.
        y_train: Training labels aligned with X_train_2d.
        X_2d: (n, 2) full-dataset features for scatter and grid range.
        y: Class labels for X_2d.
        feature_names: [x_axis_feature, y_axis_feature].
        save_path: If given, save the figure here and print the path.

    Returns:
        The matplotlib Figure.
    """
    feat_x, feat_y = feature_names

    xx, yy = np.meshgrid(
        np.linspace(X_2d[:, 0].min() - 2, X_2d[:, 0].max() + 2, 300),
        np.linspace(X_2d[:, 1].min() - 2, X_2d[:, 1].max() + 2, 300),
    )
    grid = np.c_[xx.ravel(), yy.ravel()]

    def _grid_Z(estimator: DecisionTreeClassifier) -> np.ndarray:
        return np.searchsorted(clf.classes_, estimator.predict(grid)).reshape(xx.shape)

    dt_d1 = DecisionTreeClassifier(max_depth=1, random_state=clf.random_state)
    dt_d1.fit(X_train_2d, y_train)

    Z_d1 = _grid_Z(dt_d1)
    Z_d2 = _grid_Z(clf)

    _t = clf.tree_
    root_thresh  = _t.threshold[0]
    left_thresh  = _t.threshold[_t.children_left[0]]
    right_thresh = _t.threshold[_t.children_right[0]]

    Z_left_only = Z_d2.copy()
    Z_left_only[xx > root_thresh] = Z_d1[xx > root_thresh]
    x_lo, x_hi = xx.min(), xx.max()

    panels = [
        (Z_d1,        f"1. Root:  {feat_x} ≤ {root_thresh:.1f}",  False, False),
        (Z_left_only, f"2. Left:  {feat_y} ≤ {left_thresh:.1f}",  True,  False),
        (Z_d2,        f"3. Right: {feat_y} ≤ {right_thresh:.1f}", True,  True),
    ]

    cls_colors = {sp: f"C{i}" for i, sp in enumerate(clf.classes_)}
    bg_colors  = [cls_colors[sp] for sp in clf.classes_]

    fig, axes = plt.subplots(1, 3, figsize=(14, 4), sharex=True, sharey=True)
    for ax, (Z, title, show_left, show_right) in zip(axes, panels):
        ax.contourf(xx, yy, Z, levels=[-0.5, 0.5, 1.5, 2.5], colors=bg_colors, alpha=0.25)
        for sp in clf.classes_:
            m = y == sp
            ax.scatter(X_2d[m, 0], X_2d[m, 1], label=sp, s=12,
                       color=cls_colors[sp], edgecolors="k", linewidths=0.3)
        ax.axvline(root_thresh, color="k", lw=1.2, ls="--")
        if show_left:
            ax.plot([x_lo, root_thresh], [left_thresh,  left_thresh],  color="k", lw=1.2, ls="--")
        if show_right:
            ax.plot([root_thresh, x_hi], [right_thresh, right_thresh], color="k", lw=1.2, ls="--")
        ax.set_xlabel(feat_x)
        ax.set_title(title, fontsize=9)
    axes[0].set_ylabel(feat_y)
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path)
        print(f"Saved: {save_path}")
    plt.show()
    return fig
