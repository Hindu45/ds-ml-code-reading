import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.colors import Normalize
from sklearn.ensemble import IsolationForest

PLOT_DIR = Path(__file__).parent / "plots"
PLOT_DIR.mkdir(exist_ok=True)
RNG = np.random.default_rng(42)

X_normal   = RNG.multivariate_normal([0, 0], [[1, 0.4], [0.4, 1]], 160)
X_outliers = np.array([[ 4.2,  3.8], [-4.0,  3.5], [ 3.5, -4.0],
                        [-3.8, -3.5], [ 5.0, -0.5], [-5.0,  1.5]])
X_toy = np.vstack([X_normal, X_outliers])
y_toy = np.array([0] * 160 + [1] * 6)

iso1 = IsolationForest(n_estimators=1, max_samples=len(X_toy),
                       random_state=42, contamination=0.04)
iso1.fit(X_toy)
tree = iso1.estimators_[0].tree_


def collect_leaves(tr, node: int, x0: float, x1: float,
                   y0: float, y1: float, depth: int) -> list:
    if tr.children_left[node] == -1:
        return [(x0, x1, y0, y1, depth)]
    f = tr.feature[node]
    t = tr.threshold[node]
    if f == 0:
        tc = np.clip(t, x0, x1)
        return (collect_leaves(tr, tr.children_left[node],  x0, tc, y0, y1, depth + 1) +
                collect_leaves(tr, tr.children_right[node], tc, x1, y0, y1, depth + 1))
    else:
        tc = np.clip(t, y0, y1)
        return (collect_leaves(tr, tr.children_left[node],  x0, x1, y0, tc, depth + 1) +
                collect_leaves(tr, tr.children_right[node], x0, x1, tc, y1, depth + 1))


pad = 1.2
x0g, x1g = X_toy[:, 0].min() - pad, X_toy[:, 0].max() + pad
y0g, y1g = X_toy[:, 1].min() - pad, X_toy[:, 1].max() + pad
leaves = collect_leaves(tree, 0, x0g, x1g, y0g, y1g, depth=0)
max_d  = max(d for *_, d in leaves)

cmap_A = plt.cm.RdYlGn
fig, ax = plt.subplots(figsize=(7, 6))
for x0, x1, y0, y1, d in leaves:
    ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0,
                            color=cmap_A(d / max_d), alpha=0.5,
                            linewidth=0, zorder=0))

ax.scatter(X_toy[y_toy == 0, 0], X_toy[y_toy == 0, 1],
           c="navy", s=18, alpha=0.7, zorder=2, label="normal")
ax.scatter(X_toy[y_toy == 1, 0], X_toy[y_toy == 1, 1],
           c="crimson", s=80, marker="x", linewidths=2, zorder=3, label="outlier")

sm = plt.cm.ScalarMappable(cmap=cmap_A, norm=Normalize(0, max_d))
sm.set_array([])
plt.colorbar(sm, ax=ax, label=f"Isolation depth  (0 = 1 cut, {max_d} = {max_d} cuts)")
ax.set_xlim(x0g, x1g)
ax.set_ylim(y0g, y1g)
ax.set_xlabel("Feature 1")
ax.set_ylabel("Feature 2")
ax.set_title("Isolation Forest — single tree  (synthetic data)\n"
             "Red: few cuts needed → anomalous    Green: many cuts → normal")
ax.legend(fontsize=9)
fig.tight_layout()
fig.savefig(PLOT_DIR / "if_explainer_A_toy_partitions.png", dpi=150)
plt.show()
