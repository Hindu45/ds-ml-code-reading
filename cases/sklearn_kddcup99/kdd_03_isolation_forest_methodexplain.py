"""Two plots showing HOW Isolation Forest isolates anomalies (not just that it works).

KDD Cup 1999: full-forest anomaly-score landscape over (serror_rate, log_src_bytes).
"""
#%%
""" [1] Imports & config"""
import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.colors import Normalize
from pathlib import Path
from sklearn.ensemble import IsolationForest
from sklearn.metrics import roc_auc_score
from cases.sklearn_kddcup99.kdd_utils import NUMERIC_FEATURES as FEATURES, load_kddcup99

PLOT_DIR = Path(__file__).parent / "plots"
PLOT_DIR.mkdir(exist_ok=True)
RNG = np.random.default_rng(42)

df = load_kddcup99()
for col in ["src_bytes", "dst_bytes", "duration"]:
    df[f"log_{col}"] = np.log10(df[col].clip(lower=0) + 1)

y_kdd = df["is_attack"].values

TOP_N     = 8   # how many pairs to fully visualize
TREE_SEED = 42  # seed for the single-tree draws; adjust to find a clear partition


# %%
""" [2] Feature-pair AUC ranking
Trains a 10-tree Isolation Forest on every pair of the 11 features and ranks
by ROC-AUC (normal-only fit, anomaly score = negative avg path length).
The top-8 pairs drive the visualisation in cell [2].
"""

def avg_path_length(iso_model: IsolationForest, X: np.ndarray) -> np.ndarray:
    """Mean decision-path depth across all trees: shorter = more anomalous."""
    total = np.zeros(len(X))
    for est in iso_model.estimators_:
        # decision_path returns a sparse indicator matrix (n_samples × n_nodes);
        # row sum = number of nodes visited = path length in edges + 1
        total += np.asarray(est.decision_path(X).sum(axis=1)).ravel() - 1
    return total / len(iso_model.estimators_)


pair_aucs = {}
for F1, F2 in itertools.combinations(FEATURES, 2):
    X_pair = df[[F1, F2]].values
    iso_pre = IsolationForest(n_estimators=10, contamination=0.10, random_state=42)
    iso_pre.fit(X_pair[y_kdd == 0])
    depths = avg_path_length(iso_pre, X_pair)
    pair_aucs[(F1, F2)] = roc_auc_score(y_kdd, -depths)

ranked = sorted(pair_aucs.items(), key=lambda kv: kv[1], reverse=True)
top_pairs = [pair for pair, _ in ranked[:TOP_N]]
print(f"Ranked {len(ranked)} feature pairs — top {TOP_N} selected for visualisation:")
for i, ((f1, f2), auc) in enumerate(ranked[:4], 1):
    print(f"  {i}. {f1}  ×  {f2}  AUC={auc:.3f}")

# %%
""" [3] Single-tree partition vs full-forest average depth
Left: axis-aligned cuts of one tree, coloured by isolation depth.
Right: average depth over 100 trees as a contour map.
Anomalous points (red ×) should cluster in shallow (green) regions.
"""

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



cmap_BC = plt.cm.RdYlGn
norm_BC  = Normalize(0, 8)
idx_sub  = RNG.choice(len(df), 3000, replace=False)
sel_n    = idx_sub[y_kdd[idx_sub] == 0]
sel_a    = idx_sub[y_kdd[idx_sub] == 1]
res      = 200

for rank, (F1, F2) in enumerate(top_pairs, 1):
    X_kdd = df[[F1, F2]].values
    x_lo, x_hi = X_kdd[:, 0].min(), X_kdd[:, 0].max()
    y_lo, y_hi = X_kdd[:, 1].min(), X_kdd[:, 1].max()

    # ── left: single tree ───────────────────────────────────────────────────
    iso1 = IsolationForest(n_estimators=1, max_samples=256,
                           random_state=TREE_SEED, contamination=0.10)
    iso1.fit(X_kdd[y_kdd == 0])
    tr = iso1.estimators_[0].tree_

    x_pad = (x_hi - x_lo) * 0.01
    y_pad = (y_hi - y_lo) * 0.01
    leaves = collect_leaves(tr, 0,
                            x_lo - x_pad, x_hi + x_pad,
                            y_lo - y_pad, y_hi + y_pad, depth=0)

    # ── right: full forest avg depth ────────────────────────────────────────
    iso_full = IsolationForest(n_estimators=100, contamination=0.10, random_state=42)
    iso_full.fit(X_kdd[y_kdd == 0])

    xx, yy = np.meshgrid(np.linspace(x_lo, x_hi, res),
                         np.linspace(y_lo, y_hi, res))
    Z = avg_path_length(iso_full, np.c_[xx.ravel(), yy.ravel()]).reshape(res, res)

    # ── figure ───────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(15, 5), constrained_layout=True)
    gs  = fig.add_gridspec(1, 3, width_ratios=[1, 1, 0.04], wspace=0.05)
    ax_l  = fig.add_subplot(gs[0])
    ax_r  = fig.add_subplot(gs[1], sharex=ax_l, sharey=ax_l)
    ax_cb = fig.add_subplot(gs[2])

    for x0, x1, y0, y1, d in leaves:
        ax_l.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0,
                                  color=cmap_BC(norm_BC(d)), alpha=0.5,
                                  linewidth=0, zorder=0))

    ax_r.contourf(xx, yy, Z, levels=25, cmap=cmap_BC, norm=norm_BC, alpha=0.85)

    for ax in (ax_l, ax_r):
        ax.scatter(X_kdd[sel_n, 0], X_kdd[sel_n, 1],
                   c="navy", s=18, alpha=0.3, zorder=2, label="normal")
        ax.scatter(X_kdd[sel_a, 0], X_kdd[sel_a, 1],
                   c="crimson", s=80, marker="x", linewidths=2,
                   alpha=0.7, zorder=3, label="attack")
        ax.set_xlabel(F1, fontsize=11)
        ax.set_xlim(x_lo - x_pad, x_hi + x_pad)
        ax.set_ylim(y_lo - y_pad, y_hi + y_pad)

    ax_l.set_ylabel(F2, fontsize=11)
    ax_l.set_title(f"Single tree", fontsize=12)
    ax_r.set_title(f"Full forest: Average isolation depth", fontsize=12)
    ax_l.legend(fontsize=12, markerscale=1.5)

    sm = plt.cm.ScalarMappable(cmap=cmap_BC, norm=norm_BC)
    sm.set_array([])
    fig.colorbar(sm, cax=ax_cb, label="Isolation depth  (fewer cuts → anomalous)")
    fig.suptitle(f"{F1}  ×  {F2}    ·    KDD Cup 1999", fontsize=15)
    slug = f"{F1}__{F2}".replace("_", "-")
    path = PLOT_DIR / f"if_explainer_BC_{slug}.png"
    fig.savefig(path, dpi=120)
    plt.show()
    plt.close(fig)
    print(f"Saved: {path}")
