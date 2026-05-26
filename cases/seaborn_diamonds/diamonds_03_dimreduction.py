"""What structure do diamond features reveal in 2D projections (PCA and t-SNE) without using price?"""

# %%
""" [1] Imports & config"""
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

PLOT_DIR     = Path(__file__).parent / "plots"
PLOT_DIR.mkdir(exist_ok=True)

RANDOM_STATE = 42
TSNE_SAMPLE  = 5_000    # t-SNE is slow at full scale (~54k); increase if hardware allows

NUMERIC_COLS  = ["carat", "depth", "table", "x", "y", "z"]
CUT_ORDER     = ["Fair", "Good", "Very Good", "Premium", "Ideal"]
COLOR_ORDER   = ["J", "I", "H", "G", "F", "E", "D"]        # D is colorless / best
CLARITY_ORDER = ["I1", "SI2", "SI1", "VS2", "VS1", "VVS2", "VVS1", "IF"]

# %%
""" [2] Load & clean
Same cleanup as regression script: drop 20 rows with zero-dimension measurements.
"""
df = sns.load_dataset("diamonds")
df = df[(df[["x", "y", "z"]] > 0).all(axis=1)].copy()
print(f"Rows after cleanup: {len(df):,}")

# %%
""" [3] Feature prep — ordinal encode categoricals, scale all features
Quality grades carry a meaningful ordering so ordinal encoding (integer rank) is
appropriate here and keeps the feature matrix compact (9 features vs 23 with OHE),
making the PCA biplot readable with one arrow per original variable.
Price is excluded — the goal is to see whether price-relevant structure emerges
from the quality and size features alone.
All 9 features are z-scored so PCA variance is not dominated by scale differences
(e.g. carat measured in fractions vs. x/y/z in millimetres).
"""
df["cut_ord"]     = pd.Categorical(df["cut"],     categories=CUT_ORDER,     ordered=True).codes
df["color_ord"]   = pd.Categorical(df["color"],   categories=COLOR_ORDER,   ordered=True).codes
df["clarity_ord"] = pd.Categorical(df["clarity"], categories=CLARITY_ORDER, ordered=True).codes

ALL_FEATURES = NUMERIC_COLS + ["cut_ord", "color_ord", "clarity_ord"]
X            = df[ALL_FEATURES].values.astype(float)
X_scaled     = StandardScaler().fit_transform(X)
log_price    = np.log1p(df["price"].values)

print(f"Feature matrix: {X_scaled.shape}   features: {ALL_FEATURES}")

# %%
""" [4] PCA — variance explained
Scree plot with cumulative variance. The 'elbow' indicates intrinsic
dimensionality. The four collinear size features (carat, x, y, z) should
collapse onto a single dominant component, leaving the quality grades for
subsequent components.
"""
pca     = PCA(random_state=RANDOM_STATE)
X_pca   = pca.fit_transform(X_scaled)
evr     = pca.explained_variance_ratio_
cum_var = np.cumsum(evr)

fig, ax = plt.subplots(figsize=(7, 4))
ax.bar(range(1, len(evr) + 1), evr, alpha=0.7, label="per component")
ax.plot(range(1, len(evr) + 1), cum_var, "o-", color="tab:orange", label="cumulative")
ax.axhline(0.90, linestyle="--", color="gray", linewidth=0.8, label="90% threshold")
ax.set_xlabel("Principal component")
ax.set_ylabel("Explained variance ratio")
ax.set_title("PCA — variance explained")
ax.legend()
fig.tight_layout()
fig.savefig(PLOT_DIR / "diamonds_pca_variance.png")
plt.show()

print(f"PC1 explains {evr[0]:.1%} of variance")
print(f"Components to reach 90%: {int(np.argmax(cum_var >= 0.90)) + 1}")
print(f"Saved: {PLOT_DIR / 'diamonds_pca_variance.png'}")

# %%
""" [5] PCA — scatter PC1 vs PC2 coloured by price
Each point is one diamond projected onto the first two PCs.
Colour encodes log(price) to compress the right-skewed price range.
A clear price gradient along PC1 would confirm that 'size' (the dominant axis)
drives price — consistent with the EDA finding that carat is the top predictor.
"""
fig, ax = plt.subplots(figsize=(7, 6))
sc = ax.scatter(X_pca[:, 0], X_pca[:, 1], c=log_price, cmap="plasma", alpha=0.3, s=3)
plt.colorbar(sc, ax=ax, label="log(1 + price)")
ax.set_xlabel(f"PC1  ({evr[0]:.1%})")
ax.set_ylabel(f"PC2  ({evr[1]:.1%})")
ax.set_title("PCA — coloured by price")
fig.tight_layout()
fig.savefig(PLOT_DIR / "diamonds_pca_price.png")
plt.show()
corr_pc1 = np.corrcoef(X_pca[:, 0], log_price)[0, 1]
print(f"r(PC1, log-price): {corr_pc1:.3f}")
print(f"Saved: {PLOT_DIR / 'diamonds_pca_price.png'}")

# %%
""" [6] PCA — biplot
Loading vectors show each feature's contribution to PC1 and PC2.
Arrow direction = feature axis; length = influence on that component.
Arrows pointing the same way indicate positively correlated features.
Arrow scale is set so the longest arrow reaches ~80% of the PC score spread.
"""
loadings    = pca.components_[:2].T    # (n_features, 2)
score_range = min(np.percentile(np.abs(X_pca[:, 0]), 90),
                  np.percentile(np.abs(X_pca[:, 1]), 90))
arrow_scale = score_range * 0.8 / np.abs(loadings).max()

fig, ax = plt.subplots(figsize=(8, 7))
ax.scatter(X_pca[:, 0], X_pca[:, 1], c=log_price, cmap="plasma", alpha=0.15, s=2)
for i, name in enumerate(ALL_FEATURES):
    dx, dy = loadings[i] * arrow_scale
    ax.annotate("", xy=(dx, dy), xytext=(0, 0),
                arrowprops=dict(arrowstyle="->", color="crimson", lw=1.5))
    ax.text(dx * 1.12, dy * 1.12, name.replace("_ord", ""), fontsize=8, color="crimson")
ax.set_xlabel(f"PC1  ({evr[0]:.1%})")
ax.set_ylabel(f"PC2  ({evr[1]:.1%})")
ax.set_title("PCA biplot — feature loadings")
fig.tight_layout()
fig.savefig(PLOT_DIR / "diamonds_pca_biplot.png")
plt.show()
top3 = sorted(zip(ALL_FEATURES, loadings[:, 0]), key=lambda x: abs(x[1]), reverse=True)[:3]
print("Top PC1 loadings:", "  ".join(f"{n}: {v:.2f}" for n, v in top3))
print(f"Saved: {PLOT_DIR / 'diamonds_pca_biplot.png'}")

# %%
""" [7] t-SNE — subsample and fit
t-SNE preserves local neighbourhood distances and can reveal non-linear cluster
structure that PCA (linear) misses.
A random subsample of TSNE_SAMPLE rows keeps runtime manageable; the fixed seed
makes results reproducible. perplexity=30 (default) balances local vs. global
structure for datasets of this size.
"""
rng       = np.random.default_rng(RANDOM_STATE)
idx       = rng.choice(len(X_scaled), size=TSNE_SAMPLE, replace=False)
X_sub     = X_scaled[idx]
log_p_sub = log_price[idx]
cut_sub   = df["cut"].values[idx]

print(f"Running t-SNE on {TSNE_SAMPLE:,} samples …")
X_tsne = TSNE(n_components=2, perplexity=30, random_state=RANDOM_STATE).fit_transform(X_sub)
print("Done.")

# %%
""" [8] t-SNE — coloured by price
Compare to the PCA price scatter: if the gradient is similarly aligned, PCA
already captures the price-relevant structure and t-SNE adds little.
Tighter clusters within the price gradient would indicate finer-grained structure
that the linear PCA projection misses.
"""
fig, ax = plt.subplots(figsize=(7, 6))
sc = ax.scatter(X_tsne[:, 0], X_tsne[:, 1], c=log_p_sub, cmap="plasma", alpha=0.5, s=6)
plt.colorbar(sc, ax=ax, label="log(1 + price)")
ax.set_xlabel("t-SNE 1")
ax.set_ylabel("t-SNE 2")
ax.set_title("t-SNE — coloured by price")
fig.tight_layout()
fig.savefig(PLOT_DIR / "diamonds_tsne_price.png")
plt.show()
corr_t1 = np.corrcoef(X_tsne[:, 0], log_p_sub)[0, 1]
print(f"r(t-SNE 1, log-price): {corr_t1:.3f}")
print(f"Saved: {PLOT_DIR / 'diamonds_tsne_price.png'}")

# %%
""" [9] t-SNE — coloured by cut grade
Discrete colour-coding by cut grade reveals whether cut is a visible structural
axis in the feature space. If cut grades mix throughout the embedding rather than
forming clean clusters, cut alone is a weak direct signal — consistent with the
price paradox observed in EDA (large stones tend to receive lower-grade cuts).
"""
cut_codes = pd.Categorical(cut_sub, categories=CUT_ORDER).codes
palette   = plt.cm.tab10(np.arange(len(CUT_ORDER)) / 10.0)

fig, ax = plt.subplots(figsize=(7, 6))
for code, label in enumerate(CUT_ORDER):
    mask = cut_codes == code
    ax.scatter(X_tsne[mask, 0], X_tsne[mask, 1],
               color=palette[code], alpha=0.5, s=6, label=label)
ax.legend(title="Cut", markerscale=3, loc="best")
ax.set_xlabel("t-SNE 1")
ax.set_ylabel("t-SNE 2")
ax.set_title("t-SNE — coloured by cut grade")
fig.tight_layout()
fig.savefig(PLOT_DIR / "diamonds_tsne_cut.png")
plt.show()
counts = pd.Series(cut_sub).value_counts().reindex(CUT_ORDER)
print(counts.to_string())
print(f"Saved: {PLOT_DIR / 'diamonds_tsne_cut.png'}")
