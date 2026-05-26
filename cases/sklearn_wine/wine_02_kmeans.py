"""Market segmentation with k-means: can unsupervised clustering recover wine cultivar groups from physicochemical features alone?"""

# %%
""" [1] Imports & config"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.datasets import load_wine
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, silhouette_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

PLOT_DIR = Path(__file__).parent / "plots"
PLOT_DIR.mkdir(exist_ok=True)
RANDOM_STATE = 42


# %%
""" [2] Load, split, scale
Split 80/20 (stratified on cultivar label) then fit StandardScaler on train only.
Applying the scaler before clustering is the critical step: proline alone (~700 units)
would otherwise dominate Euclidean distance and make all other features irrelevant.
"""
bunch = load_wine(as_frame=True)
df = bunch.frame.copy()
FEATURE_COLS = list(bunch.feature_names)
CLASS_NAMES = list(bunch.target_names)
X = df[FEATURE_COLS].values
y = df["target"].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
)
scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc = scaler.transform(X_test)

print(f"Train: {X_train_sc.shape}  |  Test: {X_test_sc.shape}")
print(f"Train class counts: {dict(zip(*np.unique(y_train, return_counts=True)))}")


# %%
""" [3] Choose k: elbow and silhouette
Inertia (within-cluster sum of squares) decreases with k; the elbow marks diminishing returns.
Silhouette score measures how well each point fits its own cluster vs the nearest other cluster.
Both are computed on the scaled training set so the choice of k is not informed by test labels.
"""
k_range = range(2, 9)
inertias, silhouettes = [], []
for k in k_range:
    km = KMeans(n_clusters=k, n_init=10, random_state=RANDOM_STATE)
    labels = km.fit_predict(X_train_sc)
    inertias.append(km.inertia_)
    silhouettes.append(silhouette_score(X_train_sc, labels))

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
ax1.plot(list(k_range), inertias, marker="o", color="steelblue")
ax1.set_xlabel("k")
ax1.set_ylabel("Inertia (WCSS)")
ax1.set_title("Elbow plot")

ax2.plot(list(k_range), silhouettes, marker="o", color="tomato")
ax2.set_xlabel("k")
ax2.set_ylabel("Silhouette score")
ax2.set_title("Silhouette score by k")

fig.suptitle("Choosing k for k-means (scaled features, train set)")
fig.tight_layout()
fig.savefig(PLOT_DIR / "wine_kmeans_k_selection.png")
plt.show()
print(f"Saved: {PLOT_DIR / 'wine_kmeans_k_selection.png'}")
best_k = list(k_range)[int(np.argmax(silhouettes))]
print(f"Best k by silhouette: {best_k}  |  Silhouette at k=3: {silhouettes[1]:.3f}")


# %%
""" [4] Fit final k-means (k=3) and assign clusters
k=3 matches both the silhouette peak and the known cultivar count -- in a real segmentation task
the ground truth is absent, so the silhouette result alone would justify this choice.
n_init=10 restarts with different initialisations and keeps the best result by inertia.
"""
km = KMeans(n_clusters=3, n_init=10, random_state=RANDOM_STATE)
train_labels = km.fit_predict(X_train_sc)
test_labels = km.predict(X_test_sc)

print(f"Train cluster sizes: {dict(zip(*np.unique(train_labels, return_counts=True)))}")
print(f"Test cluster sizes:  {dict(zip(*np.unique(test_labels, return_counts=True)))}")


# %%
""" [5] Evaluate: ARI and confusion matrix
Adjusted Rand Index (ARI) measures label agreement corrected for chance: 0 is random, 1 is perfect.
The confusion matrix shows how cultivar classes map to cluster indices; indices are arbitrary
so any one-to-one alignment between rows and columns constitutes a clean recovery.
"""
ari_train = adjusted_rand_score(y_train, train_labels)
ari_test = adjusted_rand_score(y_test, test_labels)
cm = confusion_matrix(y_test, test_labels)

fig, ax = plt.subplots(figsize=(5, 4))
im = ax.imshow(cm, cmap="Blues")
plt.colorbar(im, ax=ax)
ax.set_xticks(range(3))
ax.set_xticklabels([f"cluster_{i}" for i in range(3)])
ax.set_yticks(range(3))
ax.set_yticklabels(CLASS_NAMES)
for i in range(3):
    for j in range(3):
        ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=11)
ax.set_xlabel("Predicted cluster")
ax.set_ylabel("True cultivar")
ax.set_title(f"Cluster vs cultivar (test set)  --  ARI = {ari_test:.3f}")
fig.tight_layout()
fig.savefig(PLOT_DIR / "wine_kmeans_confusion.png")
plt.show()
print(f"Saved: {PLOT_DIR / 'wine_kmeans_confusion.png'}")
print(f"ARI -- train: {ari_train:.3f}  |  test: {ari_test:.3f}  (random baseline = 0.00)")


# %%
""" [6] Cluster centroid profiles
Centroids are in standardised units; each bar shows standard deviations above or below the
overall mean for that feature. Features near zero are not distinctive for that segment;
large absolute values identify the chemical signature separating the three market groups.
"""
centroid_df = pd.DataFrame(km.cluster_centers_, columns=FEATURE_COLS)

fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
colors = ["steelblue", "tomato", "seagreen"]
for i, ax in enumerate(axes):
    n = int((train_labels == i).sum())
    ax.bar(FEATURE_COLS, centroid_df.iloc[i], color=colors[i])
    ax.axhline(0, color="black", linewidth=0.7)
    ax.set_ylabel("Std devs from mean")
    ax.set_title(f"Cluster {i} centroid  (n={n} train samples)")
    ax.tick_params(axis="x", rotation=45)
fig.suptitle("Cluster centroid profiles (standardised features)")
fig.tight_layout()
fig.savefig(PLOT_DIR / "wine_kmeans_centroids.png")
plt.show()
print(f"Saved: {PLOT_DIR / 'wine_kmeans_centroids.png'}")
top_feat = centroid_df.abs().max(axis=0).sort_values(ascending=False)
print(f"Most distinctive features by max centroid deviation:\n{top_feat.head(4).round(2).to_string()}")


# %%
""" [7] Scaling ablation: k-means without StandardScaler
Re-running k-means on raw features shows the effect of scale dominance.
Proline (~700 units) and magnesium (~100 units) overwhelm all other features in Euclidean
distance, so the clustering degrades toward arbitrary splits rather than chemical groupings.
"""
km_raw = KMeans(n_clusters=3, n_init=10, random_state=RANDOM_STATE)
train_labels_raw = km_raw.fit_predict(X_train)
test_labels_raw = km_raw.predict(X_test)

ari_raw_train = adjusted_rand_score(y_train, train_labels_raw)
ari_raw_test = adjusted_rand_score(y_test, test_labels_raw)

print(f"ARI with scaling    -- train: {ari_train:.3f}  |  test: {ari_test:.3f}")
print(f"ARI without scaling -- train: {ari_raw_train:.3f}  |  test: {ari_raw_test:.3f}")
