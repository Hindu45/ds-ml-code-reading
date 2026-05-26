"""EDA for the sklearn wine dataset: which physicochemical features drive cultivar separation, and do outliers cluster to specific samples?"""

# %%
""" [1] Imports & config"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.datasets import load_wine

PLOT_DIR = Path(__file__).parent / "plots"
PLOT_DIR.mkdir(exist_ok=True)


# %%
""" [2] Load & general assessment
Load the bundled sklearn wine dataset as a DataFrame.
178 samples, 13 physicochemical features, target encodes 3 cultivar classes (mildly imbalanced).
No missing values; no duplicate rows.
"""
bunch = load_wine(as_frame=True)
df = bunch.frame.copy()
FEATURE_COLS = list(bunch.feature_names)
CLASS_NAMES = list(bunch.target_names)
TARGET_COL = "target"

print(f"Shape: {df.shape}  |  Missing: {df.isnull().sum().sum()}  |  Duplicates: {df.duplicated().sum()}")
print(f"Class counts: {df[TARGET_COL].value_counts().sort_index().to_dict()}")


# %%
""" [3] Feature distributions
Histograms for all 13 features reveal shape and spread before class-level breakdown.
Features span very different scales: proline and magnesium are 10-100x larger than hue or ash.
"""
fig, axes = plt.subplots(3, 5, figsize=(16, 9))
for ax, col in zip(axes.flat, FEATURE_COLS):
    ax.hist(df[col], bins=25, color="steelblue", edgecolor="none")
    ax.set_title(col, fontsize=8)
    ax.tick_params(labelsize=7)
for ax in axes.flat[len(FEATURE_COLS):]:
    ax.set_visible(False)
fig.suptitle("Feature distributions (all 13 features)")
fig.tight_layout()
fig.savefig(PLOT_DIR / "wine_feature_dists.png")
plt.show()
print(f"Saved: {PLOT_DIR / 'wine_feature_dists.png'}")
print(f"Largest std: {df[FEATURE_COLS].std().idxmax()}  |  Smallest std: {df[FEATURE_COLS].std().idxmin()}")


# %%
""" [4] Feature distributions by class
Boxplots grouped by cultivar class show which features separate the three classes cleanly.
Features where boxes do not overlap across classes are the strongest classification signals.
The scale difference is visible here too: proline and magnesium axes are orders of magnitude above hue or ash.
"""
fig, axes = plt.subplots(3, 5, figsize=(16, 9))
for ax, col in zip(axes.flat, FEATURE_COLS):
    groups = [df.loc[df[TARGET_COL] == i, col].values for i in range(len(CLASS_NAMES))]
    ax.boxplot(groups, tick_labels=CLASS_NAMES, widths=0.5)
    ax.set_title(col, fontsize=8)
    ax.tick_params(axis="x", labelsize=7)
for ax in axes.flat[len(FEATURE_COLS):]:
    ax.set_visible(False)
fig.suptitle("Feature distributions by cultivar class")
fig.tight_layout()
fig.savefig(PLOT_DIR / "wine_feature_dists_by_class.png")
plt.show()
print(f"Saved: {PLOT_DIR / 'wine_feature_dists_by_class.png'}")


# %%
""" [5] Coefficient of variation by class
CV (std / |mean|) measures relative variability within each class.
The spread of CV across classes (max CV minus min CV) indicates which features vary most differently between cultivars.
A high CV spread means the feature is not just discriminating on average, but also on variability.
"""
cv_by_class = (
    df.groupby(TARGET_COL)[FEATURE_COLS].std()
    / df.groupby(TARGET_COL)[FEATURE_COLS].mean().abs()
)
cv_spread = (cv_by_class.max() - cv_by_class.min()).sort_values(ascending=False)

fig, ax = plt.subplots(figsize=(9, 4))
ax.bar(cv_spread.index, cv_spread.values, color="steelblue")
ax.set_ylabel("CV spread across classes (max CV - min CV)")
ax.set_title("Coefficient of variation spread by class")
ax.tick_params(axis="x", rotation=45)
fig.tight_layout()
fig.savefig(PLOT_DIR / "wine_cv_spread.png")
plt.show()
print(f"Saved: {PLOT_DIR / 'wine_cv_spread.png'}")
print(f"Top 3 features by CV spread:\n{cv_spread.head(3).round(3).to_string()}")


# %%
""" [6] Correlation matrix
Full 13x13 Pearson correlation heatmap reveals collinear feature groups.
total_phenols, flavanoids, and od280/od315 form a tightly correlated phenol cluster (r > 0.7).
These collinearities mean feature importances from tree models will differ from LDA loadings.
"""
corr = df[FEATURE_COLS].corr()
n = len(FEATURE_COLS)

fig, ax = plt.subplots(figsize=(11, 9))
im = ax.imshow(corr, vmin=-1, vmax=1, cmap="RdBu_r")
plt.colorbar(im, ax=ax)
ax.set_xticks(range(n))
ax.set_xticklabels(FEATURE_COLS, rotation=45, ha="right", fontsize=8)
ax.set_yticks(range(n))
ax.set_yticklabels(FEATURE_COLS, fontsize=8)
for i in range(n):
    for j in range(n):
        ax.text(j, i, f"{corr.iloc[i, j]:.2f}",
                ha="center", va="center", fontsize=6)
ax.set_title("Feature correlation matrix")
fig.tight_layout()
fig.savefig(PLOT_DIR / "wine_correlation.png")
plt.show()
print(f"Saved: {PLOT_DIR / 'wine_correlation.png'}")
phenol_cols = ["total_phenols", "flavanoids", "od280/od315_of_diluted_wines"]
phenol_corr = corr.loc[phenol_cols, phenol_cols].round(2)
print(f"Phenol group (total_phenols, flavanoids, od280):\n{phenol_corr.to_string()}")


# %%
""" [7] Outlier investigation
Z-scores flag samples with |z| > 3 per feature; 8 features contain at least one such sample.
Checking whether malic_acid and color_intensity flag the same rows tests if outliers are systemic (one unusual wine)
or feature-specific (measurement anomalies in independent assays).
"""
z = (df[FEATURE_COLS] - df[FEATURE_COLS].mean()) / df[FEATURE_COLS].std()
flagged_counts = (z.abs() > 3).sum().sort_values(ascending=False)
flagged_malic = set(df.index[z["malic_acid"].abs() > 3])
flagged_color = set(df.index[z["color_intensity"].abs() > 3])
overlap = flagged_malic & flagged_color

fig, ax = plt.subplots(figsize=(9, 4))
ax.bar(flagged_counts.index, flagged_counts.values, color="steelblue")
ax.set_ylabel("Count of |z| > 3 samples")
ax.set_title("Outlier flags per feature (|z| > 3)")
ax.tick_params(axis="x", rotation=45)
fig.tight_layout()
fig.savefig(PLOT_DIR / "wine_outliers.png")
plt.show()
print(f"Saved: {PLOT_DIR / 'wine_outliers.png'}")
print(f"Features with outliers: {flagged_counts[flagged_counts > 0].to_dict()}")
print(f"malic_acid outlier rows: {sorted(flagged_malic)}  |  color_intensity: {sorted(flagged_color)}")
print(f"Overlap: {sorted(overlap) if overlap else 'none -- outliers are feature-specific'}")
