"""Research question: Can we predict diamond price from cut, color, clarity, and physical dimensions?"""

# %% Imports & config
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path

PLOT_DIR = Path(__file__).parent / "plots"
PLOT_DIR.mkdir(exist_ok=True)

NUMERIC_COLS = ["carat", "depth", "table", "x", "y", "z"]
CAT_COLS     = ["cut", "color", "clarity"]

# %% Load & inspect
"""
53,940 rows — one per diamond.
Numeric: carat, depth (%), table (%), x/y/z (mm dimensions).
Categorical: cut (5 grades), color (7 grades D–J), clarity (8 grades).
Target: price in USD.
A small number of rows have x=y=z=0 (data entry errors) — we drop them.
"""
df = sns.load_dataset("diamonds")
print(df.shape)
print(df.dtypes)
print(df[NUMERIC_COLS + ["price"]].describe().round(2))
print("\nZero-dimension rows:", (df[["x", "y", "z"]] == 0).any(axis=1).sum())

df = df[(df[["x", "y", "z"]] > 0).all(axis=1)].copy()
print(f"After cleanup: {len(df):,} rows")

# %% EDA — price distribution
"""
Price is strongly right-skewed; a log transform yields a near-normal shape.
We model the raw price to keep predictions in USD, but show both scales.
"""
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
axes[0].hist(df["price"], bins=60, edgecolor="none")
axes[0].set_xlabel("Price (USD)")
axes[0].set_title("Price — raw")

axes[1].hist(np.log1p(df["price"]), bins=60, color="tab:orange", edgecolor="none")
axes[1].set_xlabel("log(1 + price)")
axes[1].set_title("Price — log scale")

fig.tight_layout()
fig.savefig(PLOT_DIR / "diamonds_price_dist.png")
plt.show()

# %% EDA — correlation heatmap
"""
carat, x, y, z are near-perfectly collinear (all measure stone size).
This multicollinearity inflates OLS variance — the core motivation for Ridge.
"""
corr = df[NUMERIC_COLS + ["price"]].corr()

fig, ax = plt.subplots(figsize=(7, 6))
im = ax.imshow(corr, vmin=-1, vmax=1, cmap="RdBu_r")
plt.colorbar(im, ax=ax)
ticks = np.arange(len(corr.columns))
ax.set_xticks(ticks); ax.set_xticklabels(corr.columns, rotation=45, ha="right")
ax.set_yticks(ticks); ax.set_yticklabels(corr.columns)
for i in range(len(corr)):
    for j in range(len(corr)):
        ax.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center", fontsize=8)
ax.set_title("Correlation matrix — numerics + price")
fig.tight_layout()
fig.savefig(PLOT_DIR / "diamonds_corr_heatmap.png")
plt.show()

# %% EDA — price by categorical features
"""
Counterintuitive pattern: Fair-cut diamonds often have higher median price than
Ideal-cut diamonds. Reason: carat and cut are correlated — larger stones are more
likely to receive lower-grade cuts. The model must learn this joint structure.
"""
fig, axes = plt.subplots(1, 3, figsize=(13, 4))
for ax, col in zip(axes, CAT_COLS):
    order   = df.groupby(col)["price"].median().sort_values().index
    medians = df.groupby(col)["price"].median().loc[order]
    ax.bar(range(len(order)), medians.values)
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels(order, rotation=40, ha="right")
    ax.set_ylabel("Median price (USD)")
    ax.set_title(f"Price by {col}")
fig.tight_layout()
fig.savefig(PLOT_DIR / "diamonds_price_by_cat.png")
plt.show()

# %% EDA — carat vs price
fig, ax = plt.subplots(figsize=(7, 5))
ax.scatter(df["carat"], df["price"], alpha=0.05, s=5)
ax.set_xlabel("Carat")
ax.set_ylabel("Price (USD)")
ax.set_title("Price vs. carat — non-linear relationship visible")
fig.tight_layout()
fig.savefig(PLOT_DIR / "diamonds_price_vs_carat.png")
plt.show()
