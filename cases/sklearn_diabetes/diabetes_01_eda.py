"""EDA for the sklearn diabetes dataset: which physiological features predict one-year disease progression?"""

# %%
""" [1] Imports & config"""
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.datasets import load_diabetes

PLOT_DIR = Path(__file__).parent / "plots"
PLOT_DIR.mkdir(exist_ok=True)

SERUM_COLS = ["s1", "s2", "s3", "s4", "s5", "s6"]
FEATURE_COLS = ["age", "sex", "bmi", "bp"] + SERUM_COLS


# %%
""" [2] Load & general assessment
Load the bundled sklearn diabetes dataset as a DataFrame.
All 10 features are pre-standardized to mean ~0, std ~0.048.
The target (disease progression score) is NOT standardized -- range 25 to 346.
"""
bunch = load_diabetes(as_frame=True)
df = bunch.frame.copy()

print(f"Shape: {df.shape}  |  Missing: {df.isnull().sum().sum()}  |  Duplicates: {df.duplicated().sum()}")
print(f"Target range: {df['target'].min():.0f}–{df['target'].max():.0f}  |  Mean: {df['target'].mean():.1f}")


# %%
""" [3] Target distribution
The target is a continuous disease-progression score measured one year after baseline.
It is the only column that is NOT z-score standardized.
Mild right skew; a few patients show very high progression (>300).
"""
fig, ax = plt.subplots(figsize=(7, 4))
ax.hist(df["target"], bins=30, color="steelblue", edgecolor="none")
ax.set_xlabel("Disease progression score")
ax.set_ylabel("Count")
ax.set_title("Target distribution -- disease progression one year post-baseline")
fig.tight_layout()
fig.savefig(PLOT_DIR / "diabetes_target_dist.png")
plt.show()
print(f"Saved: {PLOT_DIR / 'diabetes_target_dist.png'}")


# %%
""" [4] Feature distributions
All 10 input features are z-score standardized (mean ~0, std ~0.048).
'sex' has only two distinct float values (-0.045 and 0.051) -- binary encoded as float, not 0/1.
Features bmi, s1, s2, s3, s4 show mild outliers; the others are close to normal.
"""
fig, axes = plt.subplots(2, 5, figsize=(15, 6))
for ax, col in zip(axes.flat, FEATURE_COLS):
    ax.hist(df[col], bins=25, color="steelblue", edgecolor="none")
    ax.set_title(col)
    ax.set_xlabel("z-score")
fig.suptitle("Feature distributions (all z-score standardized)")
fig.tight_layout()
fig.savefig(PLOT_DIR / "diabetes_feature_dists.png")
plt.show()
print(f"Saved: {PLOT_DIR / 'diabetes_feature_dists.png'}")


# %%
""" [5] Feature-target correlations
Pearson r between each feature and the progression target.
bmi and s5 (log serum triglycerides) are the strongest positive predictors.
s3 (HDL cholesterol) shows the strongest negative correlation -- higher HDL is associated with lower progression.
"""
pearson_r = df[FEATURE_COLS].corrwith(df["target"]).sort_values()

bar_colors = ["tomato" if v > 0 else "steelblue" for v in pearson_r]
fig, ax = plt.subplots(figsize=(7, 4))
ax.barh(pearson_r.index, pearson_r.values, color=bar_colors)
ax.axvline(0, color="black", linewidth=0.8)
ax.set_xlabel("Pearson r with target")
ax.set_title("Feature-target correlations")
fig.tight_layout()
fig.savefig(PLOT_DIR / "diabetes_feature_target_corr.png")
plt.show()
print(f"Saved: {PLOT_DIR / 'diabetes_feature_target_corr.png'}")
print(f"Top 3 positive r:\n{pearson_r.tail(3).round(3).to_string()}")
print(f"Top 3 negative r:\n{pearson_r.head(3).round(3).to_string()}")


# %%
""" [6] BMI vs target -- linearity check
BMI is the single strongest positive predictor (r ~0.59).
The scatter with an OLS overlay checks whether the relationship is approximately linear.
Increasing variance at higher BMI values (heteroscedasticity) is worth noting for OLS diagnostics.
"""
m, b = np.polyfit(df["bmi"], df["target"], 1)
x_line = np.linspace(df["bmi"].min(), df["bmi"].max(), 200)

fig, ax = plt.subplots(figsize=(6, 5))
ax.scatter(df["bmi"], df["target"], alpha=0.35, s=18, color="steelblue")
ax.plot(x_line, m * x_line + b, color="tomato", linewidth=1.5, label="OLS fit")
ax.set_xlabel("BMI (z-score)")
ax.set_ylabel("Disease progression score")
ax.set_title("BMI vs progression -- linearity check")
ax.legend()
fig.tight_layout()
fig.savefig(PLOT_DIR / "diabetes_bmi_scatter.png")
plt.show()
print(f"Saved: {PLOT_DIR / 'diabetes_bmi_scatter.png'}")
print(f"OLS slope (bmi → target): {m:.3f}")


# %%
""" [7] Sex group comparison
'sex' takes two float values: -0.045 and 0.051, encoding a binary variable.
Box plots compare progression distributions between groups; the scatter overlay
shows whether any gap persists after accounting for BMI (the strongest confounder).
"""
sex_vals = sorted(df["sex"].unique())
groups = [df.loc[df["sex"] == v, "target"].values for v in sex_vals]
labels = [f"sex={v:.3f}" for v in sex_vals]

fig, axes = plt.subplots(1, 2, figsize=(11, 4))

axes[0].boxplot(groups, labels=labels)
axes[0].set_ylabel("Disease progression score")
axes[0].set_title("Progression by sex group")

for v, color in zip(sex_vals, ["steelblue", "tomato"]):
    mask = df["sex"] == v
    axes[1].scatter(df.loc[mask, "bmi"], df.loc[mask, "target"],
                    alpha=0.35, s=15, color=color, label=f"sex={v:.3f}")
axes[1].set_xlabel("BMI (z-score)")
axes[1].set_ylabel("Disease progression score")
axes[1].set_title("BMI vs progression, colored by sex")
axes[1].legend(fontsize=8)

fig.tight_layout()
fig.savefig(PLOT_DIR / "diabetes_sex_comparison.png")
plt.show()
print(f"Saved: {PLOT_DIR / 'diabetes_sex_comparison.png'}")
mean_by_sex = df.groupby("sex")["target"].mean().round(1)
print(f"Mean progression by sex group:\n{mean_by_sex.to_string()}")


# %%
""" [8] Serum feature correlation matrix
The six serum features (s1-s6) are biochemically related: s1 (total cholesterol)
and s2 (LDL) are strongly correlated since LDL is a component of total cholesterol.
s3 (HDL) is negatively correlated with s4 (TC/HDL ratio) by construction.
These collinearities motivate regularization (Ridge, Lasso) over plain OLS.
"""
corr_cols = SERUM_COLS + ["bmi", "bp", "target"]
corr = df[corr_cols].corr()
n = len(corr_cols)

fig, ax = plt.subplots(figsize=(8, 7))
im = ax.imshow(corr, vmin=-1, vmax=1, cmap="RdBu_r")
plt.colorbar(im, ax=ax)
ax.set_xticks(range(n))
ax.set_xticklabels(corr_cols, rotation=45, ha="right")
ax.set_yticks(range(n))
ax.set_yticklabels(corr_cols)
for i in range(n):
    for j in range(n):
        ax.text(j, i, f"{corr.iloc[i, j]:.2f}",
                ha="center", va="center", fontsize=7)
ax.set_title("Correlation matrix -- serum features, BMI, BP, and target")
fig.tight_layout()
fig.savefig(PLOT_DIR / "diabetes_correlation.png")
plt.show()
print(f"Saved: {PLOT_DIR / 'diabetes_correlation.png'}")
