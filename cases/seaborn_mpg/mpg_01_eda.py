"""Research question: Can we predict a car's fuel efficiency (mpg) from engine and body specs?"""

# %%
""" [1] Imports & config"""
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path

PLOT_DIR = Path(__file__).parent / "plots"
PLOT_DIR.mkdir(exist_ok=True)

NUMERIC_COLS = ["cylinders", "displacement", "horsepower", "weight", "acceleration", "model_year"]

# %%
""" [2] Load & inspect
398 rows, 9 columns. One row = one car model/trim.
Numeric predictors: cylinders, displacement (cu. in.), horsepower, weight (lbs), acceleration (0-60 mph time in sec), model_year.
Categorical: origin, name (car model string — dropped later).
Target: mpg (miles per gallon).
6 rows have horsepower missing — the only missingness in the dataset.
"""
df_raw = sns.load_dataset("mpg")
print(df_raw.shape)
print(f"numeric: {df_raw.select_dtypes('number').columns.tolist()}")
print(f"categorical: {df_raw.select_dtypes('object').columns.tolist()}")
print(f"origin categories: {df_raw['origin'].unique().tolist()}")
print(df_raw.isnull().sum()[df_raw.isnull().sum() > 0])

# %%
""" [3] Missing horsepower
The 6 missing rows include 4 US cars and 2 European (Renault), spanning model years
71–82 — scattered across the dataset, not clustered at the start.
The reason for missingness is unclear; the mix of origins and years makes a single
MNAR explanation unlikely.
We drop them (1.5% loss) rather than impute, to keep the analysis transparent.
"""
missing_hp = df_raw[df_raw["horsepower"].isnull()]
print(missing_hp[["name", "origin", "model_year", "cylinders"]])

df = df_raw.dropna(subset=["horsepower"]).copy()
print(f"\nAfter dropping missing horsepower: {len(df):,} rows")

# %%
""" [4] Target distribution — mpg
mpg is moderately right-skewed (range 9–46.6). The fuel consumption inverse
(gallons-per-mile = 1/mpg) is closer to symmetric and physically more natural,
but mpg is the industry-standard reported quantity — we model it directly.
"""
fig, axes = plt.subplots(1, 2, figsize=(10, 4))

axes[0].hist(df["mpg"], bins=30, edgecolor="none")
axes[0].set_xlabel("mpg")
axes[0].set_title("mpg: raw")

axes[1].hist(1 / df["mpg"], bins=30, color="tab:orange", edgecolor="none")
axes[1].set_xlabel("1 / mpg  (gallons per mile)")
axes[1].set_title("Fuel consumption: physical inverse")

fig.tight_layout()
fig.savefig(PLOT_DIR / "mpg_target_dist.png")
plt.show()
print(f"Saved: {PLOT_DIR / 'mpg_target_dist.png'}")

# %%
""" [5] mpg over time — the oil crisis signal
Model year runs 1970–1982. Fleet average mpg rises sharply after the 1973 oil
embargo and again after 1979. This time trend is a strong predictor even after
controlling for engine specs — it captures regulatory and design-era effects.
"""
by_year = df.groupby("model_year")["mpg"].agg(["mean", "median", "std"]).reset_index()

fig, ax = plt.subplots(figsize=(9, 4))
ax.fill_between(by_year["model_year"],
                by_year["mean"] - by_year["std"],
                by_year["mean"] + by_year["std"],
                alpha=0.2, label="±1 std")
ax.plot(by_year["model_year"], by_year["mean"],   marker="o", label="mean")
ax.plot(by_year["model_year"], by_year["median"], marker="s", linestyle="--", label="median")
ax.axvline(73, color="red",  linestyle=":", linewidth=1.2, label="1973 oil embargo")
ax.axvline(79, color="darkred", linestyle=":", linewidth=1.2, label="1979 crisis")
ax.set_xlabel("Model year (70 = 1970)")
ax.set_ylabel("mpg")
ax.set_title("Fleet fuel efficiency over time")
ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig(PLOT_DIR / "mpg_by_year.png")
plt.show()
print(f"mean mpg: {by_year['mean'].iloc[0]:.1f} (yr {by_year['model_year'].iloc[0]}) → {by_year['mean'].iloc[-1]:.1f} (yr {by_year['model_year'].iloc[-1]})")
print(f"Saved: {PLOT_DIR / 'mpg_by_year.png'}")

# %%
""" [6] mpg by origin
Japanese and European cars are concentrated in the 4-cylinder economy segment,
yielding higher median mpg. US cars dominate the 6- and 8-cylinder range.
Origin is partly a proxy for cylinders/displacement: they're correlated predictors.
"""
fig, axes = plt.subplots(1, 2, figsize=(10, 4))

origin_order = df.groupby("origin")["mpg"].median().sort_values(ascending=False).index
counts = df["origin"].value_counts()

axes[0].bar(origin_order, [counts[o] for o in origin_order])
axes[0].set_ylabel("Number of cars")
axes[0].set_title("Car count by origin")

for origin, grp in df.groupby("origin"):
    axes[1].hist(grp["mpg"], bins=20, alpha=0.5, label=origin)
axes[1].set_xlabel("mpg")
axes[1].set_title("mpg distribution by origin")
axes[1].legend()

fig.tight_layout()
fig.savefig(PLOT_DIR / "mpg_by_origin.png")
plt.show()
print(df.groupby("origin")["mpg"].median().sort_values(ascending=False).round(1).to_string(header=False))
print(f"Saved: {PLOT_DIR / 'mpg_by_origin.png'}")

# %%
""" [7] mpg by cylinders
Cylinders takes only 5 discrete values (3, 4, 5, 6, 8) — it behaves more like
an ordinal category than a continuous feature. The 3- and 5-cylinder classes
are rare (4 and 3 cars respectively). The 4 vs 8 gap is the dominant signal.
"""
cyl_counts = df["cylinders"].value_counts().sort_index()

fig, axes = plt.subplots(1, 2, figsize=(10, 4))

axes[0].bar(cyl_counts.index.astype(str), cyl_counts.values)
axes[0].set_xlabel("Cylinders")
axes[0].set_ylabel("Count")
axes[0].set_title("Cars per cylinder class")

for cyl, grp in df.groupby("cylinders"):
    axes[1].hist(grp["mpg"], bins=15, alpha=0.5, label=str(cyl))
axes[1].set_xlabel("mpg")
axes[1].set_title("mpg by cylinder count")
axes[1].legend(title="cylinders", fontsize=8)

fig.tight_layout()
fig.savefig(PLOT_DIR / "mpg_by_cylinders.png")
plt.show()
print(cyl_counts.to_string())
print(f"Saved: {PLOT_DIR / 'mpg_by_cylinders.png'}")

# %%
""" [8] Predictor correlations with mpg
Weight (-0.83) and displacement (-0.81) are the strongest negative predictors;
horsepower and cylinders follow, both at -0.78.
Model year is a strong positive predictor — the time trend visible above.
Acceleration correlates positively because lighter/smaller-engined cars both
accelerate slowly (high 0-60 time) and get better mpg.
"""
corr_with_mpg = df[NUMERIC_COLS + ["mpg"]].corr()["mpg"].drop("mpg").sort_values()

fig, ax = plt.subplots(figsize=(6, 4))
colors = ["tab:red" if v < 0 else "tab:blue" for v in corr_with_mpg]
ax.barh(corr_with_mpg.index, corr_with_mpg.values, color=colors)
ax.axvline(0, color="black", linewidth=0.8)
ax.set_xlabel("Pearson r with mpg")
ax.set_title("Feature correlations with mpg")
fig.tight_layout()
fig.savefig(PLOT_DIR / "mpg_feature_corr.png")
plt.show()
print(corr_with_mpg.head(3).round(3).to_string())
print(corr_with_mpg.tail(3).round(3).to_string())
print(f"Saved: {PLOT_DIR / 'mpg_feature_corr.png'}")

# %%
""" [9] Multicollinearity among predictors
Cylinders, displacement, and weight are highly intercorrelated (r > 0.9).
Including all three gives the model redundant information and inflates OLS
coefficient variance — the primary motivation for Ridge regularisation.
"""
corr_matrix = df[NUMERIC_COLS].corr()

fig, ax = plt.subplots(figsize=(7, 6))
im = ax.imshow(corr_matrix, vmin=-1, vmax=1, cmap="RdBu_r")
plt.colorbar(im, ax=ax)
ticks = np.arange(len(NUMERIC_COLS))
ax.set_xticks(ticks); ax.set_xticklabels(NUMERIC_COLS, rotation=45, ha="right")
ax.set_yticks(ticks); ax.set_yticklabels(NUMERIC_COLS)
for i in range(len(NUMERIC_COLS)):
    for j in range(len(NUMERIC_COLS)):
        ax.text(j, i, f"{corr_matrix.iloc[i, j]:.2f}",
                ha="center", va="center", fontsize=8)
ax.set_title("Predictor correlation matrix — multicollinearity check")
fig.tight_layout()
fig.savefig(PLOT_DIR / "mpg_predictor_corr.png")
plt.show()
print(f"Saved: {PLOT_DIR / 'mpg_predictor_corr.png'}")

# %%
""" [10] mpg vs weight and displacement — non-linearity
The relationship between mpg and heavy predictors (weight, displacement) is
curved: mpg drops steeply for lighter cars but flattens for heavy ones.
This is physically expected — fuel consumption (1/mpg) scales more linearly
with weight. Linear regression will underfit this curvature.
"""
fig, axes = plt.subplots(1, 2, figsize=(11, 4))

for ax, col in zip(axes, ["weight", "displacement"]):
    sc = ax.scatter(df[col], df["mpg"], c=df["model_year"],
                    cmap="viridis", alpha=0.5, s=15)
    plt.colorbar(sc, ax=ax, label="model_year")
    ax.set_xlabel(col)
    ax.set_ylabel("mpg")
    ax.set_title(f"mpg vs {col} (colour = year)")

fig.tight_layout()
fig.savefig(PLOT_DIR / "mpg_vs_weight_displacement.png")
plt.show()
print(f"Saved: {PLOT_DIR / 'mpg_vs_weight_displacement.png'}")

# %%
""" [11] Horsepower outliers
Horsepower has a long right tail (max 230 hp vs mean ~104). These are high-
displacement American muscle cars from the early 1970s. They also have the
lowest mpg values in the dataset — horsepower and efficiency trade-off directly.
"""
hp_z = (df["horsepower"] - df["horsepower"].mean()) / df["horsepower"].std()
outliers = df[hp_z.abs() > 2.5][["name", "origin", "model_year", "cylinders", "horsepower", "mpg"]]

fig, ax = plt.subplots(figsize=(6, 4))
ax.scatter(df["horsepower"], df["mpg"], alpha=0.4, s=15, label="normal")
ax.scatter(outliers["horsepower"], outliers["mpg"], color="red", s=40,
           zorder=5, label=f"outliers (|z|>2.5)")
ax.set_xlabel("Horsepower")
ax.set_ylabel("mpg")
ax.set_title("mpg vs horsepower — outlier cars")
ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig(PLOT_DIR / "mpg_horsepower_outliers.png")
plt.show()
print(f"High-horsepower outliers (|z| > 2.5): {len(outliers)}")
print(outliers.sort_values("horsepower", ascending=False).head(4).to_string())
print(f"Saved: {PLOT_DIR / 'mpg_horsepower_outliers.png'}")

# %%
""" [12] Acceleration — the confounded predictor
Acceleration (0–60 mph time in seconds) is positively correlated with mpg,
but this is a suppression effect: slow acceleration = small/light engine = good mpg.
It is NOT that driving slowly improves fuel economy. High-acceleration outliers
(short 0-60 times, low mpg) are the muscle cars from the horsepower analysis.
"""
fig, ax = plt.subplots(figsize=(6, 4))
sc = ax.scatter(df["acceleration"], df["mpg"],
                c=df["horsepower"], cmap="plasma", alpha=0.5, s=15)
plt.colorbar(sc, ax=ax, label="horsepower")
ax.set_xlabel("Acceleration (seconds, 0–60 mph)")
ax.set_ylabel("mpg")
ax.set_title("mpg vs acceleration — horsepower as confound")
fig.tight_layout()
fig.savefig(PLOT_DIR / "mpg_acceleration.png")
plt.show()
print(f"Saved: {PLOT_DIR / 'mpg_acceleration.png'}")
