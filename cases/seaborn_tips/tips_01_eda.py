"""EDA for the tips dataset: what drives tip percentage at a restaurant?

Questions (increasing complexity):
  1. What is the distribution of tip percentage, and what is a "typical" tip rate?
  2. Do tip rates differ by sex, smoker status, day of week, or time of day?
  3. After controlling for party size, is the effect of sex on tip percentage
     still present, or does it disappear?
"""

# %%
""" [1] Setup
Load the tips dataset and engineer the target variable: tip_pct = tip / total_bill * 100.
Print a quick sanity check (shape, dtypes, missing values) and flag the one known
duplicate row before any analysis.
"""
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path

PLOT_DIR = Path(__file__).parent / "plots"
PLOT_DIR.mkdir(exist_ok=True)

tips = sns.load_dataset("tips")
tips["tip_pct"] = tips["tip"] / tips["total_bill"] * 100

print(f"Shape: {tips.shape}   Duplicates: {tips.duplicated().sum()}")
print(tips.dtypes)


# %%
""" [2] Tip vs total bill
Before normalising, look at the raw relationship: does tip scale linearly with
the bill?  A regression line through the scatter reveals the marginal tip
rate (slope = Δtip / Δbill; note this is not the average tip rate, which is mean(tip)/mean(bill)).  This motivates why tip_pct is a better response
variable than raw tip — if the slope were perfectly consistent, all points would
lie on one line and there would be nothing to explain.
"""
from scipy import stats

fig, ax = plt.subplots(figsize=(7, 5))
sns.scatterplot(data=tips, x="total_bill", y="tip", alpha=0.5, ax=ax)

slope, intercept, *_ = stats.linregress(tips["total_bill"], tips["tip"])
x_line = np.array([tips["total_bill"].min(), tips["total_bill"].max()])
ax.plot(x_line, intercept + slope * x_line, color="tomato", linewidth=1.8,
        label=f"OLS slope = {slope:.3f}  (≈ {slope*100:.1f}% tip rate)")
ax.set_xlabel("Total bill ($)")
ax.set_ylabel("Tip ($)")
ax.set_title("Tip vs total bill")
ax.legend()

fig.tight_layout()
fig.savefig(PLOT_DIR / "tips_tip_vs_bill.png")
plt.show()
print(f"Saved: {PLOT_DIR / 'tips_tip_vs_bill.png'}")
print(f"OLS: tip = {intercept:.2f} + {slope:.3f} × total_bill")


# %%
""" [3] Q1 — Distribution of tip percentage
Histogram + KDE to judge shape (skew, modality) and a box plot to read off
the median and IQR at a glance.  Vertical lines mark the mean and median so
students can see whether they diverge — a sign of skew.
"""
fig, axes = plt.subplots(1, 2, figsize=(11, 4))

ax = axes[0]
tips["tip_pct"].plot.hist(bins=30, density=True, alpha=0.6, color="steelblue", ax=ax)
tips["tip_pct"].plot.kde(color="steelblue", linewidth=1.8, ax=ax)
ax.axvline(tips["tip_pct"].mean(),   color="tomato",  linestyle="--", label=f"mean  {tips['tip_pct'].mean():.1f}%")
ax.axvline(tips["tip_pct"].median(), color="seagreen", linestyle="--", label=f"median {tips['tip_pct'].median():.1f}%")
ax.set_xlabel("Tip (%)")
ax.set_title("Distribution of tip percentage")
ax.legend()

ax = axes[1]
tips.boxplot(column="tip_pct", ax=ax)
ax.set_ylabel("Tip (%)")
ax.set_title("Tip percentage — box plot")

fig.tight_layout()
fig.savefig(PLOT_DIR / "tips_tip_pct_distribution.png")
plt.show()
print(f"Saved: {PLOT_DIR / 'tips_tip_pct_distribution.png'}")
print(f"tip_pct: mean={tips['tip_pct'].mean():.1f}%  median={tips['tip_pct'].median():.1f}%  std={tips['tip_pct'].std():.1f}%")


# %%
""" [4] Q2 — Tip rate by categorical variables
Four side-by-side box plots — one per categorical predictor (sex, smoker, day, time).
Box plots reveal median differences and spread without hiding the shape.
Overlaid strip plots (jitter) let students see the actual data points, which is
important given the small n (244 rows).
"""
cat_cols = ["sex", "smoker", "day", "time"]
fig, axes = plt.subplots(1, 4, figsize=(14, 5), sharey=True)

for ax, col in zip(axes, cat_cols):
    sns.boxplot(data=tips, x=col, y="tip_pct", hue=col, ax=ax, width=0.45,
                palette="pastel", legend=False, flierprops=dict(marker="x", markersize=4))
    sns.stripplot(data=tips, x=col, y="tip_pct", ax=ax,
                  color="steelblue", alpha=0.35, size=3, jitter=True)
    ax.set_xlabel(col)
    ax.set_ylabel("Tip (%)" if col == "sex" else "")
    ax.set_title(f"Tip % by {col}")

fig.suptitle("Tip percentage by categorical predictors", y=1.02)
fig.tight_layout()
fig.savefig(PLOT_DIR / "tips_tip_pct_by_category.png")
plt.show()
print(f"Saved: {PLOT_DIR / 'tips_tip_pct_by_category.png'}")
print(tips.groupby("sex")["tip_pct"].describe().round(2))
print(tips.groupby("smoker")["tip_pct"].describe().round(2))
print(tips.groupby("day")["tip_pct"].describe().round(2))

# %%
""" [5] Bill size by day and time
If day or time affects tip_pct (seen in Q2), one explanation is that people
order bigger meals on certain days/at certain times, and bill size itself
drives the tip rate.  Box plots of total_bill by day and time let us check
whether the groups differ in spend — a potential confound to flag.
"""
fig, axes = plt.subplots(1, 2, figsize=(10, 4))

sns.boxplot(data=tips, x="day", y="total_bill", hue="day", ax=axes[0],
            palette="pastel", legend=False, order=["Thur", "Fri", "Sat", "Sun"],
            flierprops=dict(marker="x", markersize=4))
sns.stripplot(data=tips, x="day", y="total_bill", ax=axes[0],
              order=["Thur", "Fri", "Sat", "Sun"],
              color="steelblue", alpha=0.35, size=3, jitter=True)
axes[0].set_title("Total bill by day")
axes[0].set_ylabel("Total bill ($)")

sns.boxplot(data=tips, x="time", y="total_bill", hue="time", ax=axes[1],
            palette="pastel", legend=False,
            flierprops=dict(marker="x", markersize=4))
sns.stripplot(data=tips, x="time", y="total_bill", ax=axes[1],
              color="steelblue", alpha=0.35, size=3, jitter=True)
axes[1].set_title("Total bill by time")
axes[1].set_ylabel("")

fig.suptitle("Is bill size different across days / meal times?")
fig.tight_layout()
fig.savefig(PLOT_DIR / "tips_bill_by_day_time.png")
plt.show()
print(f"Saved: {PLOT_DIR / 'tips_bill_by_day_time.png'}")
print(tips.groupby("day")["total_bill"].median().round(2))
print(tips.groupby("time")["total_bill"].median().round(2))


# %%
""" [6] Smoker × sex interaction
Main effects (Q2) can mask interactions: perhaps female smokers tip very
differently from male smokers even though neither sex nor smoker alone shows a
strong signal.  A 2×2 heatmap of mean tip_pct makes the interaction pattern
immediately readable — any row or column that is NOT monotone is a hint of an
interaction effect.
"""
pivot_interaction = tips.pivot_table(
    values="tip_pct", index="smoker", columns="sex", aggfunc="mean"
)

fig, ax = plt.subplots(figsize=(5, 3))
im = ax.imshow(pivot_interaction.values, cmap="RdYlGn", aspect="auto",
               vmin=tips["tip_pct"].quantile(0.1),
               vmax=tips["tip_pct"].quantile(0.9))
ax.set_xticks([0, 1])
ax.set_xticklabels(pivot_interaction.columns)
ax.set_yticks([0, 1])
ax.set_yticklabels(pivot_interaction.index)
ax.set_xlabel("Sex")
ax.set_ylabel("Smoker")
ax.set_title("Mean tip % — smoker × sex interaction")
for i in range(2):
    for j in range(2):
        ax.text(j, i, f"{pivot_interaction.values[i, j]:.1f}%",
                ha="center", va="center", fontsize=12, fontweight="bold")
fig.colorbar(im, ax=ax, label="Mean tip %")

fig.tight_layout()
fig.savefig(PLOT_DIR / "tips_smoker_sex_interaction.png")
plt.show()
print(f"Saved: {PLOT_DIR / 'tips_smoker_sex_interaction.png'}")
print(pivot_interaction.round(2))


# %%
""" [7] Q3 — Sex effect controlled for party size
Party size (1–6) is a potential confounder: larger parties may tip differently
and may skew toward one sex.  We visualise two things:
  a) tip_pct vs. size, coloured by sex — does one group cluster at certain sizes?
  b) box plots of tip_pct by sex, faceted by size — does the sex gap persist
     within each size level, or does it vanish once size is held constant?
"""
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# a) scatter: size vs tip_pct coloured by sex
ax = axes[0]
for sex, grp in tips.groupby("sex"):
    ax.scatter(grp["size"] + (0.08 if sex == "Male" else -0.08),
               grp["tip_pct"], label=sex, alpha=0.45, s=20)
ax.set_xlabel("Party size")
ax.set_ylabel("Tip (%)")
ax.set_title("Tip % vs party size (by sex)")
ax.legend(title="Sex")

# b) box plots faceted by size — sex on x, size in columns
ax = axes[1]
sizes = sorted(tips["size"].unique())
width = 0.35
x_positions = np.arange(len(sizes))
for i, sex in enumerate(["Male", "Female"]):
    subset = tips[tips["sex"] == sex]
    vals_by_size = [subset.loc[subset["size"] == s, "tip_pct"].values for s in sizes]
    bp = ax.boxplot(vals_by_size,
                    positions=x_positions + (i - 0.5) * width,
                    widths=width * 0.85,
                    patch_artist=True,
                    boxprops=dict(facecolor="steelblue" if sex == "Male" else "salmon", alpha=0.6),
                    medianprops=dict(color="black"),
                    flierprops=dict(marker="x", markersize=3),
                    label=sex)

ax.set_xticks(x_positions)
ax.set_xticklabels(sizes)
ax.set_xlabel("Party size")
ax.set_ylabel("Tip (%)")
ax.set_title("Tip % by sex — controlled for party size")
ax.legend(title="Sex")

fig.tight_layout()
fig.savefig(PLOT_DIR / "tips_sex_effect_by_size.png")
plt.show()
print(f"Saved: {PLOT_DIR / 'tips_sex_effect_by_size.png'}")
pivot = tips.pivot_table(values="tip_pct", index="size", columns="sex", aggfunc="mean")
print(pivot.round(2))
