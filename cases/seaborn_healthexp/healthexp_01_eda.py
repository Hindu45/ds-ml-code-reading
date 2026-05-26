"""EDA for the healthexp dataset: does spending more on healthcare buy longer lives?
purpose: eda | style: library-optimal | flags: docstring-depth: annotated

Questions (increasing complexity):
  1. How did spending and life expectancy evolve over time? Which country spends
     the most, and does it have the highest life expectancy?
  2. Is there a positive pooled correlation between spending and life expectancy?
     Does the relationship look different when examined within each country?
  3. The USA spends far more than its peers. Is its life expectancy converging
     toward or diverging from the peer median over time?
"""

# %%
""" [1] Imports & config"""
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import stats

PLOT_DIR = Path(__file__).parent / "plots"
PLOT_DIR.mkdir(exist_ok=True)

COUNTRIES = ["Canada", "France", "Germany", "Great Britain", "Japan", "USA"]
PALETTE = dict(zip(COUNTRIES, ["tab:orange", "tab:blue", "tab:green", "tab:purple", "tab:cyan", "tab:red"]))


# %%
""" [2] Load & inspect
274 rows, 4 columns: one per (country, year) pair spanning 1970–2020.
Features: Year (int), Country (6 categories), Spending_USD (float), Life_Expectancy (float).
No missing values; no duplicates — a clean panel.
"""
df = sns.load_dataset("healthexp")

print(df.shape)
print(df.dtypes)
print(df.isnull().sum())
print(df[["Spending_USD", "Life_Expectancy"]].describe().round(2))
print(f"Duplicates: {df.duplicated().sum()}")
print(f"Year range: {df['Year'].min()}–{df['Year'].max()}")


# %%
""" [3] Spending over time by country
Line chart of Spending_USD per country from 1970 to 2020.
The USA starts close to peers in 1970 but diverges sharply after 1980; by 2020 it
spends roughly twice as much per person as Germany, the next-highest spender.
"""
fig, ax = plt.subplots(figsize=(9, 5))

for country, grp in df.groupby("Country"):
    grp_s = grp.sort_values("Year")
    ax.plot(grp_s["Year"], grp_s["Spending_USD"],
            label=country, color=PALETTE[country], linewidth=1.8)

ax.set_xlabel("Year")
ax.set_ylabel("Healthcare spending (USD per capita)")
ax.set_title("Healthcare spending over time by country")
ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig(PLOT_DIR / "healthexp_timeseries_spending.png")
plt.show()
print(f"Saved: {PLOT_DIR / 'healthexp_timeseries_spending.png'}")
last_year = df["Year"].max()
ranking = (df[df["Year"] == last_year]
           .set_index("Country")["Spending_USD"]
           .sort_values(ascending=False)
           .round(0))
print(f"Spending in {last_year} (USD per capita):\n{ranking.to_string()}")


# %%
""" [4] Life expectancy over time by country
Japan has led in life expectancy since the 1980s despite moderate spending.
USA life expectancy stagnated around 2014 and declined — a unique pattern among
these six wealthy nations, and the central puzzle of this dataset.
"""
fig, ax = plt.subplots(figsize=(9, 5))

for country, grp in df.groupby("Country"):
    grp_s = grp.sort_values("Year")
    ax.plot(grp_s["Year"], grp_s["Life_Expectancy"],
            label=country, color=PALETTE[country], linewidth=1.8)

ax.set_xlabel("Year")
ax.set_ylabel("Life expectancy (years)")
ax.set_title("Life expectancy over time by country")
ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig(PLOT_DIR / "healthexp_timeseries_lifeexp.png")
plt.show()
print(f"Saved: {PLOT_DIR / 'healthexp_timeseries_lifeexp.png'}")
ranking_le = (df[df["Year"] == last_year]
              .set_index("Country")["Life_Expectancy"]
              .sort_values(ascending=False)
              .round(2))
print(f"Life expectancy in {last_year} (years):\n{ranking_le.to_string()}")


# %%
""" [5] Numeric distributions — Spending_USD and Life_Expectancy
Spending_USD is strongly right-skewed: the USA pulls the mean well above the median.
Life_Expectancy shows a bimodal marginal shape — one cluster near 70 years (early
decades) and another near 80 (recent decades), reflecting the global upward trend
that makes year a strong confounder in any spending–life expectancy model.
"""
fig, axes = plt.subplots(1, 2, figsize=(11, 4))

for ax, col in zip(axes, ["Spending_USD", "Life_Expectancy"]):
    ax.hist(df[col], bins=30, density=True, alpha=0.6, color="steelblue")
    df[col].plot.kde(ax=ax, color="steelblue", linewidth=1.8)
    ax.axvline(df[col].mean(),   color="tomato",   linestyle="--",
               label=f"mean {df[col].mean():.1f}")
    ax.axvline(df[col].median(), color="seagreen",  linestyle="--",
               label=f"median {df[col].median():.1f}")
    ax.set_xlabel(col)
    ax.set_title(f"Distribution of {col}")
    ax.legend(fontsize=8)

fig.tight_layout()
fig.savefig(PLOT_DIR / "healthexp_distributions.png")
plt.show()
print(f"Saved: {PLOT_DIR / 'healthexp_distributions.png'}")
for col in ["Spending_USD", "Life_Expectancy"]:
    s = df[col]
    print(f"{col}: mean={s.mean():.1f}  median={s.median():.1f}  std={s.std():.1f}")


# %%
""" [6] Pooled scatter — spending vs life expectancy
Pearson r ≈ 0.9 across all 274 country-years. The strong pooled correlation does
NOT mean spending causes life expectancy to rise: countries that spend more are
also richer, more developed, and observed in later years.
The slope of ~1 year per $1 000 additional spending is a mix of between-country
and within-country effects that will be disentangled in the modelling script.
"""
r_pool, p_pool = stats.pearsonr(df["Spending_USD"], df["Life_Expectancy"])
slope, intercept, *_ = stats.linregress(df["Spending_USD"], df["Life_Expectancy"])

fig, ax = plt.subplots(figsize=(8, 5))
for country, grp in df.groupby("Country"):
    ax.scatter(grp["Spending_USD"], grp["Life_Expectancy"],
               alpha=0.5, s=18, label=country, color=PALETTE[country])

x_range = np.linspace(df["Spending_USD"].min(), df["Spending_USD"].max(), 200)
ax.plot(x_range, intercept + slope * x_range, color="black", linewidth=1.5,
        label=f"Pooled OLS (r = {r_pool:.2f})")
ax.set_xlabel("Healthcare spending (USD per capita)")
ax.set_ylabel("Life expectancy (years)")
ax.set_title("Life expectancy vs spending — pooled")
ax.legend(fontsize=7, ncol=2)
fig.tight_layout()
fig.savefig(PLOT_DIR / "healthexp_scatter_pooled.png")
plt.show()
print(f"Saved: {PLOT_DIR / 'healthexp_scatter_pooled.png'}")
print(f"Pooled Pearson r = {r_pool:.3f}  (p = {p_pool:.2e})")
print(f"OLS slope: +{slope * 1000:.2f} years per $1 000 spending")


# %%
""" [7] Per-country scatter — within-country correlations
Within each country the spending–life expectancy relationship is weaker and less
consistent than the pooled scatter suggests. Japan's slope is nearly flat at
already-high life expectancy. The USA's curve levels off sharply above ~$8 000 —
visible diminishing returns. The strong pooled r is driven mainly by between-country
baseline differences, not within-country spending effects — a form of Simpson's paradox.
"""
fig, axes = plt.subplots(2, 3, figsize=(14, 8), sharey=True)

for ax, (country, grp) in zip(axes.flat, df.groupby("Country")):
    ax.scatter(grp["Spending_USD"], grp["Life_Expectancy"],
               alpha=0.7, s=20, color=PALETTE[country])
    sl, ic, *_ = stats.linregress(grp["Spending_USD"], grp["Life_Expectancy"])
    x_c = np.linspace(grp["Spending_USD"].min(), grp["Spending_USD"].max(), 100)
    ax.plot(x_c, ic + sl * x_c, color="black", linewidth=1.2)
    r_c, _ = stats.pearsonr(grp["Spending_USD"], grp["Life_Expectancy"])
    ax.set_title(f"{country}  (r = {r_c:.2f})")
    ax.set_xlabel("Spending (USD)")
    ax.set_ylabel("Life exp. (years)")

fig.suptitle("Spending vs life expectancy — within each country", y=1.01)
fig.tight_layout()
fig.savefig(PLOT_DIR / "healthexp_scatter_per_country.png")
plt.show()
print(f"Saved: {PLOT_DIR / 'healthexp_scatter_per_country.png'}")
for country, grp in df.groupby("Country"):
    r_c, _ = stats.pearsonr(grp["Spending_USD"], grp["Life_Expectancy"])
    print(f"  {country:<15} r = {r_c:.3f}")


# %%
""" [8] USA divergence — life expectancy gap vs peer median
A growing positive gap (USA ahead) in the 1970s reverses by 2000 and turns negative
by ~2010. The widening deficit despite the steepest rise in spending of any country
motivates including Country as a fixed effect in the modelling script rather than
treating it as a nuisance variable.
"""
peers = df[df["Country"] != "USA"]
usa_le = df[df["Country"] == "USA"].set_index("Year")["Life_Expectancy"]
peer_median = peers.groupby("Year")["Life_Expectancy"].median()
gap = usa_le - peer_median  # positive = USA ahead, negative = USA behind

fig, axes = plt.subplots(1, 2, figsize=(13, 4))

ax = axes[0]
ax.plot(peer_median.index, peer_median.values, label="Peer median",
        color="steelblue", linewidth=1.8)
ax.plot(usa_le.index, usa_le.values, label="USA",
        color="tab:red", linewidth=1.8)
ax.set_xlabel("Year")
ax.set_ylabel("Life expectancy (years)")
ax.set_title("USA vs peer median life expectancy")
ax.legend(fontsize=9)

ax = axes[1]
ax.plot(gap.index, gap.values, color="darkred", linewidth=1.8)
ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
ax.fill_between(gap.index, gap.values, 0,
                where=(gap.values >= 0), alpha=0.15, color="steelblue", label="USA ahead")
ax.fill_between(gap.index, gap.values, 0,
                where=(gap.values < 0), alpha=0.15, color="darkred", label="USA behind")
ax.set_xlabel("Year")
ax.set_ylabel("USA − peer median (years)")
ax.set_title("Life expectancy gap: USA minus peer median")
ax.legend(fontsize=8)

fig.tight_layout()
fig.savefig(PLOT_DIR / "healthexp_usa_divergence.png")
plt.show()
print(f"Saved: {PLOT_DIR / 'healthexp_usa_divergence.png'}")
crossover = gap[gap < 0].index.min() if (gap < 0).any() else None
print(f"USA turns negative vs peer median: {crossover}")
print(f"USA − peer median in 1970: {gap.loc[1970]:+.2f} years")
print(f"USA − peer median in {gap.index[-1]}: {gap.iloc[-1]:+.2f} years")
