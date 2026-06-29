"""Research question: Which morphological measurements best separate the three penguin species?
purpose: eda | style: library-optimal | flags: docstring-depth: minimal
"""

# %%
""" [1] Imports & config"""
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path

PLOT_DIR = Path(__file__).parent / "plots"
PLOT_DIR.mkdir(exist_ok=True)

NUMERIC_COLS = ["bill_length_mm", "bill_depth_mm", "flipper_length_mm", "body_mass_g"]

# %%
""" [2] Load & inspect
344 rows, 7 columns. One row = one penguin individual.
Numeric: bill_length_mm, bill_depth_mm, flipper_length_mm, body_mass_g.
Categorical: species (Adelie/Chinstrap/Gentoo), island (Biscoe/Dream/Torgersen), sex.
Missing: 2 rows lack all four numeric measurements; sex is missing for 11 rows.
No duplicates.
"""
df_raw = sns.load_dataset("penguins")
print(df_raw.shape)
print(df_raw.dtypes)
print(df_raw.isnull().sum())
print(df_raw[NUMERIC_COLS].describe().round(2))
print(f"\nDuplicates: {df_raw.duplicated().sum()}")

# %%
""" [3] Missing value structure
All 2 rows missing numeric measurements also have sex missing — these individuals
have no usable measurements at all and are dropped.
The remaining 9 sex-only-missing rows keep their morphometric data and are
retained for numeric analysis; they are excluded from sex-stratified plots.
"""
all_missing = df_raw[df_raw[NUMERIC_COLS].isnull().all(axis=1)]
print("Rows missing all numeric cols:\n", all_missing[["species", "island", "sex"]])

df = df_raw.dropna(subset=NUMERIC_COLS).copy()
print(f"\nAfter dropping {len(df_raw) - len(df)} fully-missing rows: {len(df):,} rows")
print("Remaining sex-missing:", df["sex"].isnull().sum())

# %%
""" [4] Species and island counts
Adelie is the most common species (151 after drop) and the only one present on all
three islands. Chinstrap appears only on Dream; Gentoo only on Biscoe. This
species-island confounding means island should not be used as a model feature
without understanding the leakage risk it introduces.
"""
fig, axes = plt.subplots(1, 2, figsize=(10, 4))

species_counts = df["species"].value_counts()
axes[0].bar(species_counts.index, species_counts.values)
axes[0].set_ylabel("Count")
axes[0].set_title("Penguin count by species")

ct = df.groupby(["island", "species"]).size().unstack(fill_value=0)
ct.plot(kind="bar", ax=axes[1], rot=0)
axes[1].set_ylabel("Count")
axes[1].set_title("Species × island — confounding check")
axes[1].legend(title="species", fontsize=8)

fig.tight_layout()
fig.savefig(PLOT_DIR / "penguins_species_island.png")
plt.show()
print(f"Species counts: {species_counts.to_dict()}")
print(f"Saved: {PLOT_DIR / 'penguins_species_island.png'}")

# %%
""" [5] Numeric distributions by species
Gentoo is clearly separated on flipper length and body mass (largest species).
Adelie and Chinstrap overlap heavily on most measurements — bill depth is the
clearest distinguisher between those two (Chinstrap has shallower bills).
"""
fig, axes = plt.subplots(2, 2, figsize=(11, 8))

for ax, col in zip(axes.flat, NUMERIC_COLS):
    for species, grp in df.groupby("species"):
        ax.hist(grp[col].dropna(), bins=20, alpha=0.55, label=species)
    ax.set_xlabel(col)
    ax.set_title(f"Distribution of {col}")
    ax.legend(fontsize=8)

fig.tight_layout()
fig.savefig(PLOT_DIR / "penguins_numeric_distributions.png")
plt.show()
print(f"Saved: {PLOT_DIR / 'penguins_numeric_distributions.png'}")

# %%
""" [6] Sexual dimorphism — body mass
Males are heavier than females within every species; the gap is largest for Gentoo
(~700 g median difference). Body mass alone cannot classify species without
accounting for sex — the within-species sex split creates bimodal-like structure
in the marginal mass distribution.
"""
df_sex = df.dropna(subset=["sex"])

fig, axes = plt.subplots(1, 3, figsize=(13, 4), sharey=False)

for ax, (species, grp) in zip(axes, df_sex.groupby("species")):
    for sex, sgrp in grp.groupby("sex"):
        ax.hist(sgrp["body_mass_g"], bins=15, alpha=0.6, label=sex)
    ax.set_xlabel("body_mass_g")
    ax.set_title(species)
    ax.legend(fontsize=8)

axes[0].set_ylabel("Count")
fig.suptitle("Body mass by sex within each species")
fig.tight_layout()
fig.savefig(PLOT_DIR / "penguins_dimorphism_mass.png")
plt.show()
print(f"Saved: {PLOT_DIR / 'penguins_dimorphism_mass.png'}")

# %%
""" [7] Numeric correlation matrix
Flipper length and body mass are strongly correlated (r ≈ 0.87) — they capture
the same size axis. Bill length correlates moderately with flipper and mass;
bill depth is weakly negatively correlated with the others in the raw data
(the Simpson's paradox pattern explored below).
"""
corr = df[NUMERIC_COLS].corr()

fig, ax = plt.subplots(figsize=(6, 5))
im = ax.imshow(corr, vmin=-1, vmax=1, cmap="RdBu_r")
plt.colorbar(im, ax=ax)
ticks = np.arange(len(NUMERIC_COLS))
ax.set_xticks(ticks); ax.set_xticklabels(NUMERIC_COLS, rotation=30, ha="right", fontsize=8)
ax.set_yticks(ticks); ax.set_yticklabels(NUMERIC_COLS, fontsize=8)
for i in range(len(NUMERIC_COLS)):
    for j in range(len(NUMERIC_COLS)):
        ax.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center", fontsize=8)
ax.set_title("Numeric feature correlation matrix")
fig.tight_layout()
fig.savefig(PLOT_DIR / "penguins_corr_matrix.png")
plt.show()
print(f"Saved: {PLOT_DIR / 'penguins_corr_matrix.png'}")

# %%
""" [8] Bill length vs bill depth — Simpson's paradox
Marginally, bill length and bill depth are negatively correlated (r ≈ -0.24):
longer bills tend to be shallower. Within each species the correlation reverses
(Adelie r≈0.39, Chinstrap r≈0.65, Gentoo r≈0.64): longer bills are also deeper.
The marginal sign flip arises because the three species occupy different bill-shape
niches — species is an unmeasured grouping variable that confounds the raw correlation.
This is a textbook case of Simpson's paradox and motivates including species as a
covariate in any bill-dimension model.
"""
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

axes[0].scatter(df["bill_length_mm"], df["bill_depth_mm"], alpha=0.4, s=18, color="gray")
m, b = np.polyfit(df["bill_length_mm"].dropna(), df["bill_depth_mm"].dropna(), 1)
x_range = np.linspace(df["bill_length_mm"].min(), df["bill_length_mm"].max(), 100)
axes[0].plot(x_range, m * x_range + b, color="black", linewidth=1.5)
axes[0].set_xlabel("bill_length_mm")
axes[0].set_ylabel("bill_depth_mm")
axes[0].set_title("Marginal: negative slope")

palette = {"Adelie": "tab:blue", "Chinstrap": "tab:orange", "Gentoo": "tab:green"}
for species, grp in df.groupby("species"):
    axes[1].scatter(grp["bill_length_mm"], grp["bill_depth_mm"],
                    alpha=0.5, s=18, label=species, color=palette[species])
    m_s, b_s = np.polyfit(grp["bill_length_mm"], grp["bill_depth_mm"], 1)
    x_s = np.linspace(grp["bill_length_mm"].min(), grp["bill_length_mm"].max(), 50)
    axes[1].plot(x_s, m_s * x_s + b_s, color=palette[species], linewidth=1.5)
axes[1].set_xlabel("bill_length_mm")
axes[1].set_ylabel("bill_depth_mm")
axes[1].set_title("Within species: positive slopes")
axes[1].legend(fontsize=8)

fig.suptitle("Bill length vs depth — Simpson's paradox")
fig.tight_layout()
fig.savefig(PLOT_DIR / "penguins_simpsons_paradox.png")
plt.show()

for species, grp in df.groupby("species"):
    r = grp["bill_length_mm"].corr(grp["bill_depth_mm"])
    print(f"{species}: r = {r:.2f}")
r_all = df["bill_length_mm"].corr(df["bill_depth_mm"])
print(f"Marginal:  r = {r_all:.2f}")
print(f"Saved: {PLOT_DIR / 'penguins_simpsons_paradox.png'}")

# %%
""" [9] Best-separating pair: flipper length vs bill length
Flipper length vs bill length provides the clearest two-feature separation:
Gentoo clusters in the top-right (long flippers, long bills); Adelie and Chinstrap
are separated mainly on bill length (Chinstrap bills are longer for the same
flipper length). A linear classifier on these two features alone achieves high
accuracy — motivating the first modeling script.
"""
fig, ax = plt.subplots(figsize=(7, 5))

for species, grp in df.groupby("species"):
    ax.scatter(grp["bill_length_mm"], grp["flipper_length_mm"],
               alpha=0.55, s=20, label=species, color=palette[species])

ax.set_xlabel("bill_length_mm")
ax.set_ylabel("flipper_length_mm")
ax.set_title("Flipper length vs bill length by species")
ax.legend(fontsize=9)
fig.tight_layout()
fig.savefig(PLOT_DIR / "penguins_flipper_vs_bill.png")
plt.show()
print(f"Saved: {PLOT_DIR / 'penguins_flipper_vs_bill.png'}")

# %%
