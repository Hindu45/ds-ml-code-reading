"""EDA for the planets dataset: which exoplanet discoveries stand out as anomalies in the confirmed population?"""

# %%
""" [1] Imports & config"""
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path

PLOT_DIR = Path(__file__).parent / "plots"
PLOT_DIR.mkdir(exist_ok=True)

NUMERIC_COLS = ["orbital_period", "mass", "distance", "year"]


# %%
""" [2] Load & inspect
1035 rows, 6 columns. One row = one confirmed exoplanet (data up to 2014).
  method         — detection technique (10 unique values)
  number         — planets in the host system (1–7)
  orbital_period — orbit duration in days (0.09 – 730,000); 4.2% missing
  mass           — planet mass in Jupiter masses (0.004 – 25); 50.4% missing
  distance       — distance to host star in parsecs (1 pc ≈ 3.26 ly); 21.9% missing
  year           — year of discovery (1989–2014)
4 duplicate rows are dropped before analysis.
mass and distance missingness are physically determined by detection method — not random.
The three continuous physical quantities span orders of magnitude and require log
scaling throughout: linear histograms are illegible.
"""
df_raw = sns.load_dataset("planets")
print(df_raw.shape)
print(df_raw.dtypes)
print(df_raw.isnull().sum())
print(df_raw[NUMERIC_COLS].describe().round(3))

df = df_raw.drop_duplicates().copy()
print(f"\nAfter dropping {df_raw.duplicated().sum()} duplicates: {len(df):,} rows")


# %%
""" [3] Missing data
mass (50%) and distance (22%) are the dominant missing columns.
Missingness is NOT random — it is determined by detection physics:
  Radial Velocity measures the host star's Doppler wobble → can infer planet mass.
  Transit photometry detects the planet's shadow → cannot directly measure mass.
Treating missing values as random would bias anomaly scores.
For the isolation forest step we will either (a) restrict to RV planets where
mass is available, or (b) drop mass and score on orbital_period + distance only.
"""
missing_pct = df.isnull().mean() * 100

fig, ax = plt.subplots(figsize=(7, 4))
bars = ax.barh(missing_pct.index, missing_pct.values)
ax.bar_label(bars, fmt="%.1f%%", padding=3)
ax.set_xlabel("Missing (%)")
ax.set_title("Missing values by column")
ax.set_xlim(0, 65)
fig.tight_layout()
fig.savefig(PLOT_DIR / "planets_missing.png")
plt.show()

mass_missing_rate = (
    df.groupby("method")["mass"]
    .apply(lambda s: s.isnull().mean() * 100)
    .sort_values(ascending=False)
    .round(1)
)
print("Mass missing rate by detection method (%):")
print(mass_missing_rate.to_string())


# %%
""" [4] Detection method frequency
Radial Velocity (RV, 553) and Transit (397) account for 92% of all discoveries.
The 8 minor methods — Imaging, Microlensing, Eclipse Timing Variations, etc. —
cover a physically distinct corner of parameter space: wide orbits, young systems,
or galactic-bulge distances. These rare detections are strong anomaly candidates
even before any algorithm is applied.
"""
method_counts = df["method"].value_counts()

fig, ax = plt.subplots(figsize=(9, 4))
ax.barh(method_counts.index[::-1], method_counts.values[::-1])
for i, (v, _) in enumerate(zip(method_counts.values[::-1], method_counts.index[::-1])):
    ax.text(v + 3, i, str(v), va="center", fontsize=8)
ax.set_xlabel("Number of planets")
ax.set_title("Exoplanet discoveries by detection method")
fig.tight_layout()
fig.savefig(PLOT_DIR / "planets_method_counts.png")
plt.show()


# %%
""" [5] Individual distributions (log-scaled)
All three physical quantities span multiple orders of magnitude:
  orbital_period: 0.09 days (ultra-hot Jupiters) to 730,000 days (~2,000 years)
  mass:           0.004 to 25 Jupiter masses (> 13 MJ = brown dwarf territory)
  distance:       1.35 to 8,500 parsecs — from nearby stars to the galactic bulge
Log₁₀ histograms reveal multimodal structure invisible on a linear scale.
orbital_period shows a pronounced gap around log₁₀(period) ≈ 1.5–2 (30–100 days)
separating hot Jupiters from cold, wide-orbit planets.
"""
log_cols = ["orbital_period", "mass", "distance"]
fig, axes = plt.subplots(1, 3, figsize=(13, 4))
for ax, col in zip(axes, log_cols):
    vals = np.log10(df[col].dropna())
    ax.hist(vals, bins=40, edgecolor="none")
    ax.axvline(vals.mean(),   color="red",    linestyle="--", linewidth=1.2, label="mean")
    ax.axvline(vals.median(), color="orange", linestyle=":",  linewidth=1.2, label="median")
    ax.set_xlabel(f"log₁₀({col})")
    ax.set_title(col)
    ax.legend(fontsize=7)
fig.suptitle("Log₁₀ distributions of physical quantities  (n varies due to missing values)")
fig.tight_layout()
fig.savefig(PLOT_DIR / "planets_log_distributions.png")
plt.show()


# %%
""" [6] Discovery year trend
Discovery rate per year is the clearest signal of instrumentation history:
  1989–2008: ground-based RV surveys; slow ramp (~10–30/year at peak)
  2009:      Kepler Space Telescope first light — launches an exponential rise
  2014:      last year in this dataset
The Kepler-era planets are mostly compact, short-period, low-mass worlds —
they form the dense "normal" cluster in parameter space that Isolation Forest
will use as its baseline when identifying unusual detections.
"""
discoveries_per_year = df["year"].value_counts().sort_index()

fig, ax = plt.subplots(figsize=(9, 4))
ax.bar(discoveries_per_year.index, discoveries_per_year.values, width=0.8)
ax.axvline(2009, color="red", linestyle="--", linewidth=1.2, label="Kepler launch (2009)")
ax.set_xlabel("Year")
ax.set_ylabel("Planets discovered")
ax.set_title("Exoplanet discoveries per year")
ax.legend()
fig.tight_layout()
fig.savefig(PLOT_DIR / "planets_discovery_year.png")
plt.show()


# %%
""" [7] Multi-planet systems
Each row is an individual planet; 'number' records how many confirmed planets
share that host star. Most host stars (~78%) have only one known planet —
partly true scarcity, partly observational incompleteness.
A 7-planet system contributes 7 rows all with number=7.
High planet counts indicate Solar-System-like architectures, not extreme physics.
They are not physical outliers, but they are a distinct structural sub-population.
"""
number_counts = df["number"].value_counts().sort_index()
print("Planets per system distribution:")
print(number_counts.to_string())
print(f"Single-planet fraction: {number_counts[1] / number_counts.sum() * 100:.1f}%")

fig, ax = plt.subplots(figsize=(6, 4))
ax.bar(number_counts.index.astype(str), number_counts.values)
ax.set_xlabel("Planets in host system")
ax.set_ylabel("Rows (individual planets)")
ax.set_title("Multi-planet system distribution")
fig.tight_layout()
fig.savefig(PLOT_DIR / "planets_multiplanet.png")
plt.show()


# %%
""" [8] Bivariate: orbital period vs mass
The classic exoplanet parameter space on log-log axes reveals three populations:
  Hot Jupiters  — short period (< 10 d), high mass (> 0.3 MJ); RV + Transit
  Cold Jupiters — long period (> 100 d), moderate-to-high mass; mostly RV
  Super-Earths  — short period (< 100 d), low mass (< 0.1 MJ); mostly Transit
Points in the top-right corner (wide orbit + high mass) and isolated points far
from any cluster are the prime candidates for Isolation Forest to flag.
Only rows with both columns are shown (n ≈ 500; most missing-mass rows are Transit).
"""
has_both = df.dropna(subset=["orbital_period", "mass"])
methods = has_both["method"].unique()
palette = plt.cm.tab10(np.linspace(0, 1, len(methods)))

fig, ax = plt.subplots(figsize=(9, 6))
for method, color in zip(methods, palette):
    sub = has_both[has_both["method"] == method]
    ax.scatter(
        np.log10(sub["orbital_period"]),
        np.log10(sub["mass"]),
        label=f"{method} (n={len(sub)})",
        alpha=0.6, s=20, color=color,
    )
ax.set_xlabel("log₁₀(orbital period, days)")
ax.set_ylabel("log₁₀(mass, Jupiter masses)")
ax.set_title("Orbital period vs mass — by detection method")
ax.legend(fontsize=7, loc="upper left", framealpha=0.8)
fig.tight_layout()
fig.savefig(PLOT_DIR / "planets_period_vs_mass.png")
plt.show()


# %%
""" [9] Correlation matrix (log-transformed)
Correlations computed on log-scaled physical quantities plus year and number.
Pair-wise complete observations only — n differs per pair due to missingness.
Key relationships:
  log_period ↔ log_distance (+0.4): long-period planets cluster around nearby bright
    stars — only those were monitored continuously long enough to complete an orbit.
  log_mass ↔ log_period (+0.3): detection bias — heavier planets are easier to detect
    at any period, but wide-orbit detections are rare and filter for the most massive.
  year ↔ log_period (negative): Kepler's 4-year mission window strongly favours
    short-period planets; the post-2009 data pull the mean period downward.
"""
df_log = df.copy()
for col in ["orbital_period", "mass", "distance"]:
    df_log[col] = np.log10(df_log[col])

corr_cols = ["orbital_period", "mass", "distance", "year", "number"]
corr_labels = ["log period", "log mass", "log distance", "year", "n_planets"]
corr = df_log[corr_cols].corr()

fig, ax = plt.subplots(figsize=(6, 5))
im = ax.imshow(corr, vmin=-1, vmax=1, cmap="RdBu_r")
plt.colorbar(im, ax=ax)
ticks = np.arange(len(corr_cols))
ax.set_xticks(ticks)
ax.set_xticklabels(corr_labels, rotation=45, ha="right")
ax.set_yticks(ticks)
ax.set_yticklabels(corr_labels)
for i in range(len(corr_cols)):
    for j in range(len(corr_cols)):
        ax.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center", fontsize=8)
ax.set_title("Correlation matrix — log-scaled features")
fig.tight_layout()
fig.savefig(PLOT_DIR / "planets_correlation.png")
plt.show()


# %%
""" [10] Extreme planets — manual outlier candidates
Manual inspection of the most extreme planets along each axis.
These rows are the ones most likely to receive high anomaly scores from
Isolation Forest in the follow-up model script.
  Longest orbital period (~730,000 days ≈ 2,000 years): directly imaged wide-orbit
    giants around HR 8799; detected by Imaging, not RV — a physically separate regime.
  Highest mass (~25 MJ): borderline brown dwarfs; by IAU convention objects > 13 MJ
    are no longer classified as planets — these sit at the definition boundary.
  Furthest distance (8,500 pc): microlensing detections toward the galactic bulge;
    detected via gravitational lensing of background stars, not host-star light.
"""
print("\n--- Longest orbital period ---")
print(df.nlargest(5, "orbital_period")[["method", "orbital_period", "mass", "distance", "year"]].to_string())

print("\n--- Highest mass ---")
print(df.nlargest(5, "mass")[["method", "orbital_period", "mass", "distance", "year"]].to_string())

print("\n--- Most distant ---")
print(df.nlargest(5, "distance")[["method", "orbital_period", "mass", "distance", "year"]].to_string())

print("\n--- Most planets in host system ---")
print(df.nlargest(5, "number")[["method", "number", "orbital_period", "year"]].to_string())
