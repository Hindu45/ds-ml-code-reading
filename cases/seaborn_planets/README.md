# `planets`: Confirmed Exoplanet Discoveries

> Discovery records for 1,035 confirmed exoplanets - a dataset where heavy missingness is physically meaningful and rare detection methods mark the most unusual objects.

## Contents

- [Domain Context](#domain-context)
- [Online Resources](#online-resources)
- [Codebook](#codebook)
- [What You Can Learn Here](#what-you-can-learn-here)
- [Research Questions](#research-questions)
- [Available Scripts](#available-scripts)

## Domain Context

Each row represents one confirmed exoplanet as catalogued through 2014. The dataset spans 25 years of discovery history and ten detection techniques, dominated by Radial Velocity (553 planets) and Transit photometry (397 planets). Physical quantities span orders of magnitude - orbital periods range from 0.09 days to roughly 2,000 years - so log scaling is required throughout. Missingness is not random: mass cannot be measured by Transit photometry and distance cannot be measured by some other methods, making this a textbook case of data missing not at random (MNAR). Practitioners use data like this to characterize planetary populations and identify selection biases in survey instruments.

## Online Resources

- **Dataset**: [seaborn-data/planets.csv](https://github.com/mwaskom/seaborn-data/blob/master/planets.csv)
- **Documentation**: [seaborn.load_dataset](https://seaborn.pydata.org/generated/seaborn.load_dataset.html)

## Codebook

| Column | Type | Range / Values | Description |
|--------|------|----------------|-------------|
| `method` | str | 10 unique values | Detection technique (Radial Velocity, Transit, Imaging, Microlensing, ...) |
| `number` | int64 | 1 - 7 | Number of confirmed planets in the host system |
| `orbital_period` | float64 | 0.091 - 730,000 days; 4.2% missing | Orbital period in days |
| `mass` | float64 | 0.004 - 25 Jupiter masses; 50.4% missing | Planet mass in Jupiter masses |
| `distance` | float64 | 1.35 - 8,500 parsecs; 21.9% missing | Distance to host star in parsecs |
| `year` | int64 | 1989 - 2014 | Year of discovery |

**Note**: `mass` missingness is physically determined by detection method - Transit photometry cannot directly measure mass - so it is missing not at random (MNAR). Any analysis using mass should be restricted to or stratified by detection method.

## What You Can Learn Here

- Missing not at random (MNAR): mass is missing because some detection methods cannot measure it, not by chance
- Log-normal distributions: all three continuous physical quantities span multiple orders of magnitude and require log scaling
- Survivorship and detection bias: Kepler's 4-year mission window over-represents short-period planets in post-2009 data
- Unsupervised anomaly detection without ground-truth labels: evaluation is qualitative, based on domain knowledge

## Research Questions

**EDA**
1. How many planets were discovered per year, and which detection methods drove the post-2009 surge?
2. How do orbital period and mass distributions differ between detection methods, and why?
3. Is mass missingness random or associated with detection method? What does this imply for any analysis that uses mass as a feature?

**Modeling**
1. Apply Isolation Forest to (log_orbital_period, log_distance) and compare which planets it flags against a simple per-feature z-score baseline.
2. How does the contamination parameter affect which planets are flagged? Which planets are flagged even at the most stringent threshold (core anomalies)?
3. Do the flagged anomalies correspond to physically unusual objects (extreme-orbit imaging detections, galactic-bulge microlensing events) - validating the algorithm against domain knowledge?

---

## Available Scripts

- `planets_01_eda.py`: missing data MNAR, log transform, detection method frequency, bivariate parameter space, discovery year trend
- `planets_02_isolation_forest.py`: Isolation Forest, anomaly detection, z-score baseline, contamination parameter, train/test split

**General intent of scripts:** outlier detection in multivariate space

See [QUESTIONS.md](QUESTIONS.md) for per-script code-reading questions.

## Disclaimer

Part of the dataset description above was compiled by AI. Check any assumptions, claims, and context.
