# `mpg`: Fuel Efficiency of 1970s–80s Cars

> Engine specs and model year for 398 US/European/Japanese cars, with fuel efficiency as the target. A compact dataset where multicollinearity and a real-world policy trend collide.

## Contents

- [Domain Context](#domain-context)
- [Online Resources](#online-resources)
- [Codebook](#codebook)
- [What You Can Learn Here](#what-you-can-learn-here)
- [Research Questions](#research-questions)
- [Available Scripts](#available-scripts)

## Domain Context

Each row is one car model/trim tested for fuel efficiency in miles per gallon (mpg). The data spans 1970–1982, a period bookended by two oil crises (1973, 1979) that forced manufacturers to downsize engines and improve efficiency. Analysts use this dataset to understand what physical attributes drive fuel economy and how regulatory pressure shows up in data, even when no policy variable is explicitly recorded. A practitioner might build a fleet audit tool or predict compliance with emissions standards from vehicle specs.

## Online Resources

- **Dataset**: [seaborn-data / mpg.csv](https://github.com/mwaskom/seaborn-data/blob/master/mpg.csv)
- **seaborn API**: [seaborn.load_dataset](https://seaborn.pydata.org/generated/seaborn.load_dataset.html)
- **Original source**: [UCI Auto MPG Dataset](https://archive.ics.uci.edu/dataset/9/auto+mpg)

## Codebook

| Column | Type | Range / Values | Description |
|--------|------|----------------|-------------|
| `mpg` | float64 | 9.0 – 46.6 | Miles per gallon — **target** |
| `cylinders` | int64 | 3, 4, 5, 6, 8 | Engine cylinder count; behaves like an ordinal category |
| `displacement` | float64 | 68 – 455 | Engine displacement in cubic inches |
| `horsepower` | float64 | 46 – 230 | Engine horsepower; **1.5% missing** (6 US cars, 1970–71) |
| `weight` | int64 | 1613 – 5140 | Curb weight in pounds |
| `acceleration` | float64 | outl. flagged | 0–60 mph time in seconds (longer = slower) |
| `model_year` | int64 | 70 – 82 | Last two digits of model year (70 = 1970) |
| `origin` | str | usa / japan / europe | Manufacturing region |
| `name` | str | 305 unique | Car model name — high cardinality, typically dropped |

**Target**: `mpg`: continuous, moderately right-skewed (range 9–46.6). The physical inverse (gallons-per-mile = 1/mpg) is closer to symmetric but mpg is the industry-standard reported form. No transformation is applied in these scripts.

## What You Can Learn Here

- Multicollinearity: cylinders, displacement, horsepower, and weight are correlated at r > 0.9 — Ridge regularisation is the natural response
- Lasso vs Ridge: Lasso's feature-selection property vs Ridge's coefficient-shrinkage trade-off, visualised via coefficient paths
- Missing not at random: the 6 missing horsepower rows are all early US cars — dropping vs imputing carries different assumptions
- Suppressor effects: acceleration correlates *positively* with mpg not because slow driving saves fuel but because small-engine cars both accelerate slowly and sip fuel
- Time as a confound: `model_year` captures regulatory and design-era effects that persist after controlling for engine specs
- Ordinal quasi-categorical features: cylinders takes only 5 discrete values and should be treated as ordinal, not continuous

## Research Questions

**EDA**
1. How does average fuel efficiency change by number of cylinders and by origin?
2. Which technical variables correlate most strongly with mpg, and are any nearly redundant with each other?
3. Controlling for vehicle weight and engine size, did fuel efficiency improve over model years? Does it suggest efficiency gains beyond just downsizing?

**Modeling**
1. Predict mpg from weight alone using linear regression; evaluate fit.
2. Add horsepower (after dropping missing rows), cylinders, and model year; assess multicollinearity (correlation matrix) and re-evaluate.
3. Use Ridge and Lasso regularisation to handle multicollinearity; compare coefficient paths and selected features to domain expectations.

---

## Available Scripts

- [`mpg_01_eda.py`](mpg_01_eda.py): distributions, missing data, time trend, origin comparison, multicollinearity heatmap, acceleration suppressor
- [`mpg_02_ridge_lasso.py`](mpg_02_ridge_lasso.py): Ridge, Lasso, OLS baseline, hyperparameter grid search, coefficient paths, three-way split
- [`mpg_03_bias_variance_ridge.py`](bias_variance_tradeoff_ridge.py): learning curves, validation curve, alpha as complexity dial, CV stability vs single split
- [`mpg_04_bias_variance_decisiontree.py`](bias_variance_tradeoff_decisiontree.py): same as above: max_depth as complexity dial

**General intent of scripts:** regularized regression

See [QUESTIONS.md](QUESTIONS.md) for per-script code-reading questions.

## Disclaimer

Part of the dataset description above was compiled by AI. Check any assumptions, claims, and context.