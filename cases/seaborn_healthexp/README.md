# `healthexp`: Healthcare Spending and Life Expectancy

> Annual health expenditure and life expectancy for six wealthy nations from 1970 to 2020 -- a compact panel dataset for studying whether spending more on healthcare buys longer lives.

## Contents

- [Domain Context](#domain-context)
- [Online Resources](#online-resources)
- [Codebook](#codebook)
- [What You Can Learn Here](#what-you-can-learn-here)
- [Research Questions](#research-questions)
- [Available Scripts](#available-scripts)

## Domain Context

Health economists and policymakers use data like this to evaluate whether higher public or private health expenditure translates into measurable population health gains. Each row represents one country-year observation: how much that country spent per person on healthcare and how long the average citizen was expected to live at birth. The dataset covers six high-income countries (USA, Germany, Japan, Canada, Great Britain, France) over five decades, creating a natural panel structure that exposes both within-country trends and cross-country differences. The USA stands out as a persistent outlier -- it outspends all peers by a wide margin without achieving a correspondingly higher life expectancy. Or does it?

## Online Resources

- **Dataset**: [seaborn-data/healthexp.csv](https://github.com/mwaskom/seaborn-data/blob/master/healthexp.csv)
- **Documentation**: [seaborn.load_dataset](https://seaborn.pydata.org/generated/seaborn.load_dataset.html)

## Codebook

| Column | Type | Range / Values | Description |
|--------|------|----------------|-------------|
| `Country` | categorical | Canada, France, Germany, Great Britain, Japan, USA | Country name |
| `Year` | int | 1970-2020 | Calendar year of observation |
| `Spending_USD` | float | USD per capita | Annual healthcare expenditure per person in USD |
| `Life_Expectancy` | float | years | Period life expectancy at birth |

**Target**: `Life_Expectancy`: continuous, in years. No missing values. The USA follows a diverging trajectory relative to peer countries despite the highest spending.

## What You Can Learn Here

- Panel data structure: interpreting within-country trends separately from cross-country level differences
- Country as a confound: why pooling all observations into one regression misleads, and how one-hot encoding absorbs baseline differences
- The USA as a high-leverage outlier: how a single country can dominate a regression slope
- Diminishing returns: spending beyond a threshold yields smaller and smaller life expectancy gains

## Research Questions

**EDA**
1. Plot spending and life expectancy over time for each country. Which country spends the most, and does it have the highest life expectancy?
2. Is there a positive correlation between spending and life expectancy across all country-years? Does the relationship look different when examined within each country separately?
3. The USA consistently spends far more than other countries. Is its life expectancy trajectory converging toward or diverging from its peers over time?

**Modeling**
1. Fit a simple linear regression of `Life_Expectancy` on `Spending_USD` across all country-years; interpret the slope in plain language.
2. Add `Country` as a one-hot encoded fixed effect and `Year` as a numeric feature; quantify how much of the life expectancy variance is explained by country identity vs. spending level.
3. Evaluate permutation importance and standardized coefficients to determine which features the model relies on most, and check residuals for systematic patterns.

---

## Available Scripts

- `healthexp_01_linreg.py`: linear regression, OLS normal equation, standardization, one-hot encoding, permutation importance, standardized coefficients, residual analysis

**General intent of scripts:** regression

See [QUESTIONS.md](QUESTIONS.md) for per-script code-reading questions.

## Disclaimer

Part of the dataset description above was compiled by AI. Check any assumptions, claims, and context.