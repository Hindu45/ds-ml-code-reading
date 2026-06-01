# `tips`: Restaurant Tipping Behavior

> 244 restaurant bills recording spend, tip, and diner attributes - a compact dataset for studying what drives tipping generosity.

## Contents

- [Domain Context](#domain-context)
- [Online Resources](#online-resources)
- [Codebook](#codebook)
- [What You Can Learn Here](#what-you-can-learn-here)
- [Research Questions](#research-questions)
- [Available Scripts](#available-scripts)

## Domain Context

The dataset captures bills recorded by a single waiter over several months at one restaurant. Each row is one table visit: the total amount billed, the tip left, and attributes of the dining party (sex of the payer, smoker status, day of week, meal time, and party size). The waiter's own behavior is held constant, so variation in tipping reflects diner choices rather than service differences. Applied use cases include hospitality analytics, consumer behavior research, and introductory regression teaching.

## Online Resources

- **Dataset**: [seaborn-data/tips.csv](https://github.com/mwaskom/seaborn-data/blob/master/tips.csv)
- **Documentation**: [seaborn.load_dataset](https://seaborn.pydata.org/generated/seaborn.load_dataset.html)

## Codebook

| Column | Type | Range / Values | Description |
|--------|------|----------------|-------------|
| `total_bill` | float64 | 3.07 - 50.81 | Total bill amount in USD |
| `tip` | float64 | 1.00 - 10.00 | Tip amount in USD |
| `sex` | category | Male, Female | Sex of the bill payer |
| `smoker` | category | Yes, No | Whether the party included a smoker |
| `day` | category | Thur, Fri, Sat, Sun | Day of the week |
| `time` | category | Dinner, Lunch | Meal time |
| `size` | int64 | 1 - 6 | Number of people in the party |

**Target**: `tip`: continuous, USD 1.00-10.00. Scripts also derive `tip_pct = tip / total_bill * 100` as an alternative target that controls for bill size.

## What You Can Learn Here

- Confounding: party size inflates both total_bill and tip simultaneously, masking causal inference
- Regression with mixed features: combining a continuous predictor (total_bill) with one-hot encoded categorical variables
- The difference between predicting tip amount vs. tip rate as targets, and when each is more appropriate
- Partial regression (Frisch-Waugh-Lovell theorem) to isolate one predictor's effect after controlling for all others
- K-means segmentation on behavioral variables and post-hoc demographic profiling of discovered clusters
- Choosing k via elbow (inertia) and silhouette score, and reading a per-sample silhouette plot

## Research Questions

**EDA**
1. What is the distribution of tip percentage (tip / total_bill x 100)? What is a "typical" tip rate?
2. Do tip rates differ by sex, smoker status, day of week, or time of day?
3. After controlling for party size, is the effect of sex on tip percentage still present, or does it disappear?

**Modeling**
1. Predict tip amount from total_bill using simple linear regression; compare the OLS closed-form solution to gradient descent from scratch.
2. Add all available features (total_bill, size, sex, smoker, day, time); assess which variables most improve prediction of tip using permutation importance.
3. Use the Frisch-Waugh-Lovell theorem to isolate the partial effect of party size after partialling out all other predictors; verify it matches the full-model coefficient.

**Clustering**
1. Do behavioral variables (spend, tip rate, party size) cluster diners into distinct archetypes? How many segments does the elbow/silhouette analysis suggest?
2. After assigning segment labels, do any segments skew demographically (sex, smoker status, time, day) — even though k-means never saw those variables?

---

## Available Scripts

- `tips_01_eda.py`: tip_pct distribution, categorical comparison, confounding, smoker-sex interaction
- `tips_02_single_feature.py`: linear regression, gradient descent, OLS normal equation, MSE, R2
- `tips_03_all_features.py`: multiple regression, one-hot encoding, permutation importance, partial regression, heteroscedasticity
- `tips_04_kmeans_segmentation.py`: k-means clustering, StandardScaler, elbow plot, silhouette score, segment profiling, demographic crosstab

**General intent of scripts:** regression (adding features sequentially), unsupervised clustering

See [QUESTIONS.md](QUESTIONS.md) for per-script code-reading questions.

## Disclaimer

Part of the dataset description above was compiled by AI. Check any assumptions, claims, and context.
