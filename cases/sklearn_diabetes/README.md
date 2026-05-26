# `diabetes`: Diabetes Disease Progression

> Physiological measurements for 442 patients predicting how much diabetes has progressed one year after baseline -- the go-to regression benchmark for studying regularization.

## Contents

- [Domain Context](#domain-context)
- [Online Resources](#online-resources)
- [Codebook](#codebook)
- [What You Can Learn Here](#what-you-can-learn-here)
- [Research Questions](#research-questions)
- [Available Scripts](#available-scripts)

## Domain Context

Each row represents one diabetes patient measured at a single baseline visit. The 10 input features capture demographics (age, sex), body composition (BMI), blood pressure, and six blood serum assays. The task is to predict a continuous disease-progression score measured one year later, making this a pure regression problem used in endocrinology research to understand which physiological factors drive worsening outcomes.

All 10 features arrive z-score standardized in the sklearn dataset -- values near 0 indicate average, with a shared standard deviation of approximately 0.048 across all columns. The target is not standardized.

## Online Resources

- **Dataset**: [sklearn.datasets.load_diabetes](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_diabetes.html)
- **Original paper**: Efron, B., Hastie, T., Johnstone, I., Tibshirani, R. (2004). Least Angle Regression. *Annals of Statistics*, 32(2), 407-499.

## Codebook

| Column | Type | Range / Values | Description |
|--------|------|----------------|-------------|
| `age` | float64 | -0.107 to 0.111 | Age (z-score standardized) |
| `sex` | float64 | -0.045 or 0.051 | Biological sex (binary, encoded as float) |
| `bmi` | float64 | -0.090 to 0.171 | Body mass index (z-score standardized) |
| `bp` | float64 | -0.112 to 0.132 | Average blood pressure (z-score standardized) |
| `s1` | float64 | -0.127 to 0.154 | Total serum cholesterol (z-score standardized) |
| `s2` | float64 | z-score, std ~0.048 | Low-density lipoproteins (z-score standardized) |
| `s3` | float64 | z-score, std ~0.048 | High-density lipoproteins (z-score standardized) |
| `s4` | float64 | z-score, std ~0.048 | Total cholesterol / HDL ratio (z-score standardized) |
| `s5` | float64 | z-score, std ~0.048 | Log of serum triglycerides (z-score standardized) |
| `s6` | float64 | z-score, std ~0.048 | Blood sugar level (z-score standardized) |

**Target**: `target` (progression): continuous, range 25-346, mean ~152. Not standardized. Quantifies diabetes disease progression one year after baseline.

Note: `sex` has only 2 unique float values (-0.045 and 0.051), not 0/1 -- a common student surprise.

## What You Can Learn Here

- Regression baseline: predicting the mean (DummyRegressor) sets the floor before fitting any model
- Regularization comparison: Ridge shrinks all coefficients, Lasso zeroes some out -- same data, different feature selection behavior
- Pre-processed data recognition: `.describe()` reveals z-score scaling; re-applying StandardScaler is a no-op but a common mistake
- Interpreting standardized coefficients: shared scale makes it valid to compare coefficient magnitudes across features

## Research Questions

**EDA**
1. Which feature correlates most strongly with the target (Pearson r)?
2. Does the relationship between BMI and progression appear linear or curved?
3. Is there a meaningful difference in progression between the two sex groups after controlling for BMI?
4. Which serum features (s1-s6) are most correlated with each other? Does that collinearity affect coefficient interpretation?

**Modeling**
1. Fit OLS linear regression; compute RMSE and R² on a held-out test set as the baseline.
2. Grid-search Ridge and Lasso alpha; compare the sets of selected features -- which does Lasso zero out first?
3. Add polynomial degree-2 features; does regularized regression outperform the linear model, and at what cost in interpretability?

---

## Available Scripts

- `diabetes_01_eda.py`: distributions, correlation, BMI scatter, sex comparison
- `diabetes_02_modeling_poly.py`: OLS, PolynomialFeatures, degree-2, residuals

**General intent of scripts:** linear regression 1D vs 2D features

See [QUESTIONS.md](QUESTIONS.md) for per-script code-reading questions.

## Disclaimer

Part of the dataset description above was compiled by AI. Check any assumptions, claims, and context.
