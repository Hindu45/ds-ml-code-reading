# `diamonds`: Diamond Prices and Quality Grades

> Physical measurements and quality grades for 53,940 diamonds, with retail price as the target -- a rich dataset for studying multicollinearity, ordinal encoding, and price paradoxes.

## Contents

- [Domain Context](#domain-context)
- [Online Resources](#online-resources)
- [Codebook](#codebook)
- [What You Can Learn Here](#what-you-can-learn-here)
- [Research Questions](#research-questions)
- [Available Scripts](#available-scripts)

## Domain Context

Each row is one diamond listed by a retailer. The four "Cs" (carat, cut, color, clarity) are the traditional trade grading system: carat measures size, while the other three are quality grades assessed by certified gemologists. Physical dimensions (x, y, z, depth, table) are derived from measurement and are closely correlated with carat. Price is set by the retailer and reflects all four Cs plus supply and demand dynamics. A key puzzle in this data is the price paradox: lower-grade cuts sometimes command higher prices than Ideal cuts because larger stones tend to receive lower-grade cuts, so carat confounds the apparent quality-price relationship.

## Online Resources

- **Dataset**: [seaborn-data/diamonds.csv](https://github.com/mwaskom/seaborn-data/blob/master/diamonds.csv)
- **Documentation**: [seaborn.load_dataset](https://seaborn.pydata.org/generated/seaborn.load_dataset.html)

## Codebook

| Column | Type | Range / Values | Description |
|--------|------|----------------|-------------|
| `carat` | float | 0.20 - 5.01 | Weight of the diamond in carats |
| `cut` | category | Fair, Good, Very Good, Premium, Ideal | Quality of the cut (ordered; Ideal is best) |
| `color` | category | D, E, F, G, H, I, J | Color grade (ordered; D is colorless / best, J is most yellow) |
| `clarity` | category | I1, SI2, SI1, VS2, VS1, VVS2, VVS1, IF | Clarity grade (ordered; IF is best) |
| `depth` | float | 43.0 - 79.0 | Total depth percentage: z / mean(x, y) x 100 |
| `table` | float | 43.0 - 95.0 | Width of the top facet relative to widest point (%) |
| `x` | float | 0.0 - 10.74 | Length in mm (20 rows have x = y = z = 0, treated as data entry errors) |
| `y` | float | 0.0+ | Width in mm |
| `z` | float | 0.0+ | Depth in mm |
| `price` | int | 326 - 18,823 | Retail price in USD |

**Target**: `price`: integer, $326-$18,823, strongly right-skewed (mean $3,933, std $3,989). A log transform yields a near-normal distribution. Raw price is modeled in the scripts to keep predictions in USD.

**Data quality notes**: 146 duplicate rows (0.3%); 20 rows with x = y = z = 0 (dropped in scripts). Outlier values flagged (|z| > 3) in carat, depth, table, x, and price.

## What You Can Learn Here

- Right-skewed targets and when to apply log transformations before or after modeling
- Multicollinearity among carat and the three physical dimensions (x, y, z), and how Ridge regression addresses inflated coefficient variance
- Ordinal encoding trade-offs for quality grades (cut, color, clarity) vs. one-hot encoding with drop_first
- The price paradox as a concrete example of confounding: naive groupby comparisons can reverse apparent effects

## Research Questions

**EDA**
1. What is the distribution of diamond prices, and how does it change under a log transform?
2. How does median price vary across cut, color, and clarity grades? Do higher quality grades always command higher prices?
3. Which numeric features correlate most strongly with price, and which groups of features are nearly redundant with each other?
4. Why do Fair-cut diamonds sometimes cost more than Ideal-cut diamonds, and which variable explains this reversal?

**Modeling**
1. Predict diamond price from carat alone using linear regression; evaluate RMSE and R2.
2. Add cut, color, and clarity as one-hot encoded features; quantify the improvement and inspect coefficients for the quality grades.
3. Apply Ridge and Lasso regression to handle multicollinearity; use a validation set to select the regularisation strength alpha and compare coefficient paths for both methods.
4. Compare OLS, Ridge, and Lasso on a held-out test set; identify which features Lasso drives to zero and whether the zeroed features are the redundant dimension columns.

---

## Available Scripts

- `diamonds_01_eda.py`: price distribution, log transform, correlation heatmap, price by category, carat scatter
- `diamonds_02_ridge_lasso.py`: OLS baseline, Ridge, Lasso, alpha grid search, coefficient paths, test evaluation

**General intent of scripts:** regression, dimension reduction

See [QUESTIONS.md](QUESTIONS.md) for per-script code-reading questions.

## Disclaimer

Part of the dataset description above was compiled by AI. Check any assumptions, claims, and context.
