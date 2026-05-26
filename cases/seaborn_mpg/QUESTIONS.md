# Code Reading Questions for seaborn_mpg

TBD: Review

- [mpg_01_eda.py](#questions-mpg_01_edapy)
- [mpg_02_ridge_lasso.py](#questions-mpg_02_ridge_lassopy)
- [mpg_03_bias_variance_decisiontree.py](#questions-mpg_03_bias_variance_decisiontreepy)
- [mpg_04_bias_variance_ridge.py](#questions-mpg_04_bias_variance_ridgepy)

## Questions `mpg_01_eda.py`

**Script topics** · EDA: Data Quality · EDA: Distributions · EDA: Correlations

**Q1** · `eda.sampling-bias`

- In cell [2], what years and origins does this dataset cover (model years 70-82, origins usa/europe/japan)?
- Which combination of origin and cylinder count is likely over-represented?
- How might this sampling affect a model trained to predict mpg for modern cars?

**Q2** · `process.simplicity`

- In cell [3], the six rows with missing horsepower are dropped. What assumption does this embed about why the values are missing?
- The docstring says the missingness is "likely unreported in early emissions records." What category of missingness (MCAR, MAR, MNAR) does that imply, and why does it matter for the drop-vs-impute decision?

**Q3** · `eda.plot-assumption`

- Cell [4] plots raw mpg and the inverse 1/mpg side by side. Describe the shape of each distribution.
- Which would be the better modeling target for ordinary least-squares regression, and why?
- The script chooses to model mpg directly. What trade-off does that decision introduce?

**Q4** · `eda.confounding`

- Cell [5] shows fleet mpg rising sharply after 1973 and 1979. Is `model_year` a genuine predictor or a proxy for unmeasured regulatory and design-era effects?
- If both `model_year` and engine-size features are included in the same regression, what does the `model_year` coefficient actually estimate?

**Q5** · `eda.confounding`

- Cell [6] shows median mpg of 31.6 (japan), 26.0 (europe), and 18.5 (usa). The docstring says origin is "partly a proxy for cylinders/displacement."
- What would happen to the estimated coefficient of `origin` if `cylinders` and `displacement` were also in the model?

**Q6** · `eda.plot-assumption`

- In cell [9], which three predictors have pairwise correlations above 0.9?
- Why does high multicollinearity inflate OLS coefficient variance?
- How does the script connect this observation to the choice of model in the next script?

**Q7** · `eda.plot-assumption`

- Cell [10] shows a curved, not straight-line, relationship between mpg and weight. What does this imply for a linear model trained on raw weight values?
- The docstring claims 1/mpg scales more linearly with weight. How could you check that claim using only the plots already generated?

**Q8** · `eda.confounding`

- Cell [12] describes the positive correlation between `acceleration` and mpg as a "suppression effect." Explain in plain language why the sign is misleading.
- Which feature, when added to a regression alongside `acceleration`, would most likely reverse or weaken its coefficient?

**Q9** · `eda.sampling-bias`

- Cell [7] counts only 4 three-cylinder and 3 five-cylinder cars out of 392 total. What does this mean for per-group statistics in the mpg-by-cylinders plot?
- If you evaluated a model's error separately for each cylinder class, would you trust the 3-cylinder and 5-cylinder estimates equally to the 4-cylinder estimate? Why or why not?

---

## Questions `mpg_02_ridge_lasso.py`

**Script topics** · Data Splits · Regularized Regression · Hyperparameter Optimization

**Q1** · `pipeline.data-leakage`

- At lines 58-59, `StandardScaler` and `y_scaler` are fitted on `X_train` and `y_train` only. What would go wrong if they were fitted on the full dataset before splitting?
- Identify every place in the script where `x_scaler.transform` or `y_scaler.transform` is called and confirm each call uses the correct split.

**Q2** · `pipeline.validation`

- Lines 52-53 create a three-way split (train / val / test). Why is a dedicated validation set needed instead of a simple two-way train/test split?

**Q3** · `modeling.algorithm`

- What penalty term does Ridge add to the OLS loss? How does Lasso's penalty differ?
- Given the multicollinearity found in the EDA (r > 0.9 among three features), why is Ridge a reasonable starting choice over plain OLS?

**Q4** · `modeling.param-effect`

- `LASSO_TRAIN_FRAC = 0.50` (line 29) makes the Lasso grid train on only half the training data. What practical problem is this addressing?
- What is the trade-off: what do you gain, and what do you risk, by reducing the training fraction for one model but not the other?

**Q5** · `training.convergence`

- The error-curves plot (lines 117-126) shows both a coarse and a fine alpha grid for Ridge and Lasso. What information does the coarse grid provide that motivates the fine grid?
- If the validation curve showed no clear minimum within the searched range, what would you conclude and what would you do next?

**Q6** · `explain.coeff-meaning`

- The coefficient-paths plot (lines 129-148) shows each feature's coefficient as alpha increases. Describe how Ridge paths differ from Lasso paths in shape.
- If a Lasso coefficient reaches exactly zero, what does that mean for the feature's role in predictions?

**Q7** · `process.controlled-change`

- The final evaluation block (lines 151-164) compares OLS, Ridge, and Lasso on the test set (OLS RMSE = 3.256 mpg, Ridge RMSE = 3.314 mpg, Lasso RMSE = 3.249 mpg). How many things change between OLS and Ridge, and between Ridge and Lasso?
- Is this comparison sufficient to attribute any RMSE difference solely to regularisation? What else could contribute?

**Q8** · `modeling.param-effect`

- `ALPHA_MAX = 0.1` (line 28) caps the alpha search range fed into both `ALPHAS_COARSE` and `ALPHAS_FINE`. Predict what would happen to the validation RMSE curve if `ALPHA_MAX` were extended to 10: at which end of the x-axis would you see a change, and what shape would it take?
- How does changing the search *range* differ from changing the search *density* (number of grid points), and which problem does each address?

**Q9** · `evaluation.overfit`

- The output shows OLS train RMSE = 3.397 mpg but val RMSE = 3.050 mpg: validation error is lower than training error. Does this mean the model generalises unusually well, or is there another explanation?
- The val and test sets each contain only 79 rows. How does set size affect the reliability of a single RMSE estimate?

**Q10** · `explain.coeff-meaning`

- The output reports "Lasso zeroed 4/8 features at alpha = 0.016." The 8 features are the 6 numeric columns plus 2 origin dummies. Given the high correlations among engine-size features found in `mpg_01_eda.py`, which 4 features would you predict are zeroed?
- Lasso test RMSE (3.249 mpg) is nearly identical to OLS (3.256 mpg) despite using only 4 active features. What does this tell you about the information content of the zeroed features?

**Q11** · `process.reproducibility`

- Lines 103-104 use `rng = np.random.default_rng(42)` to subsample training rows for the Lasso grid, while lines 52-53 use `random_state=42` in `train_test_split`. These are two different random-number systems (NumPy Generator vs. legacy random state).
- Would the Lasso grid results change if only the `train_test_split` seed were changed? Would they change if only the `rng` seed were changed?
- What does using two independent seeds mean for reproducing the exact output?

---

## Questions `mpg_03_bias_variance_decisiontree.py`

**Script topics** · Underfitting and Overfitting · Decision Trees

**Q1** · `training.convergence`

- Cell [3] plots learning curves for `max_depth=2` and a fully grown tree. Describe the expected shape of the train and CV lines in each panel as training size increases.
- The high-bias panel has a small train-CV gap. Does a small gap always mean the model generalises well?

**Q2** · `training.convergence`

- Cell [4] sweeps `max_depth` from 1 to 15. At which end of the x-axis does underfitting dominate, and at which end does overfitting dominate?
- The output shows best CV RMSE = 3.754 at max_depth = 8. What criterion selects this value, and why does the train-CV gap continue to widen for depths above 8 while CV RMSE stays roughly flat?

**Q3** · `process.controlled-change`

- Cell [5] compares a single 80/20 split with 5-fold CV over 30 random seeds. What exactly varies across the 30 seeds in each procedure?
- The output shows single-split std = 0.296 vs. CV std = 0.024 across 30 seeds (a ~12x difference). What accounts for this difference, and what does it imply for using a single holdout split on a 392-row dataset?

**Q4** · `training.convergence`

- The high-variance panel in cell [3] prints train RMSE as `-0.000` at every training size. Explain why a decision tree with `max_depth=None` produces this value and whether it represents a numerical error.
- Knowing that train RMSE = 0 regardless of training size, what is the only useful signal left in the high-variance panel's chart?

**Q5** · `process.controlled-change`

- Cell [5] runs the stability experiment using a `LinearRegression` pipeline (line 189), not the `DecisionTreeRegressor` from cells [3]-[4]. How many things change between the decision-tree cells and this stability cell?
- Does the stability result (CV std = 0.024) apply to the decision tree at max_depth = 8, or only to the linear model? What would you need to run to get a fair comparison for the decision tree?

---

## Questions `mpg_04_bias_variance_ridge.py`

**Script topics** · Underfitting and Overfitting · Regularized Regression

**Q1** · `training.convergence`

- Cell [3] uses `RIDGE_ALPHA_HIGH=1000` for the high-bias panel and `RIDGE_ALPHA_LOW=0.01` for the low-regularisation panel. What role does alpha play that is analogous to `max_depth` in the decision-tree script?
- The cell [3] docstring notes that with n/p ~65 the low-reg panel shows "the well-specified but limited baseline rather than classic high-variance." What does that mean, and how does it differ from the fully grown tree panel in the other script?

**Q2** · `training.convergence`

- Cell [4] sweeps alpha on a log scale. The docstring says "the U-shape is mirrored." Explain why the validation RMSE degrades on the right (large alpha) rather than the left (small alpha).
- The output shows best CV RMSE = 3.859 at alpha = 0.01, the smallest value in the sweep. What does landing at the left boundary suggest about how much regularisation this dataset needs, and what would you do to confirm this?

**Q3** · `process.controlled-change`

- Comparing this script with `mpg_03_bias_variance_decisiontree.py`: the overall structure (cells [2]-[5]) is nearly identical. What is the single design choice that differs?
- What does this structural similarity reveal about the purpose of both scripts?

**Q4** · `evaluation.overfit`

- The CV stability output (cell [5]) for this Ridge script (alpha = 0.01) is numerically identical to `mpg_03`'s stability output: single-split std = 0.296, CV std = 0.024, over the same 30 seeds.
- Does this mean Ridge and LinearRegression perform identically, or is there a simpler explanation?
- What would you change in the experiment to determine whether stability is driven by dataset size or model choice?
