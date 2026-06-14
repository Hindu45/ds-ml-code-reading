# Code Reading Questions for seaborn_mpg

- [mpg_01_eda.py](#questions-mpg_01_edapy)
- [mpg_02_ridge_lasso.py](#questions-mpg_02_ridge_lassopy)
- [mpg_03_bias_variance_ridge.py](#questions-mpg_03_bias_variance_ridgepy)
- [mpg_04_bias_variance_decisiontree.py](#questions-mpg_04_bias_variance_decisiontreepy)

## Questions `mpg_01_eda.py`

**Script topics** · EDA: Data Quality · EDA: Distributions · EDA: Correlations

**Q1** · `eda.data-sampling`

- According to cell [2], what years and origins does this dataset cover?
- Which combination of origin and cylinder count is likely over-represented?
- How might this sampling affect a model trained to predict mpg for modern cars?

**Q2** · `transform.trade-pred`

- In cell [3], the six rows with missing horsepower are dropped. What assumption does this embed about why the values are missing?
- What category of missingness would you assume (MCAR, MAR, MNAR), and in what sense does it matter for the drop-vs-impute decision?

**Q3** · `eda.general`

- Cell [4] plots raw mpg and the inverse 1/mpg side by side. Describe the shape of each distribution.
- Which would be the better modeling target for ordinary least-squares regression, and why?
- The script chooses to model mpg directly. What trade-off does that decision introduce?

**Q4** · `eda.confounding`

- Cell [5] shows fleet mpg rising sharply after 1973 and 1979. Is `model_year` a genuine predictor or a proxy for unmeasured regulatory and design-era effects?
- If both `model_year` and engine-size features are included in the same regression, what does the `model_year` coefficient actually estimate?

**Q5** · `eda.confounding`

- Cell [6] shows median mpg of 31.6 (japan), 26.0 (europe), and 18.5 (usa). The docstring says origin is "partly a proxy for cylinders/displacement."
- What would happen to the estimated coefficient of `origin` if `cylinders` and `displacement` were also in the model?

**Q6** · `eda.data-sampling`

- Cell [7] counts only 4 three-cylinder and 3 five-cylinder cars out of 392 total. What does this mean for per-cyclinder-group statistics in the mpg dataset?
- If you evaluated a model's error separately for each cylinder class, would you trust the 3-cylinder and 5-cylinder estimates equally to the 4-cylinder estimate? Why or why not?

**Q7** · `eda.general` · `explain.importance`

- Cell [8] ranks weight (-0.832), displacement (-0.805), horsepower and cylinders (both -0.778) as the strongest correlators with mpg.
- Cell [9] shows these same four features are intercorrelated at r > 0.9. Would you expect all four to add independent predictive power, or mostly redundant information?

**Q8** · `eda.general`

- In cell [9]: Why does high multicollinearity of features inflate OLS coefficient variance?
- What does this imply for the choice of model?

**Q9** · `eda.general`

- Cell [10] shows a curved, not straight-line, relationship between mpg and weight. What does this imply for a linear model trained on raw weight values?
- The docstring claims 1/mpg scales more linearly with weight. Given the plot in cell [10] - do you think this is correct?

**Q10** · `eda.general`

- Cell [11] flags horsepower outliers with a z-score threshold of |z| > 2.5, which assumes the variable is roughly symmetric.
- The cell's own docstring notes horsepower has "a long right tail." Does a z-score threshold suit a skewed variable, or could it misfire on one side?

**Q11** · `eda.confounding`

- Cell [12] describes the positive correlation between `acceleration` and mpg as a "suppression effect". This means that when adding a third variable to an analysis unexpectedly strengthens or reverses the observed relationship between an independent and a dependent target variable. Explain this in your own words.
- Which feature, when added to a regression alongside `acceleration`, would most likely reverse or weaken its coefficient?

---

## Questions `mpg_02_ridge_lasso.py`

**Script topics** · Data Splits · Regularized Regression (Ridge + Lasso) · Hyperparameter Optimization

**Q1** · `transform.trace-pred`

- `pd.get_dummies(..., drop_first=True)` keeps `origin_japan` and `origin_usa` but drops `origin_europe`. What does that imply for reading the two dummy coefficients?

**Q2** · `pipeline.validation`

- Cell [3] create a three-way split (train / val / test). Why is a dedicated validation set needed instead of a simple two-way train/test split?

**Q3** · `pipeline.data-leakage`

- In cell [4], `StandardScaler` is fitted on `X_train` and `y_train` only. What would go wrong if they were fitted on the full dataset before splitting?

**Q4** · `evaluation.overfit`

- The output of cell [5] shows OLS train RMSE = 3.397 mpg but val RMSE = 3.050 mpg: validation error is lower than training error. Does this mean the model generalizes unusually well, or is there another explanation?
- The val and test sets each contain only 79 rows (see cell [3]). How does set size affect the reliability of a single RMSE estimate?

**Q5** · `modeling.algorithm`

- Cell [6]: What penalty term does Ridge add to the OLS loss? How does Lasso's penalty differ?
- Given the multicollinearity found in the EDA script (r > 0.9 among three features), why is Ridge a reasonable starting choice over plain OLS?

**Q6** · `modeling.param-effect`

- `ALPHA_MAX = 0.1` defined in cell [0] caps the alpha search range, which is fed into both `ALPHAS_COARSE` and `ALPHAS_FINE`. Would it make sense to extend `ALPHA_MAX` to 1.0 to consider the full range of trade-offs between loss functions and regularization?
- How does changing the search *range* differ from changing the search *density* (number of grid points), and which problem does each address?

**Q7** · `pipeline.trace-pred` · `pipeline.error`

- Cell [8]: The error-curves plot uses RMSE error instead of the loss function that  includes MSE + the regularization term. Why?
- Observe that validation errors are consistently *below* train errors. Why can that be and what does that mean?
- Might there be a small-sample split artifact involved (see the output of cell [3])? _Note: Results from script 3 will help clarify this further_

**Q8** · `training.convergence`

- Cell [8]: The error-curves plot shows both a coarse and a fine alpha grid for Ridge and Lasso. What information does the coarse grid provide that motivates the fine grid?

**Q9** · `explain.coeff-meaning`

- Cell [9]: The coefficient-paths plot shows each feature's coefficient as alpha increases. Describe how Ridge paths differ from Lasso paths in shape.
- If a Lasso coefficient reaches exactly zero, what does that mean for the feature's role in predictions?

**Q10** · `explain.coeff-meaning`

- Cell [10]: The output reports "Lasso zeroed 4/8 features at alpha = 0.014." Given the high correlations among engine-size features found in `mpg_01_eda.py`, which 4 features would you predict are zeroed?
- Lasso test RMSE (3.260 mpg) is nearly identical to OLS (3.256 mpg) despite using only 4 active features. What does this tell you about the information content of the zeroed features?

**Q11** · `evaluation.metrics`

- Cell [11] plots predicted vs. actual mpg for Ridge and Lasso.
- What can this scatter reveal about model behaviour, e.g. systematic over- or under-prediction, that a single RMSE number can't?

## Questions `mpg_03_bias_variance_ridge.py`

**Script topics** · Underfitting and Overfitting · Regularized Regression

**Q1** · `evaluation.baseline`

- Cell [2] prints the target's mean (23.45) and std (7.80).
- A baselines that always predicts the mean would score RMSE ≈ 7.8. How does that compare to the tuned CV RMSE of ≈3.86 from cell [4]?

**Q2** · `training.convergence`

- Cell [3] compares learning curves for α=1000 (high bias) vs. α=0.01 (low regularisation). Both end with a gap of ~0.6-0.7 mpg at the largest training size.
- What do these learning curves tell you about the model? Would more data still help?

**Q3** · `training.convergence`

- Cell [4]: Explain why increasing alpha causes CV RMSE to degrade on the right side of the plot.
- Is their a mathematical argument that training error is always rising with regularization?

**Q4** · `evaluation.overfit`

- Cell [5] compares single-split vs. CV.
- Revisit Q7 from `mpg_02_ridge_lasso.py`: a single split there showed validation RMSE running below training RMSE. Does the ~12x gap between single-split std (0.296) and CV std (0.024) shown here support the "small-sample split artifact" explanation for that pattern, or point to a different cause?
- Also consider cell [4] for your answer, where CV RMSE is clearly shown to be higher than train RMSE throughout.

---

## Questions `mpg_04_bias_variance_decisiontree.py`

**Script topics** · Underfitting and Overfitting · Decision Trees

**Q1** · `process.controlled-change`

- Compare this script's structure with `mpg_03_bias_variance_ridge.py` (cells [2]-[5] and the config block in cell [1]). What is the single design choice that differs between them?
- What does this structural similarity reveal about the purpose of both scripts?

**Q2** · `quality.abstraction`

- Cell [2]'s docstring says swapping the dataset means swapping this cell. Cell [1]'s config block lists "diamonds," "taxis," "healthexp" as alternatives.
- If you switched to `healthexp` (`FEATURES=["Year","Spending_USD"]`, `TARGET="Life_Expectancy"`), would anything outside cells [1]-[2] need to change?

**Q3** · `training.convergence`

- Cell [4]: The output here shows best max_depth=8 for decision trees which is an interior point, while `mpg_03` script showed best alpha=0.01 to be the smallest alpha in the sweep. What does each boundary result tell you about how sensitive this dataset is to each complexity dial?

_**Note:** To compare plots from both scripts, after running both once, you can check out the plots folder where the plots have been saved._

**Q4** · `process.controlled-change` · `modeling.algorithm`

- Cell [5]: Here you see the the *identical* stability experiment (30 seeds, single 80/20 split vs. 5-fold CV) on the same data. What's the one thing that differs between the two cell [5]s from `mpg_03` and `mpg_04`, and why does that make the comparison fair?
- Ridge's CV std is ~12x smaller than its single-split std. The tree's CV std is only ~2.7x smaller than its single-split std. Why does averaging across folds stabilize the evaluation so much more for Ridge than for the tree?
- Connect this to cell [3]'s diagnosis of the fully-grown tree as "high variance": Might the same sensitivity to which rows land in the training set be showing up here too?