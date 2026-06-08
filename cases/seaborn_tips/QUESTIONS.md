# Code Reading Questions for seaborn_tips

- [tips_01_eda.py](#questions-tips_01_edapy)
- [tips_02_single_feature.py](#questions-tips_02_single_featurepy)
- [tips_03_all_features.py](#questions-tips_03_all_featurespy)
- [tips_04_kmeans_segmentation.py](#questions-tips_04_kmeans_segmentationpy)

## Questions `tips_01_eda.py`

**Script topics** · EDA: Descriptive Statistics · EDA: Distributions · EDA: Correlations · Business Understanding


**Q1** · `business.objective-alignment`

- Cell [2] shows a scatter of raw `tip` vs `total_bill` with an OLS regression line (slope ≈ 0.105). What does this slope imply about the "typical" tip rate?
- Why might the script then engineer `tip_pct` (line 25) rather than keeping raw `tip` as the variable of interest?

**Q2** · `business.decision-readiness`

- Cell [4] shows side-by-side box plots (sex, smoker, day, time). Which predictor shows the least difference in `tip_pct` median across its groups?
- Does a small median difference imply that the predictor should be excluded from a model?

**Q3** · `eda.plot-assumption`

- Cells [4] and [5] use box plots with overlaid strip plots. Why could strip plots be useful here? What do the strips reveal that the box plot alone cannot show?
- As a side: Is there anything you would improve in the visual appearance of the two plots?

**Q4** · `eda.confounding`

- Cell [5] plots `total_bill` by day and by time. Which days appear to have higher `tip_pct` (visible in Cell [4] output)?
- Could bill size explain this day effect?

**Q5** · `eda.confounding`

- Cell [6] shows a smoker × sex interaction heatmap of mean `tip_pct`. What would it mean if the rows were not monotone across columns (aka showing the same "trend")?
- Does the actual heatmap show a visible interaction?

**Q6** · `eda.confounding`

- Cell [7] shows `tip_pct` vs party `size` coloured by sex (scatter, left panel) and `tip_pct` by sex within each size level (box plots, right panel). Does the sex gap in `tip_pct` persist within each size level, or does it vanish once size is held constant?

---

## Questions `tips_02_single_feature.py`

**Script topics** · Linear Regression · Gradient Descent · Data Splits · Scaling and Imputation

**Q1** · `pipeline.data-leakage`

- Cell [4] computes `mu` and `sigma` from `X_train` only. What would happen if you used the full `X` instead?
- Would the output numbers change noticeably?
- Would the test score be trustworthy?

**Q2** · `transform.trace-pred`

- Cell [4]: `X_train_b = add_bias(X_train_s)`. What is the shape of `X_train_b`?
- Why 2 columns when `total_bill` is a single number?
- Hint: Trace the data through `reshape(-1,1)`, standardise, and `add_bias`.

**Q3** · `transform.trace-pred`

- Before executing cells [6] and [8]: Will OLS and gradient descent produce the same or different intercepts? Why? Under what condition would they diverge?

**Q4** · `modeling.algorithm`

- Cell [7] implements an optimizer from scratch. What is it called (_yes, a simple one_)?
- What quantity does `grad`represent mathematically?
- Where exactly in the loop is that quantity computed?

**Q5** · `training.train-dynamics`

- Still cell [7]: The learning rate `alpha=0.1` is passed. Predict what the loss curve would look like if you use alpha values `0.01` or `0.99` instead. How would you distinguish "learning rate too high" from "model not converging" in the convergence plot from cell [8]?

**Q6** · `training.convergence`

- Cell [8] produces a convergence plot with a red dashed horizontal line. What does that line represent?
- Here, we use only training data. What's missing?

**Q7** · `extension.alternatives`

- `tips_02_single_feature.py` trains on a single feature (`total_bill`). What would you need to change to train on two features instead?
- Would the gradient descent loop change? What about the OLS solution?

---

## Questions `tips_03_all_features.py`

**Script topics** · Structural Cleaning and Encoding · Scaling and Imputation · Data Splits · Feature Engineering

**Q1** · `deployment.input-assump`

- Cell [2] calls `pd.get_dummies` on the raw DataFrame. What does this silently assume about the categorical columns present in the data? 
- What would happen at prediction time if the test set contained a category not seen during training?

**Q2** · `transform.feature-engineering`

- Still cell [2] `pd.get_dummies(..., drop_first=True)`. What does `drop_first` do?
- What are the consequences of removing it for the OLS solution?

**Q3** · `transform.trace-pred`

- Cell [3]: `X.values` converts the DataFrame to a NumPy array before the split. What information is lost at that point?

**Q4** · `process.reproducibility`

- Cell [3] passes `random_state=42` to `train_test_split`. If this wasn't there which outputs in the script would change across runs, and which would stay the same?
- Why do scripts that report model quality need to be fully deterministic?
- What would be a better place to define the random state?

**Q5** · `business.objective-alignment`

- The model minimizes MSE (implicitly via `LinearRegression` in Cell [4]). Is that the right loss for predicting tip amounts?
- What metric/loss would you use if the goal were to flag outlier tippers?

**Q6** · `evaluation.overfit`

- Cell [4] reports train R² and test R². If test R² ≈ train R², does that mean the model is not overfitting?
- What else about this specific script could explain a small train/test gap?

**Q7** · `deployment.dist-shift`

- The training data covers one restaurant. If the model were applied to a different restaurant, which features are most likely to behave differently?
- How would you detect that the model's performance had degraded after deployment?

**Q8** · `business.objective-alignment`

- Is predicting `tip` the right question? Or might there be a more interesting target variable?
- Hint: Notice the commented out feature-engineering line in cell [2].
- If you swap out the target variable, what to you expect to change about results?  

---

## Questions `tips_04_kmeans_segmentation.py`

**Script topics** · k-Means Clustering · Scaling and Imputation · EDA: Descriptive Statistics

**Q1** · `transform.trace-pred`

- Cell [2] calls `StandardScaler()`. The output confirms means ≈ 0 and stds ≈ 1 for all three features.
- If you skipped scaling and ran K-means on raw values (`total_bill` ranging ~3–51, `tip_pct` ranging 3.6–71, `size` ranging 1–6), which feature would dominate the Euclidean distances, and why?

**Q2** · `modeling.param-effect`

- Cell [3] evaluates k = 2 to 8 using both inertia (elbow) and silhouette score. The output shows peak silhouette at k=2 (score=0.466).
- What does a silhouette score of 0.466 tell you about cluster separation? Would you describe this as weak, moderate, or strong?
- _Bonus question_: Silhouette peaked at k=2. Could there be a bias of silhouette toward low k (hint: How do two large, well-separated blobs score?) Since ellbow suggested k=4, what k might be a better choice than what the script does?

**Q3** · `process.reproducibility`

- Cells [3] and [4] both pass `random_state=RANDOM_STATE` and `n_init=10` to `KMeans`. If the seed were removed, which outputs in the script would change across runs, and which are deterministic regardless of seed?
- What specific failure mode does `n_init=10` protect against, and what does it not protect against?

**Q4** · `transform.trace-pred`

- Cell [5] calls `scaler.inverse_transform(km_final.cluster_centers_)` before printing the segment profiles. What values would you see if you reported the raw scaled centers directly?
- Why are the original-unit profiles (total_bill in $, tip_pct in %) more useful for interpretation than the scaled version?

**Q5** · `eda.confounding`

- Cell [7] computes crosstabs of segment vs. sex, smoker, time, and day — none of which were used during clustering. The output shows both segments are male-dominant and non-smoker-dominant.
- What does demographic uniformity across segments suggest about the relationship between demographics and spending behaviour in this dataset?
- Is this evidence that demographics do not matter, or could a sampling explanation account for it?

**Q6** · `evaluation.metrics` (Bonus)

- Cell [8] produces a per-sample silhouette plot. The dashed red line marks the mean (0.466), the same scalar used in cell [3] to select k=2.
- What would a silhouette value of exactly 0 mean for an individual data point?
- What does it indicate if most bars in one segment fall below the mean line, while the other segment is mostly above it?
