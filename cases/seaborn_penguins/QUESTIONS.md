# Code Reading Questions for seaborn_penguins

- [penguins_01_eda.py](#questions-penguins_01_edapy)
- [penguins_02_modeling_baseline.py](#questions-penguins_02_modeling_baselinepy)
- [penguins_03_modeling_trees.py](#questions-penguins_03_modeling_treespy)

## Questions `penguins_01_eda.py`

**Script topics** · EDA: Distributions · EDA: Correlations · EDA: Data Quality

**Q1** · `eda.confounding` · `deployment.input-assump`

- Cell [4] plots species counts per island. The cell docstring flags a confounding risk.
- Chinstrap appears only on Dream; Gentoo only on Biscoe. If `island` were added as a model feature, what information would it duplicate?
- Under what deployment condition would including `island` cause a silent error?

**Q2** · `eda.general`

- Cell [6] shows body mass histograms split by sex within each species.
- Which species shows the largest male-female mass gap in these histograms?
- How does the bimodal-like shape in the marginal body mass distribution (all penguins together) arise from the per-species plots shown here?

**Q3** · `eda.confounding`

- Cell [8] plots `bill_length_mm` vs `bill_depth_mm` twice: pooled (left panel, negative slope) and per species (right panel, positive slopes for all three).
- Name the unmeasured grouping variable that causes the sign flip.
- What would happen to the regression coefficient of `bill_length_mm` if you fit `bill_depth ~ bill_length` without including the grouping variable you just names as a covariate?

**Q4** · `process.controlled-change`

- Still cell [8]: What is the single change applied between the two panels (one change at a time)?

**Q5** · `eda.general`

- Cell [9] shows a scatter of `flipper_length_mm` vs `bill_length_mm` coloured by species.
- How many distinct clusters are visible, and which pair of species overlaps most?
- Name one feature not shown in this plot that might better separate the overlapping pair, and cite the cell from the script where the evidence appears.

---

## Questions `penguins_02_modeling_baseline.py`

**Script topics** · Classification Tasks · Classification Evaluation · Data Splits · Baselines

**Q1** · `evaluation.metrics` · `business.objective-alignment`

- Cell [1] declares macro F1 as the comparison criterion for the five configurations (comment above `CONFIGS`).
- Why macro F1 rather than plain accuracy for a three-class problem with unequal class frequencies?
- What might a stakeholder who only cares about detecting Gentoo penguins want instead, and why?

**Q2** · `pipeline.validation`

- Cell [1] wraps `StandardScaler` and `LogisticRegression` in a `Pipeline` for every `LogReg` config.
- When `model.fit(X_train[features], y_train)` is called on the Pipeline, which sklearn method does it invoke on `StandardScaler`: `fit`, `transform`, or `fit_transform`? What about when `model.predict(X_test[features])` is called?
- One entry in `CONFIGS` does *not* wrap its model in a `Pipeline`. Why is omitting the scaling for that specific model class?

**Q3** · `extension.follow-up`

- Cell [1]: How would you add a sixth configuration using only `["bill_depth_mm", "body_mass_g"]`?
- Are there any other changes needed or will everything downstream simply work? What's the value of this coding pattern?

**Q4** · `deployment.input-assump`

- Cell [2] calls `df.dropna(subset=NUMERIC_COLS)` before the train/test split.
- What does this step silently assume about missing values in a live prediction system?
- What would happen if `flipper_length_mm` were missing systematically for data from a new penguins subtype that you want to use for model training?

**Q5** · `pipeline.data-leakage` · `pipeline.validation`

- Cell [3] calls `model.fit(X_train[features], y_train)` then `model.predict(X_val[features])`.
- If you replaced `X_train[features]` with `X[features]` (the full dataset) on the fit line, what would go wrong?
- Would the reported macro F1 scores look better, worse, or roughly the same? Would they be trustworthy?

**Q6** · `transform.trace-pred`

- Cell [1] defines the `"LogReg -- +flipper"` config with `features = ["bill_length_mm", "flipper_length_mm"]`.
- What is the shape of the array that `StandardScaler` outputs when called on `X_train[features]`?
- What shape does `LogisticRegression` receive at fit time, and how many coefficients (including the intercept) will it learn?

**Q7** · `deployment.prod-fit`

- Cell [2] splits a curated research dataset collected on three specific islands.
- Name two assumptions the trained model makes that might not hold if deployed as a field species-identification tool. Explain why.
- _Hint_: What kinds of data distribution shifts could occur?

**Q8** · `process.controlled-change`

- Cell [1]'s `CONFIGS` list adds one feature at a time while keeping the model class, split, and random state identical across all five configurations.
- What makes this a valid controlled comparison of feature value?
- Which single feature addition produces the largest jump in macro F1? Look at cell [3] output values.

---

## Questions `penguins_03_modeling_trees.py`

**Script topics** · Decision Trees · Decision Boundaries · Model Interpretability · Random Forests · Classification Evaluation

**Q1** · `pipeline.errors`

- Cell [2] uses `X = df[NUMERIC_COLS].values`, dropping column names immediately.
- Compare this to cell [2] of `penguins_02_modeling_baseline.py`, where `X = df[NUMERIC_COLS]` keeps the DataFrame intact.
- What information is preserved in script 02 that is lost in script 03?
- Could that missing information cause a silent error anywhere in script 03?

**Q2** · `transform.trace-pred`

- Cell [3] trains a depth-2 tree on only `flipper_length_mm` and `bill_length_mm`.
- Read the printed rules alongside the boundary-sequence plot: Trace and describe the splits in plain language. Reflect what these rules mean in biological terms. 
- Trace a penguin with `flipper_length_mm = 200` and `bill_length_mm = 44` through every rule in the printed output. Which leaf does it reach, and which species is predicted?

**Q3** · `process.simplicity`
- The 2-feature, depth-2 tree from cell [3] already achieves val accuracy 0.956.
- What does that tell you about the separability of the three species in this 2-D projection alone? What does that tell you about the dataset in general?

**Q4** · `modeling.algorithm`

- Cells [3], [4] instantiate `DecisionTreeClassifier`.
- What quantity does a decision tree minimize at each split?
- Is that computation visible anywhere in these lines, or does it happen entirely inside sklearn?

**Q5** · `process.controlled-change`

- Cell [4] varies `max_depth` from 1 to 8 while holding the feature set, train/val/test split, and random state constant.
- What changes between any two consecutive depth steps in this sweep?
- What does this design let you conclude that a single fixed-depth evaluation could not?

**Q6** · `transform.trace-pred`

- Cell [4] plots train accuracy and val accuracy against `max_depth` 1--8.
- The output shows both train and val accuracy reach perfect score 1.0 at depth 5. What does this tell you about the penguins dataset compared to a typical overfitting scenario?

**Q7** · `explain.importance`

- Cell [6] shows `body_mass_g` importance = 0.028 despite `body_mass_g` being strongly correlated with `flipper_length_mm` (importance 0.544).
- Why does the tree (the one from cell [5]) assign near-zero importance to `body_mass_g`?
- If you would remove `flipper_length_mm` from the feature set and retrained, what would you expect to happen to `body_mass_g` importance, and why?

**Q8** · `pipeline.data-leakage`

- Cell [4] selects `best_depth` using `val_accs`; the test set is not touched until cell [8].
- Why must depth selection use the val set rather than the test set?
- If you had used test accuracy to pick `best_depth` and then reported that same test accuracy in cell [8], what would be wrong with the reported number?

**Q9** · `extension.alternatives`

- Cell [8] reports DecisionTree test accuracy 0.971 vs RandomForest 0.986: A small gain of +0.014.
- Which model would you prefer and why?