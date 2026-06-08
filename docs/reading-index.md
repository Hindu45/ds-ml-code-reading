# Reading Index

_A guide for code-reading sessions: when to read which scripts in which order and why._

A few principles that guide the reading progression:
- Sessions follow a dataset/problem arc rather than clustering by method. Methods appear in the context of the problem they solve, which anchors understanding and trains the ability to transfer to unfamiliar techniques.
- Concepts sometimes appear before they are formally introduced in lectures. This is intentional: encountering a technique in context first, then revisiting it in the lecture or further contexts, deepens retention.
- Scripts may be  revisited across sessions as different aspects become relevant. Repetition is a key learning feature.

Question types reference the [Question Catalogue](question-catalogue.md).

## Contents
Suggested dataset sequence:
1. [Tips: Restaurant Tipping Behavior](#tips-restaurant-tipping-behavior)
2. [Penguins: Palmer Penguins Morphology](#penguins-palmer-penguins-morphology)
3. [MPG: Automobile Fuel Efficiency](#mpg-automobile-fuel-efficiency)

The questions for datasets not listed here are not yet finalized.

Additionally: [Topic Reference](#topic-reference)

---

## Suggested Dataset Sequence

### Tips: Restaurant Tipping Behavior

*A single waiter's 244 bills:  compact, interpretable, and rich enough to carry us from EDA through regression to clustering.*

Questions for this dataset are in [cases/seaborn_tips/QUESTIONS.md](../cases/seaborn_tips/QUESTIONS.md).

- [tips_01_eda.py](../cases/seaborn_tips/tips_01_eda.py): tip_pct distribution, categorical comparison, confounding, smoker-sex interaction
- [tips_02_single_feature.py](../cases/seaborn_tips/tips_02_single_feature.py): linear regression, gradient descent from scratch, OLS normal equation, data leakage, scaling
- [tips_03_all_features.py](../cases/seaborn_tips/tips_03_all_features.py): multiple regression, one-hot encoding, permutation importance, partial regression, deployment assumptions
- [tips_04_kmeans_segmentation.py](../cases/seaborn_tips/tips_04_kmeans_segmentation.py): k-means clustering, elbow method, silhouette score, segment profiling, demographic crosstab

---

### Penguins: Palmer Penguins Morphology

*Morphological measurements for 344 penguins across three species: a modern, richer alternative to the Iris dataset that carries classification from a simple baseline through trees, with real confounds along the way.*

Questions for this dataset are in [cases/seaborn_penguins/QUESTIONS.md](../cases/seaborn_penguins/QUESTIONS.md).

- [penguins_01_eda.py](../cases/seaborn_penguins/penguins_01_eda.py): distributions, correlation, Simpson's paradox, species-island confounding
- [penguins_02_modeling_baseline.py](../cases/seaborn_penguins/penguins_02_modeling_baseline.py): logistic regression, incremental features, macro F1, confusion matrix
- [penguins_03_modeling_trees.py](../cases/seaborn_penguins/penguins_03_modeling_trees.py): decision tree, depth sweep, tree visualisation, feature importance, random forest

---

### MPG: Automobile Fuel Efficiency

*392 cars from 1970–1982: multicollinear engine specs and a strong time trend make this a natural case for regularisation, and the moderate dataset size keeps bias-variance diagnostics clean and readable.*

Questions for this dataset are in [cases/seaborn_mpg/QUESTIONS.md](../cases/seaborn_mpg/QUESTIONS.md).

- [mpg_01_eda.py](../cases/seaborn_mpg/mpg_01_eda.py): numeric distributions, multicollinearity, time trend, suppressor effect, horsepower outliers
- [mpg_02_ridge_lasso.py](../cases/seaborn_mpg/mpg_02_ridge_lasso.py): Ridge and Lasso, three-way split, coefficient paths, alpha selection, leakage-safe scaling
- [mpg_03_bias_variance_decisiontree.py](../cases/seaborn_mpg/mpg_03_bias_variance_decisiontree.py): learning curves, validation curve, depth as complexity dial, CV vs single-split stability
- [mpg_04_bias_variance_ridge.py](../cases/seaborn_mpg/mpg_04_bias_variance_ridge.py): learning curves, validation curve, alpha as inverse complexity dial, comparison with decision tree

---

## Topic Reference

Quick lookup by course topics.

### Exploratory Data Analysis (EDA)
*Course alignment: Part III · Data Understanding*

- [tips_01_eda.py](../cases/seaborn_tips/tips_01_eda.py): engineered target, categorical breakdown
- [mpg_01_eda.py](../cases/seaborn_mpg/mpg_01_eda.py): numeric distributions, suppressor effect
- [penguins_01_eda.py](../cases/seaborn_penguins/penguins_01_eda.py): cluster structure, species confound

### Data Preparation
*Course alignment: Part IV · Data Preparation*

- [tips_02_single_feature.py](../cases/seaborn_tips/tips_02_single_feature.py): leakage-safe scaling
- [penguins_02_modeling_baseline.py](../cases/seaborn_penguins/penguins_02_modeling_baseline.py): missing value choices

### Linear Regression + Gradient Descent
*Course alignment: Part V · Linear Regression, Gradient Descent*

- [tips_02_single_feature.py](../cases/seaborn_tips/tips_02_single_feature.py): GD from scratch
- [healthexp_01_linreg.py](../cases/seaborn_healthexp/healthexp_01_linreg.py): OLS with fixed effects

### Overfitting and Bias-Variance
*Course alignment: Part V · Underfitting and Overfitting*

- [mpg_03_bias_variance_decisiontree.py](../cases/seaborn_mpg/mpg_03_bias_variance_decisiontree.py): depth as complexity dial
- [mpg_04_bias_variance_ridge.py](../cases/seaborn_mpg/mpg_04_bias_variance_ridge.py): alpha as inverse complexity dial
- [penguins_03_modeling_trees.py](../cases/seaborn_penguins/penguins_03_modeling_trees.py): visible train/test gap

### Regularized Regression
*Course alignment: Part V · Regularized Regression*

- [tips_03_all_features.py](../cases/seaborn_tips/tips_03_all_features.py): encoding + OLS baseline
- [mpg_02_ridge_lasso.py](../cases/seaborn_mpg/mpg_02_ridge_lasso.py): high-multicollinearity case
- [diamonds_02_ridge_lasso.py](../cases/seaborn_diamonds/diamonds_02_ridge_lasso.py): large-scale comparison

### Classification, Decision Trees, Random Forests
*Course alignment: Part V · Classification Tasks, Decision Trees, Random Forests*

- [penguins_02_modeling_baseline.py](../cases/seaborn_penguins/penguins_02_modeling_baseline.py): logistic baseline
- [penguins_03_modeling_trees.py](../cases/seaborn_penguins/penguins_03_modeling_trees.py): depth sweep
- TBD: multi-model comparison

### Principles: Generalization, Baselines, Metrics
*Course alignment: Part VI · Principles That Transfer*

- [penguins_02_modeling_baseline.py](../cases/seaborn_penguins/penguins_02_modeling_baseline.py): baseline + bug hunt

### Anomaly Detection / Unsupervised Learning
*Course alignment: Part VII · Framing Unsupervised Learning, Isolation Forests*

- [planets_01_eda.py](../cases/seaborn_planets/planets_01_eda.py): MNAR, extreme skew
- [planets_02_isolation_forest.py](../cases/seaborn_planets/planets_02_isolation_forest.py): anomaly scoring
- [sklearn_kddcup99/02_isolation_forest.py](../cases/sklearn_kddcup99/02_isolation_forest.py): imbalanced real-world case

### Deployment and Production
*Course alignment: Part IX · Closing the Loop*

- [penguins_02_modeling_baseline.py](../cases/seaborn_penguins/penguins_02_modeling_baseline.py): research-to-field gap
- [tips_03_all_features.py](../cases/seaborn_tips/tips_03_all_features.py): single-source distribution shift
