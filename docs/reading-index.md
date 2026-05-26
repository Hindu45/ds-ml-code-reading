# Reading Index

_A guide for code-reading sessions: when to read which scripts in which order and why._

A few principles that guide the reading progression:
- Sessions follow a dataset/problem arc rather than clustering by method. Methods appear in the context of the problem they solve, which anchors understanding and trains the ability to transfer to unfamiliar techniques.
- Concepts sometimes appear before they are formally introduced in lectures. This is intentional: encountering a technique in context first, then revisiting it in the lecture or further contexts, deepens retention.
- Scripts may be  revisited across sessions as different aspects become relevant. Repetition is a key learning feature.

Question types reference the [Question Catalogue](question-catalogue.md).

**Contents**
- [Tips: Restaurant Tipping Behavior](#tips-restaurant-tipping-behavior)
- [Topic Reference](#topic-reference)

---


## Tips: Restaurant Tipping Behavior

*A single waiter's 244 bills:  compact, interpretable, and rich enough to carry us from EDA through regression to clustering.*

Questions for this dataset are in [cases/seaborn_tips/QUESTIONS.md](../cases/seaborn_tips/QUESTIONS.md).

- [tips_01_eda.py](../cases/seaborn_tips/tips_01_eda.py): tip_pct distribution, categorical comparison, confounding, smoker-sex interaction
- [tips_02_single_feature.py](../cases/seaborn_tips/tips_02_single_feature.py): linear regression, gradient descent from scratch, OLS normal equation, data leakage, scaling
- [tips_03_all_features.py](../cases/seaborn_tips/tips_03_all_features.py): multiple regression, one-hot encoding, permutation importance, partial regression, deployment assumptions
- [tips_04_kmeans_segmentation.py](../cases/seaborn_tips/tips_04_kmeans_segmentation.py): k-means clustering, elbow method, silhouette score, segment profiling, demographic crosstab

---

## Topic Reference

Quick lookup by course topics.

### Block A:  EDA
*Course alignment: Part III · Data Understanding*

- [tips_01_eda.py](../cases/seaborn_tips/tips_01_eda.py): engineered target, categorical breakdown
- [mpg_01_eda.py](../cases/seaborn_mpg/mpg_01_eda.py): numeric distributions, suppressor effect
- [penguins_01_eda.py](../cases/seaborn_penguins/penguins_01_eda.py): cluster structure, species confound

### Block B:  Data Preparation
*Course alignment: Part IV · Data Preparation*

- [tips_02_single_feature.py](../cases/seaborn_tips/tips_02_single_feature.py): leakage-safe scaling
- [penguins_02_modeling_baseline.py](../cases/seaborn_penguins/penguins_02_modeling_baseline.py): missing value choices

### Block C:  Linear Regression + Gradient Descent
*Course alignment: Part V · Linear Regression, Gradient Descent*

- [tips_02_single_feature.py](../cases/seaborn_tips/tips_02_single_feature.py): GD from scratch
- [healthexp_01_linreg.py](../cases/seaborn_healthexp/healthexp_01_linreg.py): OLS with fixed effects

### Block D:  Overfitting and Bias-Variance
*Course alignment: Part V · Underfitting and Overfitting*

- [bias_variance_tradeoff_decisiontree.py](../cases/seaborn_mpg/bias_variance_tradeoff_decisiontree.py): depth as complexity dial
- [bias_variance_tradeoff_ridge.py](../cases/seaborn_mpg/bias_variance_tradeoff_ridge.py): alpha as complexity dial
- [penguins_03_modeling_trees.py](../cases/seaborn_penguins/penguins_03_modeling_trees.py): visible train/test gap

### Block E:  Regularized Regression
*Course alignment: Part V · Regularized Regression*

- [tips_03_all_features.py](../cases/seaborn_tips/tips_03_all_features.py): encoding + OLS baseline
- [mpg_02_ridge_lasso_utils.py](../cases/seaborn_mpg/mpg_02_ridge_lasso_utils.py): high-multicollinearity case
- [diamonds_02_ridge_lasso.py](../cases/seaborn_diamonds/diamonds_02_ridge_lasso.py): large-scale comparison

### Block F:  Classification, Decision Trees, Random Forests
*Course alignment: Part V · Classification Tasks, Decision Trees, Random Forests*

- [penguins_02_modeling_baseline.py](../cases/seaborn_penguins/penguins_02_modeling_baseline.py): logistic baseline
- [penguins_03_modeling_trees.py](../cases/seaborn_penguins/penguins_03_modeling_trees.py): depth sweep
- TBD: multi-model comparison

### Block G:  Principles: Generalisation, Baselines, Metrics
*Course alignment: Part VI · Principles That Transfer*

- [penguins_02_modeling_baseline.py](../cases/seaborn_penguins/penguins_02_modeling_baseline.py): baseline + bug hunt

### Block H:  Anomaly Detection / Unsupervised Learning
*Course alignment: Part VII · Framing Unsupervised Learning, Isolation Forests*

- [planets_01_eda.py](../cases/seaborn_planets/planets_01_eda.py): MNAR, extreme skew
- [planets_02_isolation_forest.py](../cases/seaborn_planets/planets_02_isolation_forest.py): anomaly scoring
- [sklearn_kddcup99/02_isolation_forest.py](../cases/sklearn_kddcup99/02_isolation_forest.py): imbalanced real-world case

### Block I:  Deployment and Production
*Course alignment: Part IX · Closing the Loop*

- [penguins_02_modeling_baseline.py](../cases/seaborn_penguins/penguins_02_modeling_baseline.py): research-to-field gap
- [tips_03_all_features.py](../cases/seaborn_tips/tips_03_all_features.py): single-source distribution shift
