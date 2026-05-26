# `sklearn_wine`: Wine Cultivar Classification

> Chemical profiles of 178 wines from three Italian cultivars, where the challenge is knowing *when* and *why* to scale your features.

## Contents

- [Domain Context](#domain-context)
- [Online Resources](#online-resources)
- [Codebook](#codebook)
- [What You Can Learn Here](#what-you-can-learn-here)
- [Research Questions](#research-questions)
- [Available Scripts](#available-scripts)

## Domain Context

Each row represents one wine sample, described by 13 physicochemical measurements taken from grapes grown in the same region of Italy but from three different cultivars. In practice, enologists and food scientists use these profiles to authenticate wine origin and detect adulteration. The dataset's teaching value lies in its extreme scale differences between features (magnesium ~100, alcohol ~13) and the tight chemical correlations within the phenol-related feature group.

## Online Resources

- **Dataset**: [UCI ML Repository - Wine](https://archive.uci.edu/dataset/109/wine)
- **Documentation**: [sklearn.datasets.load_wine](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_wine.html)

## Codebook

| Column | Type | Range / Values | Description |
|--------|------|----------------|-------------|
| alcohol | float64 | 11.03-14.83 | Alcohol content (% vol) |
| malic_acid | float64 | 0.74-5.80 | Malic acid concentration |
| ash | float64 | 1.36-3.23 | Ash content |
| alcalinity_of_ash | float64 | 10.6-30.0 | Alkalinity of ash |
| magnesium | float64 | 70-162 | Magnesium content (mg/L) |
| total_phenols | float64 | -- | Total polyphenol content |
| flavanoids | float64 | -- | Flavanoid phenol content |
| nonflavanoid_phenols | float64 | -- | Non-flavanoid phenols |
| proanthocyanins | float64 | -- | Proanthocyanin content |
| color_intensity | float64 | 1.3-13.0 | Color intensity |
| hue | float64 | -- | Hue ratio |
| od280/od315_of_diluted_wines | float64 | -- | OD280/OD315 absorbance ratio |
| proline | float64 | -- | Proline amino acid content |

**Target**: `target` (class_0 / class_1 / class_2): mildly imbalanced (59 / 71 / 48). Labels encode cultivar identity but are not named in the bundled dataset.

## What You Can Learn Here

- Why feature scaling is required for distance-based models (KNN, SVM): raw features span orders of magnitude
- Multi-collinearity: phenol-related features are strongly intercorrelated, making LDA and random forests behave differently
- Validating unsupervised clusters against known ground-truth labels (ARI, confusion matrix)
- Class imbalance evaluation: weighted F1 vs. accuracy when class sizes differ

## Research Questions

**EDA**
1. Which feature has the widest coefficient of variation across all three cultivars?
2. Which pairs of features are most correlated? Does this make chemical sense?
3. Are the outliers in malic_acid and color_intensity the same samples?

**Modeling**
1. Compare KNN accuracy with and without StandardScaler: what difference does scaling make and which cultivar is most affected?
2. Apply LDA: how many components are needed for near-perfect separation, and which features load most heavily?
3. Train a random forest; rank feature importances and check whether they align with the LDA loadings.

---

## Available Scripts

- `wine_01_eda.py`: distributions, scale spread, correlation matrix, outlier investigation
- `wine_02_kmeans.py`: k-means clustering, k selection (elbow + silhouette), ARI evaluation, centroid profiles, scaling ablation

**General intent of scripts:** clustering (kmeans)

See [QUESTIONS.md](QUESTIONS.md) for per-script code-reading questions.

## Disclaimer

Part of the dataset description above was compiled by AI. Check any assumptions, claims, and context.
