# `penguins`: Palmer Penguins Morphology

> Morphological measurements for 344 penguins across three species on three Antarctic islands - a modern, richer alternative to the Iris dataset for classification teaching.

## Contents

- [Domain Context](#domain-context)
- [Online Resources](#online-resources)
- [Codebook](#codebook)
- [What You Can Learn Here](#what-you-can-learn-here)
- [Research Questions](#research-questions)
- [Available Scripts](#available-scripts)

## Domain Context

Data were collected at Palmer Station, Antarctica, covering three penguin species: Adelie (152), Chinstrap (68), and Gentoo (124) on three islands (Biscoe, Dream, Torgersen). Each row is one individual penguin with bill and flipper measurements, body mass, sex, and island of origin. Unlike the Iris dataset, penguins introduces real complications: sexual dimorphism, partial species-island confounding (not all species appear on all islands), and a small fraction of missing values. Field biologists use morphometric data like this to study population structure, sexual selection, and ecological niche separation.

## Online Resources

- **Dataset**: [seaborn-data/penguins.csv](https://github.com/mwaskom/seaborn-data/blob/master/penguins.csv)
- **Documentation**: [seaborn.load_dataset](https://seaborn.pydata.org/generated/seaborn.load_dataset.html)

## Codebook

| Column | Type | Range / Values | Description |
|--------|------|----------------|-------------|
| `species` | str | Adelie, Chinstrap, Gentoo | Penguin species |
| `island` | str | Biscoe, Dream, Torgersen | Island of origin |
| `bill_length_mm` | float64 | 32.1 - 59.6 mm; 0.6% missing | Culmen length |
| `bill_depth_mm` | float64 | 13.1 - 21.5 mm; 0.6% missing | Culmen depth |
| `flipper_length_mm` | float64 | 172 - 231 mm; 0.6% missing | Flipper length |
| `body_mass_g` | float64 | 2,700 - 6,300 g; 0.6% missing | Body mass in grams |
| `sex` | str | Male, Female; ~3.2% missing | Sex of the penguin |

**Target**: `species`: three-class string label (Adelie / Chinstrap / Gentoo). `sex` can also serve as a binary target for within-species dimorphism models.

## What You Can Learn Here

- Multi-class classification with overlapping classes: Adelie and Chinstrap share measurement ranges, while Gentoo is more distinct
- Sexual dimorphism as a confound: males and females differ systematically in body mass and bill dimensions within each species
- Island as a potential feature with leakage risk: species and island are correlated, so including island can inflate accuracy
- Dropping vs. encoding missing values: 11 rows have missing measurements, concentrated in the same individuals

## Research Questions

**EDA**
1. How do mean bill length and body mass compare across species and between sexes?
2. Which pair of measurements best separates all three species visually? Is one species always linearly separable from the other two?
3. Is island a proxy for species (do species occupy distinct islands), and if so, should it be used as a feature in a classification model?

**Modeling**
1. Classify species from bill length and flipper length using logistic regression; report per-class accuracy and the confusion matrix.
2. Add sex as a feature (after handling the 11 missing values); evaluate whether dimorphism improves classification.
3. Compare a model with island included to one without it; quantify the information leakage island introduces and decide whether it belongs in the feature set.

---

## Available Scripts

- `penguins_01_eda.py`: distributions, correlation, Simpson's paradox, species-island confounding
- `penguins_02_modeling_baseline.py`: logistic regression, incremental features, macro F1, confusion matrix
- `penguins_03_modeling_trees.py`: decision tree, depth sweep, tree visualisation, feature importance, random forest

**General intent of scripts:** classification, from simple to more complex

See [QUESTIONS.md](QUESTIONS.md) for per-script code-reading questions.

## Disclaimer

Part of the dataset description above was compiled by AI. Check any assumptions, claims, and context.
