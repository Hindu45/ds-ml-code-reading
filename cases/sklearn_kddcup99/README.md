# `kddcup99`: KDD Cup 1999 Network Intrusion Detection

> A benchmark dataset of ~100,000 network connection records from a simulated military environment, each labeled as normal traffic or one of several named attack categories.

## Contents

- [Domain Context](#domain-context)
- [The SA Subset](#the-sa-subset)
- [Online Resources](#online-resources)
- [Codebook](#codebook)
- [What You Can Learn Here](#what-you-can-learn-here)
- [Research Questions](#research-questions)
- [Available Scripts](#available-scripts)

## Domain Context

Network intrusion detection asks whether individual TCP/IP connections are malicious or legitimate. Each row represents one connection (42 columns total), described by packet-level statistics (byte counts, duration, error rates) and connection metadata (protocol, destination service, handshake flag). In practice, a deployed system would score live connections in real time and alert an analyst when the anomaly score crosses a threshold. The data was generated at MIT Lincoln Laboratory in 1998 as part of the DARPA Intrusion Detection Evaluation program and later used in the 1999 KDD Cup competition.

## The SA Subset

`fetch_kddcup99(subset="SA")` loads one of several named partitions of the full competition dataset. The SA subset (~100,655 rows, 42 columns) is the partition used throughout these scripts; it is large enough to reveal distributional structure while remaining fast to load and process.

**Class balance.** Normal connections are the strong majority: 97,278 normal (~96.6%) vs 3,377 attacks (~3.4%). This is a standard minority-class detection setup. The dominant attack is `smurf.` (2,409 rows, ~2.4% of total), followed by `neptune.` (898 rows, ~0.9%); all other attack types together account for fewer than 100 rows. This has direct consequences for modeling: A classifier predicting 'normal' for every row achieves ~97% accuracy while detecting zero attacks -> accuracy is a useless metric here.

## Online Resources

- **Dataset (sklearn)**: [sklearn.datasets.fetch_kddcup99](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.fetch_kddcup99.html)
- **Original competition data**: [KDD Cup 1999 -- UCI KDD Archive](http://kdd.ics.uci.edu/databases/kddcup99/kddcup99.html)
- [Original task description](https://kdd.org/kdd-cup/view/kdd-cup-1999/Tasks), including four main attack categories

## Codebook

The raw data stores every column as Python `bytes` objects regardless of underlying type. All columns must be decoded to `str` before numeric conversion -- a realistic data-loading pattern demonstrated in `kdd_01_eda.py`.

| Column group | Columns | Type | Description |
|---|---|---|---|
| Traffic volume | `src_bytes`, `dst_bytes`, `duration` | numeric | Bytes sent/received and connection length in seconds. Highly right-skewed; log-transformed before modeling. |
| Connection topology | `protocol_type`, `service`, `flag` | categorical | Protocol (tcp/udp/icmp), destination service (http, ftp, smtp, ...), connection close state: SF = normal full-duplex close; S0 = SYN sent, no response (neptune signature); REJ = connection refused. |
| Security events | `land`, `logged_in`, `root_shell`, `su_attempted`, `is_host_login`, `is_guest_login` | binary 0/1 | Rare flags that are near-diagnostic when set: `root_shell=1` indicates privilege escalation; `logged_in=0` is a strong DoS indicator. |
| 2-second window rates | `serror_rate`, `rerror_rate`, `same_srv_rate`, `diff_srv_rate` and `srv_*` / `dst_host_*` variants | float [0, 1] | Fraction of connections in the preceding 2-second window with the given property. The `serror_rate` family (SYN-error fraction) is the strongest single discriminator for neptune-style floods. |
| Host/service counters | `count`, `srv_count`, `dst_host_count`, `dst_host_srv_count` | numeric | Number of connections to the same host or service in the recent window. `count` is the strongest single-feature signal overall (flood attacks saturate it). |
| Misc event counts | `wrong_fragment`, `hot`, `num_failed_logins`, `num_compromised`, `num_root`, `num_file_creations`, `num_shells`, `num_access_files` | numeric | Counts of specific events within a connection (failed logins, root accesses, etc.). |

**Target**: `labels` (string) -- connection type, e.g. `normal.`, `smurf.`, `neptune.`, `ipsweep.`, `portsweep.`, `satan.`, `teardrop.`, `warezclient.`, `back.`. Scripts derive a binary column `is_attack` (0 = normal, 1 = any attack) for model evaluation. In the SA subset, `smurf.` is the dominant attack type (~2.4% of rows); `neptune.` is second (~0.9%); all remaining attack types together are under 0.1%.

## What You Can Learn Here

- Minority-class detection: when attacks are rare (~3%), raw accuracy is misleading and default model parameters (e.g. `contamination=0.1`) are miscalibrated
- One-class anomaly detection: training exclusively on normal examples to flag deviations, including novel attack types absent from training
- Feature engineering for skewed distributions: log-transforms, rate features, and dropping correlated redundancy before fitting a tree-based model
- Calibration-free evaluation: decoupling anomaly scores from a fixed threshold by using ROC-AUC and contamination sweeps to understand the precision/recall trade-off

## Research Questions

**EDA**
1. Which connection features show the strongest separation between normal and attack traffic, and why?
2. Why does the 3.4% attack rate in the SA subset matter when choosing a model's contamination parameter, and in which direction does the default `contamination=0.1` mis-calibrate?
3. Which categorical feature values (`flag`, `protocol_type`) are nearly exclusive to attack traffic?
4. How does the correlation structure among rate features (the `serror_rate` family) motivate feature selection before modeling?

**Modeling**
1. Why is Isolation Forest trained on normal connections only rather than on all labeled data?
2. How does the `contamination` parameter affect the precision/recall trade-off, and which setting maximises F1 on this subset?
3. Which attack categories does the model miss entirely, and why are they structurally harder to detect with numeric features alone?
4. How does the full 11-feature Isolation Forest compare to a single-feature threshold rule on `serror_rate`, and what does this reveal about feature complementarity?

---

## Available Scripts

- `kdd_01_eda.py`: class imbalance, feature discriminability, rate features, correlation matrix
- `kdd_02_isolation_forest.py`: one-class learning, ROC-AUC, contamination sweep, per-attack recall
- `kdd_03_isolation_forest_methodexplain.py`: isolation depth visualization, single-tree partitions, feature-pair ranking
- `kdd_04_binary_classifier.py`: supervised binary classification, cross-validation, class imbalance, RF balanced weights, permutation importance
- `kdd_05_multiclass_attacks.py`: supervised multi-class classification, class imbalance, compare imbalance methods

**General intent of scripts:** Anomaly analysis (unsupervised, isolation forests), classification of attacks (binary/multivariate), imbalanced

See [QUESTIONS.md](QUESTIONS.md) for per-script code-reading questions.

## Disclaimer

Part of the dataset description above was compiled by AI. Check any assumptions, claims, and context.
