"""EDA for the KDD Cup 1999 dataset: which network connection features distinguish normal traffic from intrusion attacks?"""

# %%
""" [1] Imports & config"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from cases.sklearn_kddcup99.kdd_utils import NUMERIC_COLS, BINARY_COLS, ATTACK_CATEGORY, load_kddcup99

PLOT_DIR = Path(__file__).parent / "plots"
PLOT_DIR.mkdir(exist_ok=True)

RANDOM_STATE = 42

CATEGORICAL_COLS = ["protocol_type", "service", "flag"]


# %%
""" [2] Load & decode
KDD Cup 1999 'SA' subset: ~100K network connection records from a simulated
DARPA intrusion detection environment (1998). Each row = one connection.

Raw data quirk: sklearn stores every column as Python bytes objects, regardless
of the underlying type. All columns must be decoded to str before numeric
conversion — a common real-world data loading pattern worth demonstrating.

We add a binary ground-truth label 'is_attack' immediately (1 = any attack,
0 = normal). This column is used to evaluate Isolation Forest in the next script.
"""
df = load_kddcup99(random_state=RANDOM_STATE)

print(df.shape)
print(f"\nDtype summary:\n{df.dtypes.value_counts()}")
print(f"\nMissing values after conversion:\n{df.isnull().sum()[df.isnull().sum() > 0]}")
print(f"\nFull label distribution:\n{df['labels'].value_counts().to_string()}")


# %%
""" [3] Class distribution — the inverted imbalance
Standard anomaly detection assumes anomalies are rare (typically 1–10%).
KDD Cup 1999 (SA subset) is inverted:
  'normal.' connections are only ~19% of rows.
  'neptune' (a SYN-flood DoS attack) alone accounts for ~52%.

This has a direct consequence for Isolation Forest:
  - The default contamination=0.1 is badly miscalibrated here.
  - In the modeling script we will train on 'normal' rows only (one-class setup),
    then score all rows and evaluate using is_attack as ground truth.
"""
n_normal = (df["is_attack"] == 0).sum()
n_attack = (df["is_attack"] == 1).sum()
print(f"\nNormal:  {n_normal:,}  ({n_normal / len(df) * 100:.1f}%)")
print(f"Attack:  {n_attack:,}  ({n_attack / len(df) * 100:.1f}%)")

label_counts = df["labels"].value_counts()
bar_colors = ["steelblue" if lbl == "normal." else "tomato"
              for lbl in label_counts.index]

fig, ax = plt.subplots(figsize=(10, 5))
ax.barh(label_counts.index[::-1], label_counts.values[::-1],
        color=bar_colors[::-1])
ax.set_xlabel("Number of connections")
ax.set_title("Connection type distribution — SA subset\nblue = normal, red = attack")
fig.tight_layout()
fig.savefig(PLOT_DIR / "kddcup99_class_distribution.png")
plt.show()


# %%
""" [4] Categorical features — protocol, service, flag
Three categorical features describe the connection topology:
  protocol_type — tcp / udp / icmp (3 levels)
  service       — destination service: http, ftp, smtp, etc. (many levels)
  flag          — connection state at close (11 levels):
                    SF = normal full-duplex close
                    S0 = SYN sent, no response received (neptune signature)
                    REJ = connection refused

'flag' is the most attack-discriminating single feature:
  flag='S0' maps almost entirely to neptune — the connection handshake is
  initiated but never completed, leaving a half-open SYN in the queue.
'protocol_type=icmp' is heavily associated with smurf attacks (ICMP echo flood).
"""
fig, axes = plt.subplots(1, 3, figsize=(16, 5))

for ax, col in zip(axes, CATEGORICAL_COLS):
    ct = pd.crosstab(df[col], df["is_attack"], normalize="index") * 100
    ct.columns = ["normal %", "attack %"]
    ct = ct.sort_values("attack %", ascending=True)
    ct.plot(kind="barh", stacked=True, ax=ax,
            color=["steelblue", "tomato"], legend=(ax is axes[-1]))
    ax.set_title(col)
    ax.set_xlabel("% of connections")
    ax.set_xlim(0, 100)

fig.suptitle("Attack rate by categorical feature value", y=1.01)
fig.tight_layout()
fig.savefig(PLOT_DIR / "kddcup99_categorical_attack_rates.png")
plt.show()


# %%
""" [5] src_bytes and duration distributions
src_bytes (bytes sent from source to destination):
  neptune  = 0 bytes — SYN packet only; connection never established, no payload
  smurf    = 1,032 bytes — fixed ICMP packet, repeated at volume
  normal   = spans 0 to millions depending on service and transfer size

duration (connection length in seconds):
  DoS attacks = near-zero (burst of short, incomplete connections)
  Normal interactive sessions = tens to thousands of seconds
  Most connections cluster near zero; log scale is required to see structure.

Log₁₀(x+1) handles the many zero values without dropping them.
"""
fig, axes = plt.subplots(2, 2, figsize=(12, 7))

for row_idx, col in enumerate(["src_bytes", "duration"]):
    vals_log = np.log10(df[col] + 1)

    axes[row_idx][0].hist(vals_log, bins=60, color="gray", edgecolor="none")
    axes[row_idx][0].set_title(f"log₁₀({col}+1) — all connections")
    axes[row_idx][0].set_xlabel("log₁₀(value + 1)")

    for grp_label, color, mask in [
        ("normal", "steelblue", df["is_attack"] == 0),
        ("attack",  "tomato",   df["is_attack"] == 1),
    ]:
        axes[row_idx][1].hist(vals_log[mask], bins=60,
                              alpha=0.5, color=color,
                              label=grp_label, edgecolor="none")
    axes[row_idx][1].set_title(f"log₁₀({col}+1) — normal vs attack")
    axes[row_idx][1].set_xlabel("log₁₀(value + 1)")
    axes[row_idx][1].legend(fontsize=8)

fig.tight_layout()
fig.savefig(PLOT_DIR / "kddcup99_numeric_distributions.png")
plt.show()


# %%
""" [6] Binary security flag features
Binary features signal specific security events. They are almost always 0
(rare events) but near-perfect attack indicators when they fire:
  root_shell=1  — attacker obtained a root shell; effectively diagnostic of R2L/U2R
  su_attempted=1 — privilege escalation attempted
  is_guest_login=1 — guest account used; unusual on production systems

logged_in (1 = successful authentication) shows a useful reversal:
  high in normal traffic — users authenticate before using services
  low in DoS attacks — SYN floods and ICMP floods never reach authentication

logged_in=0 is therefore a strong negative indicator for bulk DoS detection.
"""
flag_rates = pd.DataFrame({
    "normal": df[df["is_attack"] == 0][BINARY_COLS].mean() * 100,
    "attack": df[df["is_attack"] == 1][BINARY_COLS].mean() * 100,
})
print("\nBinary flag activation rates (%):")
print(flag_rates.round(3).to_string())

x = np.arange(len(BINARY_COLS))
width = 0.35
fig, ax = plt.subplots(figsize=(8, 4))
ax.bar(x - width / 2, flag_rates["normal"], width, label="normal", color="steelblue")
ax.bar(x + width / 2, flag_rates["attack"],  width, label="attack",  color="tomato")
ax.set_xticks(x)
ax.set_xticklabels(BINARY_COLS, rotation=20, ha="right")
ax.set_ylabel("% of connections where flag = 1")
ax.set_title("Binary security flags — normal vs attack")
ax.legend()
fig.tight_layout()
fig.savefig(PLOT_DIR / "kddcup99_binary_flags.png")
plt.show()


# %%
""" [7] Rate features — 2-second window statistics
Rate features summarize traffic patterns in the 2-second window before each connection:
  serror_rate — fraction of connections ending in SYN error (connection timed out)
  rerror_rate — fraction ending in REJ (connection refused)
  same_srv_rate — fraction targeting the same service (high = single-service flood)
  diff_srv_rate — fraction targeting different services (high = port-scanning)

neptune creates a sharp bimodal distribution on serror_rate:
  During the flood: serror_rate ≈ 1.0 — every connection in the window timed out.
  Normal traffic:   serror_rate ≈ 0.0 — connections complete normally.

This bimodality is central to why Isolation Forest works well here: the neptune
cluster is dense and tightly separated from normal traffic, making each neptune
point easy to isolate in fewer splits.
"""
rate_cols = ["serror_rate", "rerror_rate", "same_srv_rate", "diff_srv_rate"]
fig, axes = plt.subplots(1, 4, figsize=(15, 4))

for ax, col in zip(axes, rate_cols):
    for grp_label, color, mask in [
        ("normal", "steelblue", df["is_attack"] == 0),
        ("attack",  "tomato",   df["is_attack"] == 1),
    ]:
        ax.hist(df.loc[mask, col], bins=30,
                alpha=0.5, color=color, label=grp_label, edgecolor="none")
    ax.set_title(col, fontsize=9)
    ax.set_xlabel("rate (0–1)")
    if ax is axes[0]:
        ax.legend(fontsize=7)

fig.suptitle("Rate features (2-second window) — normal vs attack")
fig.tight_layout()
fig.savefig(PLOT_DIR / "kddcup99_rate_features.png")
plt.show()


# %%
""" [8] Correlation matrix
Correlations across a representative subset of numeric features.

Two strongly correlated blocks emerge:
  serror_rate / srv_serror_rate / dst_host_serror_rate / dst_host_srv_serror_rate
    → near-perfect correlation; all four measure the same SYN-flood signal at
      different aggregation levels (per-service vs per-host).
  rerror_rate variants form a parallel block for REJ errors.

src_bytes and dst_bytes are weakly correlated: attacks often flow in one
direction only (SYN floods: src_bytes=0; server responses never arrive).

For Isolation Forest, redundant correlated features can blur anomaly scores
by over-weighting one signal. Dropping duplicates within each block before
modeling is a reasonable preprocessing step — addressed in the next script.
"""
corr_features = [
    "src_bytes", "dst_bytes", "duration",
    "count", "srv_count", "dst_host_count",
    "serror_rate", "srv_serror_rate",
    "dst_host_serror_rate", "dst_host_srv_serror_rate",
    "rerror_rate", "srv_rerror_rate",
    "dst_host_rerror_rate", "dst_host_srv_rerror_rate",
    "same_srv_rate", "diff_srv_rate",
]
corr = df[corr_features].corr()

fig, ax = plt.subplots(figsize=(10, 9))
im = ax.imshow(corr, vmin=-1, vmax=1, cmap="RdBu_r")
plt.colorbar(im, ax=ax)
n = len(corr_features)
ax.set_xticks(range(n))
ax.set_xticklabels(corr_features, rotation=45, ha="right", fontsize=8)
ax.set_yticks(range(n))
ax.set_yticklabels(corr_features, fontsize=8)
for i in range(n):
    for j in range(n):
        ax.text(j, i, f"{corr.iloc[i, j]:.2f}",
                ha="center", va="center", fontsize=6)
ax.set_title("Correlation matrix — selected numeric features")
fig.tight_layout()
fig.savefig(PLOT_DIR / "kddcup99_correlation.png")
plt.show()


# %%
""" [9] Feature discriminability — standardized mean difference by label
Standardized mean difference per numeric feature: (attack mean − normal mean) / pooled std.
Larger absolute value = more discriminating between normal and attack connections.

Top attack indicators (red, positive):
  serror_rate family — neptune drives SYN-error rate to 1.0 across the board
  same_srv_rate — DoS floods repeat the same service in every window

Top normal indicators (blue, negative — higher in normal than attack):
  src_bytes   — normal sessions send actual application data; SYN floods send nothing
  dst_bytes   — servers respond with data in normal sessions; attacks never complete
  logged_in   — interactive users authenticate; flood attacks bypass authentication

This plot directly previews which features will contribute most to IF anomaly scores.
"""
normal_means = df[df["is_attack"] == 0][NUMERIC_COLS].mean()
attack_means = df[df["is_attack"] == 1][NUMERIC_COLS].mean()
pooled_std = df[NUMERIC_COLS].std().replace(0, 1)
effect = ((attack_means - normal_means) / pooled_std).sort_values()

bar_colors = ["tomato" if v > 0 else "steelblue" for v in effect]
fig, ax = plt.subplots(figsize=(8, 10))
ax.barh(effect.index, effect.values, color=bar_colors)
ax.axvline(0, color="black", linewidth=0.8)
ax.set_xlabel("Standardized mean difference (attack − normal) / pooled std")
ax.set_title("Feature discriminability: attack vs normal")
fig.tight_layout()
fig.savefig(PLOT_DIR / "kddcup99_feature_discriminability.png")
plt.show()


# %%
""" [10] Attack category distribution — five-class taxonomy
The 23 raw attack labels collapse into four threat categories per the official
KDD Cup 1999 taxonomy: DoS, Probe, R2L, U2R. Alongside 'Normal', this gives
the five-class framing used in kdd_05.

DoS dominates (neptune + smurf). R2L and U2R are rare — together under 0.5%
of rows, almost invisible in the binary is_attack view but central to the
multiclass imbalance problem explored in kdd_05.
"""
df["attack_category"] = df["labels"].map(ATTACK_CATEGORY)

cat_counts = df["attack_category"].value_counts()
print("\nAttack category distribution:")
print(cat_counts.to_string())

cat_colors = {
    "Normal": "steelblue", "DoS": "tomato", "Probe": "darkorange",
    "R2L": "purple", "U2R": "crimson",
}

fig, ax = plt.subplots(figsize=(8, 4))
ax.bar(cat_counts.index, cat_counts.values,
       color=[cat_colors.get(c, "gray") for c in cat_counts.index])
ax.set_yscale("log")
ax.set_ylabel("Number of connections (log scale)")
ax.set_title("Attack category distribution — five-class KDD taxonomy")
for i, (cat, n) in enumerate(cat_counts.items()):
    ax.text(i, n * 1.5, f"{n:,}", ha="center", va="bottom", fontsize=8)
fig.tight_layout()
fig.savefig(PLOT_DIR / "kddcup99_attack_category_distribution.png")
plt.show()
