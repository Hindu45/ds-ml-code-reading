"""Research question: How does tree depth affect species classification, and what does a random forest add?
purpose: modeling | style: library-optimal | flags: docstring-depth: minimal
"""

# %%
""" [1] Imports & config"""
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree
from cases.utils.tree_helpers import plot_dt_boundary_sequence

RANDOM_STATE = 42
PLOT_DIR = Path(__file__).parent / "plots"
PLOT_DIR.mkdir(exist_ok=True)

NUMERIC_COLS = ["bill_length_mm", "bill_depth_mm", "flipper_length_mm", "body_mass_g"]
MAX_DEPTH_RANGE = range(1, 9)

# %%
""" [2] Load, split
All four numeric features; same 60/20/20 stratified split as baseline script.
Scaling is not needed for tree-based models — included in the pipeline comparison (script 04).
"""
df = sns.load_dataset("penguins").dropna(subset=NUMERIC_COLS).copy()
X = df[NUMERIC_COLS].values
y = df["species"].values

X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.4, stratify=y, random_state=RANDOM_STATE
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=RANDOM_STATE
)
print(f"Train: {len(X_train)}  Val: {len(X_val)}  Test: {len(X_test)}")

# %%
""" [3] Simple 2-feature tree — decision boundary sequence
One split at a time: root splits the full space; left and right subtrees then
partition their half independently. Each panel adds one rule from the printed
export_text output, so the visual and the text can be read side by side.
"""
i_flip = NUMERIC_COLS.index("flipper_length_mm")
i_bill = NUMERIC_COLS.index("bill_length_mm")
FEAT_2D = ["flipper_length_mm", "bill_length_mm"]

dt_2d = DecisionTreeClassifier(max_depth=2, random_state=25)
dt_2d.fit(X_train[:, [i_flip, i_bill]], y_train)

plot_dt_boundary_sequence(
    dt_2d,
    X_train[:, [i_flip, i_bill]], y_train,
    X[:, [i_flip, i_bill]], y,
    feature_names=FEAT_2D,
    save_path=PLOT_DIR / "penguins_dt_boundary_2d.png",
)
print(export_text(dt_2d, feature_names=FEAT_2D))
print(f"2-feature tree (depth={dt_2d.max_depth})  val acc: {accuracy_score(y_val, dt_2d.predict(X_val[:, [i_flip, i_bill]])):.3f}")

# %%
""" [4] Decision tree — depth sweep
Train and val accuracy for max_depth 1–8. Goal: find the shallowest depth where
val accuracy plateaus — that depth achieves peak performance without added complexity.
"""
train_accs, val_accs = [], []
for d in MAX_DEPTH_RANGE:
    dt = DecisionTreeClassifier(max_depth=d, random_state=RANDOM_STATE)
    dt.fit(X_train, y_train)
    train_accs.append(accuracy_score(y_train, dt.predict(X_train)))
    val_accs.append(accuracy_score(y_val, dt.predict(X_val)))

best_depth = MAX_DEPTH_RANGE[int(np.argmax(val_accs))]

fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(MAX_DEPTH_RANGE, train_accs, marker="o", label="Train")
ax.plot(MAX_DEPTH_RANGE, val_accs, marker="s", label="Val")
ax.axvline(best_depth, color="gray", linestyle="--", linewidth=1, label=f"best depth={best_depth}")
ax.set_xlabel("max_depth")
ax.set_ylabel("Accuracy")
ax.set_title("Decision tree — depth vs accuracy")
ax.legend()
fig.tight_layout()
fig.savefig(PLOT_DIR / "penguins_dt_depth_sweep.png")
plt.show()
print(f"Best depth by val accuracy: {best_depth}  (train acc = {train_accs[best_depth - 1]:.3f},  val acc = {max(val_accs):.3f})")
print(f"Saved: {PLOT_DIR / 'penguins_dt_depth_sweep.png'}")

# %%
""" [5] Decision rules — tree diagram and printed rules
The graphic shows the split hierarchy; export_text prints every if/then rule so
any prediction can be traced by hand — the interpretability advantage over logistic
regression coefficients.
"""
dt_best = DecisionTreeClassifier(max_depth=best_depth, random_state=RANDOM_STATE)
dt_best.fit(X_train, y_train)

fig, ax = plt.subplots(figsize=(14, 6))
plot_tree(
    dt_best,
    feature_names=NUMERIC_COLS,
    class_names=dt_best.classes_,
    filled=True,
    rounded=True,
    proportion=True,
    impurity=False,
    fontsize=8,
    ax=ax,
)

ax.set_title(f"Decision tree (max_depth={best_depth})")
fig.tight_layout()
fig.savefig(PLOT_DIR / "penguins_dt_tree_plot.png")
plt.show()
print(export_text(dt_best, feature_names=NUMERIC_COLS, max_depth=1))
print(f"Saved: {PLOT_DIR / 'penguins_dt_tree_plot.png'}")

# %%
""" [6] Feature importances
Impurity-based importance from the fitted tree. Flipper length and bill length
typically dominate; bill depth contributes because it separates Adelie from Chinstrap.
"""
importances = dt_best.feature_importances_
sort_idx = np.argsort(importances)[::-1]

fig, ax = plt.subplots(figsize=(6, 4))
ax.bar(range(len(NUMERIC_COLS)), importances[sort_idx])
ax.set_xticks(range(len(NUMERIC_COLS)))
ax.set_xticklabels([NUMERIC_COLS[i] for i in sort_idx], rotation=20, ha="right")
ax.set_ylabel("Impurity-based importance")
ax.set_title(f"Decision tree feature importances (depth={best_depth})")
fig.tight_layout()
fig.savefig(PLOT_DIR / "penguins_dt_importances.png")
plt.show()
for i in sort_idx:
    print(f"  {NUMERIC_COLS[i]}: {importances[i]:.3f}")
print(f"Saved: {PLOT_DIR / 'penguins_dt_importances.png'}")

# %%
""" [7] Random forest — fit and evaluate
Random forest averages 200 trees trained on random feature subsets, reducing
variance at the cost of losing the readable single-tree rules.
"""
rf = RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_val)

print("RandomForest (n_estimators=200) — val set")
print(classification_report(y_val, y_pred_rf))

# %%
""" [8] Test set evaluation
Test set touched once. The accuracy difference makes the interpretability tradeoff
explicit: is the gain from RF worth losing the readable decision rules?
"""
acc_dt = accuracy_score(y_test, dt_best.predict(X_test))
acc_rf = accuracy_score(y_test, rf.predict(X_test))
print(f"Test accuracy  —  DecisionTree (depth={best_depth}): {acc_dt:.3f}  |  RandomForest: {acc_rf:.3f}")
print(f"Accuracy gain from RF: {acc_rf - acc_dt:+.3f}  (DT rules remain human-readable; RF rules do not)")
