"""Metadata registry for sklearn model families, names, and visual styles.

Pipelines are defined in the calling script; this module has no model initialisation code.
"""

from collections import defaultdict
from enum import Enum


class ModelFamily(Enum):
    BASELINE        = "Baseline"
    LINEAR          = "Linear"
    PROBABILISTIC   = "Probabilistic"
    TREE_BASED      = "Tree-based"
    KERNEL          = "Kernel"
    INSTANCE_BASED  = "Instance-based"
    NEURAL          = "Neural"
    ENSEMBLE        = "Ensemble"


class Model(str, Enum):
    def __new__(cls, label: str, family: ModelFamily) -> "Model":
        obj = str.__new__(cls, label)
        obj._value_ = label
        obj.family = family  # type: ignore[attr-defined]
        return obj

    DUMMY               = ("Dummy",              ModelFamily.BASELINE)
    LOGISTIC_REGRESSION = ("LogisticRegression", ModelFamily.LINEAR)
    SGD                 = ("SGD",                ModelFamily.LINEAR)
    GAUSSIAN_NB         = ("GaussianNB",         ModelFamily.PROBABILISTIC)
    BERNOULLI_NB        = ("BernoulliNB",        ModelFamily.PROBABILISTIC)
    DECISION_TREE       = ("DecisionTree",       ModelFamily.TREE_BASED)
    RANDOM_FOREST       = ("RandomForest",       ModelFamily.TREE_BASED)
    EXTRA_TREES         = ("ExtraTrees",         ModelFamily.TREE_BASED)
    GRADIENT_BOOSTING   = ("GradientBoosting",   ModelFamily.TREE_BASED)
    ADA_BOOST           = ("AdaBoost",           ModelFamily.TREE_BASED)
    SVM_RBF             = ("SVM (RBF)",          ModelFamily.KERNEL)
    SVM_LINEAR          = ("SVM (linear)",       ModelFamily.KERNEL)
    KNN_K5              = ("KNN (k=5)",          ModelFamily.INSTANCE_BASED)
    KNN_K3              = ("KNN (k=3)",          ModelFamily.INSTANCE_BASED)
    MLP                 = ("MLP",               ModelFamily.NEURAL)


FAMILY_COLORS: dict[ModelFamily, str] = {
    ModelFamily.BASELINE:        "tab:gray",
    ModelFamily.LINEAR:          "tab:blue",
    ModelFamily.PROBABILISTIC:   "tab:cyan",
    ModelFamily.TREE_BASED:      "tab:green",
    ModelFamily.KERNEL:          "tab:red",
    ModelFamily.INSTANCE_BASED:  "tab:orange",
    ModelFamily.NEURAL:          "tab:purple",
    ModelFamily.ENSEMBLE:        "tab:brown",
}

# Marker sequence applied within each family: first member → "o", second → "^", …
# Colour (FAMILY_COLORS) is the primary family differentiator; marker distinguishes siblings.
_WITHIN_FAMILY_MARKERS: list[str] = ["o", "^", "s", "D", "v"]


def assign_markers(names: list[Model]) -> dict[Model, str]:
    """Assign within-family markers by position in the provided name list.

    First member of a family gets 'o', second '^', third 's', and so on.
    The order of `names` determines sibling rank within each family.
    """
    counters: defaultdict[ModelFamily, int] = defaultdict(int)
    result: dict[Model, str] = {}
    for name in names:
        family: ModelFamily = name.family  # type: ignore[attr-defined]
        result[name] = _WITHIN_FAMILY_MARKERS[counters[family] % len(_WITHIN_FAMILY_MARKERS)]
        counters[family] += 1
    return result
