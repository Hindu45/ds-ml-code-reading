"""Resampling helpers for class-imbalance experiments.

Each function takes a DataFrame X and Series y, returns (X_resampled, y_resampled)
as a tuple so callers can unpack directly into a strategy registry.
"""
import pandas as pd


def random_undersample(
    X: pd.DataFrame,
    y: pd.Series,
    target: int,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.Series]:
    """Trim each class down to *target* rows; keep minority classes unchanged.

    Args:
        X: Feature DataFrame aligned with y.
        y: Class labels.
        target: Maximum rows to keep per class.
        random_state: Seed for reproducibility.

    Returns:
        Tuple of (X_resampled, y_resampled), shuffled.
    """
    df = X.copy()
    df["__y__"] = y.values
    parts = [
        sub.sample(n=min(len(sub), target), random_state=random_state)
        for _, sub in df.groupby("__y__")
    ]
    out = pd.concat(parts).sample(frac=1, random_state=random_state)
    return out.drop(columns="__y__"), out["__y__"].rename(y.name)


def random_oversample(
    X: pd.DataFrame,
    y: pd.Series,
    target: int,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.Series]:
    """Bootstrap minority classes up to *target* rows; keep majority classes unchanged.

    Args:
        X: Feature DataFrame aligned with y.
        y: Class labels.
        target: Minimum rows to guarantee per class (sampled with replacement).
        random_state: Seed for reproducibility.

    Returns:
        Tuple of (X_resampled, y_resampled), shuffled.
    """
    df = X.copy()
    df["__y__"] = y.values
    parts = [
        sub.sample(n=max(len(sub), target), replace=True, random_state=random_state)
        for _, sub in df.groupby("__y__")
    ]
    out = pd.concat(parts).sample(frac=1, random_state=random_state)
    return out.drop(columns="__y__"), out["__y__"].rename(y.name)
