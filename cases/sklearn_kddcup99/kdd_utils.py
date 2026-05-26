"""Shared constants and helpers for KDD Cup 1999 analysis scripts."""
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.datasets import fetch_kddcup99
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

NUMERIC_COLS = [
    "duration", "src_bytes", "dst_bytes", "wrong_fragment",
    "hot", "num_failed_logins", "num_compromised",
    "num_root", "num_file_creations", "num_shells", "num_access_files",
    "count", "srv_count",
    "serror_rate", "srv_serror_rate", "rerror_rate", "srv_rerror_rate",
    "same_srv_rate", "diff_srv_rate", "srv_diff_host_rate",
    "dst_host_count", "dst_host_srv_count",
    "dst_host_same_srv_rate", "dst_host_diff_srv_rate",
    "dst_host_same_src_port_rate", "dst_host_srv_diff_host_rate",
    "dst_host_serror_rate", "dst_host_srv_serror_rate",
    "dst_host_rerror_rate", "dst_host_srv_rerror_rate",
]
BINARY_COLS = ["land", "logged_in", "root_shell", "su_attempted", "is_host_login", "is_guest_login"]

# 11 non-redundant features; log₁₀(x+1) variants for skewed columns (see EDA in kdd_01)
NUMERIC_FEATURES = [
    "log_src_bytes", "log_dst_bytes", "log_duration",
    "count", "srv_count",
    "serror_rate", "rerror_rate",
    "same_srv_rate", "diff_srv_rate",
    "logged_in", "dst_host_count",
]

# Official KDD Cup 1999 attack taxonomy
ATTACK_CATEGORY: dict[str, str] = {
    "normal.":           "Normal",
    "back.":             "DoS",   "land.":    "DoS",   "neptune.":  "DoS",
    "pod.":              "DoS",   "smurf.":   "DoS",   "teardrop.": "DoS",
    "apache2.":          "DoS",   "udpstorm.":"DoS",   "processtable.": "DoS",
    "mailbomb.":         "DoS",   "worm.":    "DoS",
    "ipsweep.":          "Probe", "nmap.":    "Probe", "portsweep.":"Probe",
    "satan.":            "Probe", "mscan.":   "Probe", "saint.":    "Probe",
    "ftp_write.":        "R2L",   "guess_passwd.": "R2L", "imap.":  "R2L",
    "multihop.":         "R2L",   "phf.":     "R2L",   "spy.":     "R2L",
    "warezclient.":      "R2L",   "warezmaster.": "R2L", "xlock.": "R2L",
    "xsnoop.":           "R2L",   "snmpguess.": "R2L", "sendmail.":"R2L",
    "named.":            "R2L",   "httptunnel.": "R2L","snmpgetattack.": "R2L",
    "buffer_overflow.":  "U2R",   "loadmodule.": "U2R","perl.":    "U2R",
    "rootkit.":          "U2R",   "ps.":      "U2R",   "sqlattack.":"U2R",
    "xterm.":            "U2R",
}


def load_kddcup99(add_is_attack: bool = True, random_state: int = 42) -> pd.DataFrame:
    """Fetch KDD Cup 1999 SA subset, decode bytes, and cast numeric columns.

    Args:
        add_is_attack: If True, adds binary is_attack column (1 = attack, 0 = normal).
        random_state: Passed to fetch_kddcup99.

    Returns:
        Decoded and cast DataFrame ready for analysis.
    """
    bunch = fetch_kddcup99(subset="SA", as_frame=True, random_state=random_state)
    df = bunch.frame.copy()
    for col in df.columns:
        if df[col].dtype == object and isinstance(df[col].iloc[0], bytes):
            df[col] = df[col].str.decode("utf-8")
    for col in NUMERIC_COLS + BINARY_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    if add_is_attack:
        df["is_attack"] = (df["labels"] != "normal.").astype(int)
    return df


def make_pipe(clf, cat_features: list[str] | None = None) -> Pipeline:
    """Build a preprocessing + classifier Pipeline.

    Args:
        clf: Sklearn estimator to use as the final step.
        cat_features: Categorical columns to one-hot encode. Defaults to
            ["protocol_type", "flag"].

    Returns:
        Fitted-ready Pipeline with StandardScaler on NUMERIC_FEATURES and
        OneHotEncoder on cat_features.
    """
    if cat_features is None:
        cat_features = ["protocol_type", "flag"]
    return Pipeline([
        ("pre", ColumnTransformer([
            ("num", StandardScaler(), NUMERIC_FEATURES),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_features),
        ], remainder="drop")),
        ("clf", clf),
    ])
