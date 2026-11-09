"""Data loading and preprocessing for NetGuard (NSL-KDD)."""
from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

COLS = ["duration","protocol_type","service","flag","src_bytes","dst_bytes","land",
"wrong_fragment","urgent","hot","num_failed_logins","logged_in","num_compromised",
"root_shell","su_attempted","num_root","num_file_creations","num_shells",
"num_access_files","num_outbound_cmds","is_host_login","is_guest_login","count",
"srv_count","serror_rate","srv_serror_rate","rerror_rate","srv_rerror_rate",
"same_srv_rate","diff_srv_rate","srv_diff_host_rate","dst_host_count",
"dst_host_srv_count","dst_host_same_srv_rate","dst_host_diff_srv_rate",
"dst_host_same_src_port_rate","dst_host_srv_diff_host_rate","dst_host_serror_rate",
"dst_host_srv_serror_rate","dst_host_rerror_rate","dst_host_srv_rerror_rate",
"label","difficulty"]

CATEGORICAL = ["protocol_type", "service", "flag"]
DROP = ["label", "difficulty", "num_outbound_cmds", "attack_category", "is_attack"]

# Standard NSL-KDD grouping of attack names into 4 categories
ATTACK_CATEGORY = {
    "normal": "Normal",
    # Denial of Service
    **dict.fromkeys(["back","land","neptune","pod","smurf","teardrop","apache2",
                     "udpstorm","processtable","worm","mailbomb"], "DoS"),
    # Probe / port scanning
    **dict.fromkeys(["ipsweep","nmap","portsweep","satan","mscan","saint"], "Probe"),
    # Remote to Local (includes password guessing / brute force)
    **dict.fromkeys(["ftp_write","guess_passwd","imap","multihop","phf","spy",
                     "warezclient","warezmaster","sendmail","named","snmpgetattack",
                     "snmpguess","xlock","xsnoop","httptunnel"], "R2L"),
    # User to Root
    **dict.fromkeys(["buffer_overflow","loadmodule","perl","rootkit","ps",
                     "sqlattack","xterm"], "U2R"),
}


def load_nsl_kdd(data_dir=DATA_DIR):
    """Return (train, test) DataFrames with extra target columns."""
    out = []
    for name in ("KDDTrain.txt", "KDDTest.txt"):
        df = pd.read_csv(Path(data_dir) / name, names=COLS)
        df["is_attack"] = (df["label"] != "normal").astype(int)
        df["attack_category"] = df["label"].map(ATTACK_CATEGORY)
        if df["attack_category"].isna().any():
            missing = df.loc[df["attack_category"].isna(), "label"].unique()
            raise ValueError(f"Unmapped attack labels: {missing}")
        out.append(df)
    return out[0], out[1]


def split_xy(df, target="is_attack"):
    """Features (X) and chosen target (y). target: 'is_attack' or 'attack_category'."""
    return df.drop(columns=DROP), df[target]


def build_preprocessor(X):
    """Impute -> scale numeric; impute -> one-hot categorical."""
    numeric = [c for c in X.columns if c not in CATEGORICAL]
    return ColumnTransformer([
        ("num", Pipeline([("imp", SimpleImputer(strategy="median")),
                          ("scale", StandardScaler())]), numeric),
        ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                          ("ohe", OneHotEncoder(handle_unknown="ignore"))]), CATEGORICAL),
    ])