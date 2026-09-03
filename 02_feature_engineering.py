"""
02_feature_engineering.py
==========================
- Participant-independent 80/20 train/test split (mirrors paper)
- Iterative imputation for missing values
- StandardScaler normalisation
Run AFTER 01_prepare_data.py
"""

import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from sklearn.experimental import enable_iterative_imputer  # noqa
from sklearn.impute import IterativeImputer
from sklearn.preprocessing import StandardScaler

SEED    = 42
np.random.seed(SEED)

IN_CSV  = Path("data/processed/supervised.csv")
OUT_DIR = Path("data/processed")

LABEL_COLS = ["target_stress", "target_mood", "target_health"]
SKIP_COLS  = {"uid", "day"} | set(LABEL_COLS)


def load():
    df = pd.read_csv(IN_CSV)
    print(f"[INFO] Loaded {df.shape}")
    return df


def split(df):
    pids    = df["uid"].unique()
    rng     = np.random.default_rng(SEED)
    pids    = rng.permutation(pids)
    n_train = int(len(pids) * 0.8)
    train   = df[df["uid"].isin(pids[:n_train])].copy()
    test    = df[df["uid"].isin(pids[n_train:])].copy()
    print(f"[INFO] Train: {len(train)} samples ({n_train} participants)")
    print(f"[INFO] Test : {len(test)} samples ({len(pids)-n_train} participants)")
    return train, test


def process(train, test):
    feat_cols = [c for c in train.columns if c not in SKIP_COLS]

    X_tr = train[feat_cols].values.astype(float)
    X_te = test[feat_cols].values.astype(float)

    print("[INFO] Imputing …")
    imp = IterativeImputer(random_state=SEED, max_iter=10)
    X_tr = imp.fit_transform(X_tr)
    X_te = imp.transform(X_te)

    print("[INFO] Scaling …")
    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_tr)
    X_te = scaler.transform(X_te)

    return X_tr, X_te, feat_cols, imp, scaler


if __name__ == "__main__":
    df           = load()
    train, test  = split(df)
    X_tr, X_te, feat_cols, imp, scaler = process(train, test)

    y_train = {lbl: train[lbl].values for lbl in LABEL_COLS}
    y_test  = {lbl: test[lbl].values  for lbl in LABEL_COLS}

    np.save(OUT_DIR / "X_train.npy", X_tr)
    np.save(OUT_DIR / "X_test.npy",  X_te)
    joblib.dump(y_train,    OUT_DIR / "y_train.pkl")
    joblib.dump(y_test,     OUT_DIR / "y_test.pkl")
    joblib.dump(imp,        OUT_DIR / "imputer.pkl")
    joblib.dump(scaler,     OUT_DIR / "scaler.pkl")
    joblib.dump(feat_cols,  OUT_DIR / "feature_cols.pkl")
    train[["uid","day"] + LABEL_COLS].to_csv(OUT_DIR / "train_meta.csv", index=False)
    test[["uid","day"]  + LABEL_COLS].to_csv(OUT_DIR / "test_meta.csv",  index=False)

    print(f"\n[SAVED] X_train={X_tr.shape}, X_test={X_te.shape}")
