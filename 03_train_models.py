"""
03_train_models.py
==================
Trains 4 traditional ML models for next-day stress / mood / health prediction.
  1. Ridge Regression
  2. SVR (RBF kernel)
  3. Random Forest
  4. Gradient Boosting
  5. Multi-task Ridge (joint prediction — closest to paper's multi-task LSTM)

5-fold CV grid search on train set, then evaluate on held-out test participants.
Run AFTER 02_feature_engineering.py
"""

import numpy as np
import pandas as pd
import joblib
import json
import warnings
from pathlib import Path
from sklearn.linear_model import Ridge
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import GridSearchCV, KFold, cross_val_score
from sklearn.metrics import mean_absolute_error
from xgboost import XGBRegressor


warnings.filterwarnings("ignore")
SEED = 42
np.random.seed(SEED)

DATA_DIR  = Path("data/processed")
MODEL_DIR = Path("models")
MODEL_DIR.mkdir(parents=True, exist_ok=True)

LABEL_COLS = ["target_stress", "target_mood", "target_health"]
SHORT      = ["stress", "mood", "health"]


def load():
    X_tr   = np.load(DATA_DIR / "X_train.npy")
    X_te   = np.load(DATA_DIR / "X_test.npy")
    y_tr   = joblib.load(DATA_DIR / "y_train.pkl")
    y_te   = joblib.load(DATA_DIR / "y_test.pkl")
    return X_tr, X_te, y_tr, y_te


MODELS = {
    "ridge": (
        Ridge(random_state=SEED),
        {"alpha": [0.01, 0.1, 1.0, 10.0, 100.0]},
    ),
    "xgb":(XGBRegressor(n_estimators=200, random_state=SEED),{"max_depth": [3, 5, 7], "learning_rate": [0.01, 0.1]}),
    "svr": (
        SVR(kernel="rbf"),
        {"C": [0.1, 1.0, 10.0], "epsilon": [0.5, 1.0, 2.0], "gamma": ["scale","auto"]},
    ),
    "random_forest": (
        RandomForestRegressor(n_estimators=200, random_state=SEED, n_jobs=-1),
        {"max_depth": [None, 5, 10], "min_samples_leaf": [1, 5, 10]},
    ),
    "gradient_boosting": (
        GradientBoostingRegressor(n_estimators=200, random_state=SEED),
        {"learning_rate": [0.05, 0.1, 0.2], "max_depth": [3, 5]},
    ),
}


def train_single(name, estimator, param_grid, X_tr, y_1d):
    kf = KFold(n_splits=5, shuffle=True, random_state=SEED)
    gs = GridSearchCV(estimator, param_grid, cv=kf,
                      scoring="neg_mean_absolute_error",
                      n_jobs=-1, refit=True)
    gs.fit(X_tr, y_1d)
    cv_mae = -gs.best_score_
    print(f"    {name}: CV-MAE={cv_mae:.2f}  params={gs.best_params_}")
    return gs.best_estimator_


def train_multitask(X_tr, y_tr, X_te, y_te):
    Y_tr = np.column_stack([y_tr[l] for l in LABEL_COLS])
    Y_te = np.column_stack([y_te[l] for l in LABEL_COLS])
    valid_tr = ~np.isnan(Y_tr).any(axis=1)
    valid_te = ~np.isnan(Y_te).any(axis=1)
    X_tr, Y_tr = X_tr[valid_tr], Y_tr[valid_tr]
    X_te, Y_te = X_te[valid_te], Y_te[valid_te]
    kf   = KFold(n_splits=5, shuffle=True, random_state=SEED)
    best_alpha, best_cv = None, np.inf
    for alpha in [0.01, 0.1, 1.0, 10.0, 100.0]:
        mo     = MultiOutputRegressor(Ridge(alpha=alpha), n_jobs=-1)
        scores = cross_val_score(mo, X_tr, Y_tr, cv=kf,
                                 scoring="neg_mean_absolute_error")
        cv_mae = -scores.mean()
        if cv_mae < best_cv:
            best_cv, best_alpha = cv_mae, alpha
    print(f"    multitask_ridge: CV-MAE={best_cv:.2f}  alpha={best_alpha}")
    mo = MultiOutputRegressor(Ridge(alpha=best_alpha), n_jobs=-1)
    mo.fit(X_tr, Y_tr)
    Y_pred = mo.predict(X_te)
    maes   = {s: float(mean_absolute_error(Y_te[:,i], Y_pred[:,i]))
              for i, s in enumerate(SHORT)}
    preds  = {s: Y_pred[:,i].tolist() for i, s in enumerate(SHORT)}
    joblib.dump(mo, MODEL_DIR / "multitask_ridge.pkl")
    return maes, preds


def main():
    X_tr, X_te, y_tr, y_te = load()
    print(f"X_train={X_tr.shape}  X_test={X_te.shape}\n")

    results, all_preds = {}, {}

    for short, lbl in zip(SHORT, LABEL_COLS):
        print(f"\n── {short.upper()} ──")
        y_1d = y_tr[lbl]
        y_te_1d = y_te[lbl]
        valid = ~np.isnan(y_1d)
        valid_te = ~np.isnan(y_te_1d)

        for name, (est, grid) in MODELS.items():
            best = train_single(name, est, grid, X_tr[valid], y_1d[valid])
            preds = best.predict(X_te[valid_te])
            test_mae = mean_absolute_error(y_te_1d[valid_te], preds)
            print(f"      → Test MAE: {test_mae:.2f}")
            if name not in results:
                results[name] = {}
                all_preds[name] = {}
            results[name][short]   = round(test_mae, 3)
            all_preds[name][short] = preds.tolist()
            joblib.dump(best, MODEL_DIR / f"{name}_{short}.pkl")

    print("\n── MULTI-TASK RIDGE ──")
    mt_maes, mt_preds = train_multitask(X_tr, y_tr, X_te, y_te)
    results["multitask_ridge"]   = {k: round(v,3) for k,v in mt_maes.items()}
    all_preds["multitask_ridge"] = mt_preds

    # Save
    with open(MODEL_DIR / "results.json", "w") as f:
        json.dump(results, f, indent=2)
    joblib.dump(all_preds, MODEL_DIR / "all_preds.pkl")
    y_test_true = {s: y_te[l].tolist() for s, l in zip(SHORT, LABEL_COLS)}
    with open(MODEL_DIR / "y_test_true.json", "w") as f:
        json.dump(y_test_true, f, indent=2)

    # Print summary
    print(f"\n{'='*58}")
    print(f"{'Model':<22} {'Stress':>8} {'Mood':>8} {'Health':>8}")
    print("-" * 58)
    for m, r in results.items():
        print(f"{m:<22} {r.get('stress',float('nan')):>8.2f} "
              f"{r.get('mood',float('nan')):>8.2f} "
              f"{r.get('health',float('nan')):>8.2f}")
    print("-" * 58)
    print(f"{'[Paper] Deep LSTM':<22} {'16.80':>8} {'15.70':>8} {'15.60':>8}")
    print(f"{'[Paper] TL LSTM':<22} {'14.40':>8} {'13.50':>8} {'13.20':>8}")


if __name__ == "__main__":
    main()
