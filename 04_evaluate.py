"""
04_evaluate.py
==============
Produces:
  1. Comparison table (all models vs paper's LSTM baselines)
  2. Transfer learning simulation (mirrors paper Figure 3)
  3. Feature importance plot (Gradient Boosting)
  4. Predicted vs actual scatter plots
  5. Per-participant stability vs MAE plot

Run AFTER 03_train_models.py
"""

import numpy as np
import pandas as pd
import joblib, json, warnings
from pathlib import Path
from sklearn.metrics import mean_absolute_error
from sklearn.linear_model import Ridge

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")
SEED = 42
np.random.seed(SEED)

DATA_DIR    = Path("data/processed")
MODEL_DIR   = Path("models")
RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

SHORT      = ["stress", "mood", "health"]
LABEL_COLS = ["target_stress", "target_mood", "target_health"]
COLORS     = {"stress": "#E87B4C", "mood": "#4C9BE8", "health": "#56C271"}


def load_all():
    X_tr       = np.load(DATA_DIR / "X_train.npy")
    X_te       = np.load(DATA_DIR / "X_test.npy")
    y_tr       = joblib.load(DATA_DIR / "y_train.pkl")
    y_te       = joblib.load(DATA_DIR / "y_test.pkl")
    feat_cols  = joblib.load(DATA_DIR / "feature_cols.pkl")
    results    = json.load(open(MODEL_DIR / "results.json"))
    all_preds  = joblib.load(MODEL_DIR / "all_preds.pkl")
    test_meta  = pd.read_csv(DATA_DIR / "test_meta.csv")
    return X_tr, X_te, y_tr, y_te, feat_cols, results, all_preds, test_meta


# ── 1. Comparison table ───────────────────────────────────────────────────────

def comparison_table(results):
    rows = [{"Model": m,
             "Stress MAE": r.get("stress", float("nan")),
             "Mood MAE":   r.get("mood",   float("nan")),
             "Health MAE": r.get("health", float("nan"))}
            for m, r in results.items()]
    rows += [
        {"Model": "[Paper] Deep LSTM",    "Stress MAE": 16.8, "Mood MAE": 15.7, "Health MAE": 15.6},
        {"Model": "[Paper] CNN-LSTM",     "Stress MAE": 17.8, "Mood MAE": 16.8, "Health MAE": 16.3},
        {"Model": "[Paper] TL-LSTM 80%",  "Stress MAE": 14.4, "Mood MAE": 13.5, "Health MAE": 13.2},
    ]
    df = pd.DataFrame(rows)
    print("\n" + df.to_string(index=False))
    df.to_csv(RESULTS_DIR / "comparison_table.csv", index=False)
    print(f"[SAVED] results/comparison_table.csv")
    return df


# ── 2. Transfer learning simulation ──────────────────────────────────────────

def simulate_transfer_learning(X_tr, y_tr, X_te, y_te):
    lbl    = "target_stress"
    y_tr1d = y_tr[lbl]; y_te1d = y_te[lbl]
    valid_tr = ~np.isnan(y_tr1d); valid_te = ~np.isnan(y_te1d)

    base = Ridge(alpha=1.0).fit(X_tr[valid_tr], y_tr1d[valid_tr])
    base_mae = mean_absolute_error(y_te1d[valid_te], base.predict(X_te[valid_te]))

    fracs, tl_maes, rt_maes = [], [], []
    X_te_v, y_te_v = X_te[valid_te], y_te1d[valid_te]
    n = len(y_te_v)

    for frac in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
        n_adapt = max(1, int(n * frac))
        if n - n_adapt < 2:
            continue
        X_a, y_a = X_te_v[:n_adapt], y_te_v[:n_adapt]
        X_e, y_e = X_te_v[n_adapt:], y_te_v[n_adapt:]

        # Transfer: augment training data with new participant samples
        tl = Ridge(alpha=0.1).fit(
            np.vstack([X_tr[valid_tr], X_a]),
            np.concatenate([y_tr1d[valid_tr], y_a])
        )
        # Retrain from scratch on new participant only
        rt = Ridge(alpha=1.0).fit(X_a, y_a)

        fracs.append(int(frac * 100))
        tl_maes.append(mean_absolute_error(y_e, tl.predict(X_e)))
        rt_maes.append(mean_absolute_error(y_e, rt.predict(X_e)))
        print(f"  {int(frac*100):3d}%  TL={tl_maes[-1]:.2f}  Retrain={rt_maes[-1]:.2f}")

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(fracs, tl_maes,  "o-", color="#4C9BE8", lw=2, label="Transfer Learning")
    ax.plot(fracs, rt_maes,  "s-", color="#E87B4C", lw=2, label="Re-training")
    ax.axhline(base_mae, color="gray",   ls="--", lw=1.5, label="Baseline (0% new data)")
    ax.axhline(16.8,     color="#AAB7B8", ls=":",  lw=1.5, label="[Paper] LSTM stress=16.8")
    ax.set_xlabel("% of New Participant Data Used")
    ax.set_ylabel("Stress MAE (out of 100)")
    ax.set_title("Transfer Learning vs Re-training (Stress)", fontweight="bold")
    ax.legend(); ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "transfer_learning.png", dpi=150)
    plt.close()
    print("[SAVED] results/transfer_learning.png")


# ── 3. Feature importance ─────────────────────────────────────────────────────

def plot_feature_importance(feat_cols, top_n=15):
    fig, axes = plt.subplots(1, 3, figsize=(16, 6))
    fig.suptitle("Feature Importances — Gradient Boosting", fontweight="bold")
    for ax, short in zip(axes, SHORT):
        mp = MODEL_DIR / f"gradient_boosting_{short}.pkl"
        if not mp.exists():
            ax.set_title(f"{short} — not found"); continue
        m    = joblib.load(mp)
        imps = m.feature_importances_
        idx  = np.argsort(imps)[::-1][:top_n]
        ax.barh(range(top_n), imps[idx][::-1], color=COLORS[short], alpha=0.85)
        ax.set_yticks(range(top_n))
        ax.set_yticklabels([feat_cols[i] for i in idx][::-1], fontsize=7)
        ax.set_title(short.capitalize(), fontweight="bold")
        ax.set_xlabel("Importance")
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "feature_importance.png", dpi=150)
    plt.close()
    print("[SAVED] results/feature_importance.png")


# ── 4. Scatter plots ──────────────────────────────────────────────────────────

def plot_scatter(all_preds, y_te, best_model):
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    fig.suptitle(f"Predicted vs Actual  [{best_model}]", fontweight="bold")
    for ax, short, lbl in zip(axes, SHORT, LABEL_COLS):
        if best_model not in all_preds or short not in all_preds[best_model]:
            continue
        preds   = np.array(all_preds[best_model][short])
        actuals = y_te[lbl]
        valid   = ~np.isnan(actuals)
        p, a    = preds[valid], actuals[valid]
        ax.scatter(a, p, alpha=0.4, color=COLORS[short], s=20, edgecolors="none")
        mn, mx = min(a.min(), p.min()), max(a.max(), p.max())
        ax.plot([mn, mx], [mn, mx], "k--", lw=1)
        ax.set_title(f"{short.capitalize()}  MAE={mean_absolute_error(a,p):.2f}",
                     fontweight="bold")
        ax.set_xlabel("Actual"); ax.set_ylabel("Predicted")
        ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "scatter.png", dpi=150)
    plt.close()
    print("[SAVED] results/scatter.png")


# ── 5. Stability vs MAE ───────────────────────────────────────────────────────

def plot_stability(all_preds, test_meta, y_te, best_model):
    if best_model not in all_preds:
        return
    pids = test_meta["uid"].values
    rows = []
    for pid in np.unique(pids):
        mask = pids == pid
        for short, lbl in zip(SHORT, LABEL_COLS):
            if short not in all_preds[best_model]:
                continue
            p = np.array(all_preds[best_model][short])[mask]
            a = y_te[lbl][mask]
            valid = ~np.isnan(a)
            if valid.sum() < 2:
                continue
            rows.append({"uid": pid, "target": short,
                         "mae": mean_absolute_error(a[valid], p[valid]),
                         "label_std": a[valid].std()})
    df = pd.DataFrame(rows)
    df.to_csv(RESULTS_DIR / "per_participant_mae.csv", index=False)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    fig.suptitle("Label Stability vs MAE (stable users → lower error)",
                 fontweight="bold")
    for ax, short in zip(axes, SHORT):
        sub = df[df["target"] == short]
        ax.scatter(sub["label_std"], sub["mae"],
                   alpha=0.7, color=COLORS[short], edgecolors="none")
        if len(sub) > 2:
            z  = np.polyfit(sub["label_std"], sub["mae"], 1)
            xs = np.linspace(sub["label_std"].min(), sub["label_std"].max(), 50)
            ax.plot(xs, np.poly1d(z)(xs), "k--", lw=1.5)
        ax.set_xlabel("Std Dev of Label"); ax.set_ylabel("Per-Participant MAE")
        ax.set_title(short.capitalize(), fontweight="bold"); ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "stability_vs_mae.png", dpi=150)
    plt.close()
    print("[SAVED] results/stability_vs_mae.png")


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    X_tr, X_te, y_tr, y_te, feat_cols, results, all_preds, test_meta = load_all()

    comparison_table(results)

    best_model = min(results, key=lambda m: results[m].get("stress", 999))
    print(f"\nBest model (stress MAE): {best_model}")

    plot_feature_importance(feat_cols)
    plot_scatter(all_preds, y_te, best_model)

    print("\n── Transfer Learning Simulation ──")
    simulate_transfer_learning(X_tr, y_tr, X_te, y_te)

    plot_stability(all_preds, test_meta, y_te, best_model)

    print(f"\nAll outputs in: results/")
