"""
01_prepare_data.py
==================
Loads the real StudentLife dataset and builds a daily feature matrix.

Labels (targets, 0-100 scaled):
  - stress  : EMA Stress.level       (1=great → 5=stressed out, inverted)
  - mood    : EMA Behavior (calm/enthusiastic vs anxious)
  - health  : EMA Sleep (quality + hours)

Features (passive sensing, aggregated per user per day):
  - call_log, sms, wifi, app_usage

Set DATASET_DIR to your local path before running.

Requirements:
    pip install pandas numpy scikit-learn joblib matplotlib seaborn
"""

import numpy as np
import pandas as pd
from pathlib import Path

# ── CONFIG ────────────────────────────────────────────────────────────────────
DATASET_DIR    = Path(r"C:\Users\vipul\Documents\AML projecct\dataset")
OUT_DIR        = Path("data/processed")
STUDY_START_TS = 1362096000   # Mar 1 2013 UTC — anchors day indices
WINDOW         = 7            # rolling window size (days), matches paper
# ─────────────────────────────────────────────────────────────────────────────

OUT_DIR.mkdir(parents=True, exist_ok=True)
LABEL_COLS = ["stress", "mood", "health"]


def ts_to_day(ts_series):
    return ((ts_series.astype(float) - STUDY_START_TS) // 86400).astype(int)


# ─── EMA Labels ──────────────────────────────────────────────────────────────

def read_ema_json(path: Path) -> pd.DataFrame:
    """Load an EMA JSON file (list of dicts) into a DataFrame, dropping
    records that have no useful response fields (only null/location)."""
    import json
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    df = pd.DataFrame(data)
    # Drop rows where every column except resp_time/location/null is NaN
    return df


def load_ema_labels() -> pd.DataFrame:
    """
    Real structure:
      EMA/Stress/Stress_u00.json, Stress_u01.json, …
      EMA/Sleep/Sleep_u00.json,   Sleep_u01.json,  …
      EMA/Behavior/Behavior_u00.json, …

    Each JSON is a list of dicts with at minimum:
      resp_time  (unix timestamp)
      level / hour / rate / calm / anxious / …  (may be absent in some records)
    """
    ema_dir     = DATASET_DIR / "EMA/response"
    all_records = []

    # ── Stress ──────────────────────────────────────────────────────────────
    for f in sorted((ema_dir / "Stress").glob("Stress_u*.json")):
        uid = f.stem.split("_")[-1]          # "u00"
        df  = read_ema_json(f)
        if "resp_time" not in df.columns or "level" not in df.columns:
            continue
        df = df.dropna(subset=["level"])
        df["level"] = pd.to_numeric(df["level"], errors="coerce")
        df = df.dropna(subset=["level"])
        df["day"] = ts_to_day(df["resp_time"])
        daily = df.groupby("day")["level"].mean().reset_index()
        # 1=a little stressed … 3=stressed out, 4=feeling good, 5=feeling great
        # Invert: higher value → more stressed
        daily["stress"] = (5 - daily["level"]) / 4 * 100
        daily["uid"] = uid
        all_records.append(daily[["uid", "day", "stress"]])

    # ── Health (Sleep proxy) ─────────────────────────────────────────────────
    sleep_dir = ema_dir / "Sleep"
    if sleep_dir.exists():
        for f in sorted(sleep_dir.glob("Sleep_u*.json")):
            uid = f.stem.split("_")[-1]
            df  = read_ema_json(f)
            if "resp_time" not in df.columns:
                continue
            df["day"] = ts_to_day(df["resp_time"])
            parts = []

            if "hour" in df.columns:
                hour_map = {1:2.5,2:3.5,3:4,4:4.5,5:5,6:5.5,7:6,8:6.5,
                            9:7,10:7.5,11:8,12:8.5,13:9,14:9.5,15:10,
                            16:10.5,17:11,18:11.5,19:12}
                df["hour"] = pd.to_numeric(df["hour"], errors="coerce")
                df["sleep_hrs_val"] = df["hour"].map(hour_map)
                h = df.groupby("day")["sleep_hrs_val"].mean()
                parts.append(((h - 2.5) / 9.5 * 100).rename("h_norm"))

            if "rate" in df.columns:
                df["rate"] = pd.to_numeric(df["rate"], errors="coerce")
                r = df.groupby("day")["rate"].mean()
                parts.append(((4 - r) / 3 * 100).rename("r_norm"))  # 1=very good

            if parts:
                combined = pd.concat(parts, axis=1).mean(axis=1).reset_index()
                combined.columns = ["day", "health"]
                combined["uid"] = uid
                all_records.append(combined)

    # ── Mood (Behavior EMA) ──────────────────────────────────────────────────
    behavior_dir = ema_dir / "Behavior"
    if behavior_dir.exists():
        for f in sorted(behavior_dir.glob("Behavior_u*.json")):
            uid = f.stem.split("_")[-1]
            df  = read_ema_json(f)
            if "resp_time" not in df.columns:
                continue
            df["day"] = ts_to_day(df["resp_time"])
            mood_parts = []
            for pos_col in ["calm", "enthusiastic", "sympathetic"]:
                if pos_col in df.columns:
                    mood_parts.append(
                        pd.to_numeric(df[pos_col], errors="coerce"))
            for neg_col in ["anxious", "critical", "disorganized"]:
                if neg_col in df.columns:
                    mood_parts.append(
                        6 - pd.to_numeric(df[neg_col], errors="coerce"))
            if mood_parts:
                df["mood_raw"] = pd.concat(mood_parts, axis=1).mean(axis=1)
                daily = df.groupby("day")["mood_raw"].mean().reset_index()
                daily["mood"] = (daily["mood_raw"] - 1) / 4 * 100
                daily["uid"]  = uid
                all_records.append(daily[["uid", "day", "mood"]])

    if not all_records:
        raise RuntimeError(
            "No EMA data found.\n"
            f"Looked in: {ema_dir}\n"
            "Expected: EMA/Stress/Stress_u00.json, EMA/Sleep/Sleep_u00.json, ..."
        )

    merged = None
    for col in LABEL_COLS:
        sub = (pd.concat([r for r in all_records if col in r.columns],
                         ignore_index=True)
                 .groupby(["uid", "day"])[col].mean().reset_index())
        merged = sub if merged is None else merged.merge(sub, on=["uid","day"], how="outer")

    print(f"[INFO] EMA labels: {len(merged)} rows, {merged['uid'].nunique()} participants")
    return merged


# ─── Sensing Features ─────────────────────────────────────────────────────────

def load_call_features() -> pd.DataFrame:
    rows = []
    for f in sorted((DATASET_DIR / "call_log").glob("call_log_u*.csv")):
        uid = f.stem.split("_")[-1]
        df  = pd.read_csv(f)
        ts_col  = next((c for c in df.columns if "time" in c.lower()), None)
        dur_col = next((c for c in df.columns if "duration" in c.lower()), None)
        if ts_col is None:
            continue
        df["day"] = ts_to_day(df[ts_col])
        agg = {"call_count": (ts_col, "count")}
        if dur_col:
            df[dur_col] = pd.to_numeric(df[dur_col], errors="coerce")
            agg["call_dur_total"] = (dur_col, "sum")
            agg["call_dur_mean"]  = (dur_col, "mean")
        daily = df.groupby("day").agg(**agg).reset_index()
        daily["uid"] = uid
        rows.append(daily)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def load_sms_features() -> pd.DataFrame:
    rows = []
    for f in sorted((DATASET_DIR / "sms").glob("sms_u*.csv")):
        uid = f.stem.split("_")[-1]
        df  = pd.read_csv(f)
        ts_col = next((c for c in df.columns if "time" in c.lower()), None)
        if ts_col is None:
            continue
        df["day"] = ts_to_day(df[ts_col])
        daily = df.groupby("day").size().reset_index(name="sms_count")
        daily["uid"] = uid
        rows.append(daily)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def load_wifi_features() -> pd.DataFrame:
    # Try sensing/wifi/ first, then flat sensing/
    for wifi_dir in [DATASET_DIR/"sensing"/"wifi", DATASET_DIR/"sensing"]:
        files = list(wifi_dir.glob("wifi_u*.csv")) if wifi_dir.exists() else []
        if files:
            break
    rows = []
    for f in sorted(files):
        uid = f.stem.split("_")[-1]
        df  = pd.read_csv(f)
        if "time" not in df.columns:
            continue
        df["day"] = ts_to_day(df["time"])
        agg = df.groupby("day").agg(
            wifi_unique_ap  =("BSSID",  "nunique"),
            wifi_level_mean =("level",  "mean"),
            wifi_level_std  =("level",  "std"),
            wifi_scan_count =("time",   "count"),
        ).reset_index()
        agg["uid"] = uid
        rows.append(agg)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def load_app_features() -> pd.DataFrame:
    rows = []
    for f in sorted((DATASET_DIR / "app_usage").glob("running_app_u*.csv")):
        uid = f.stem.replace("running_app_", "")
        df  = pd.read_csv(f)
        ts_col = next((c for c in df.columns if "time" in c.lower()), None)
        if ts_col is None:
            continue
        df["day"] = ts_to_day(df[ts_col])
        agg = df.groupby("day").agg(app_events=(ts_col, "count")).reset_index()
        pkg_col = next((c for c in df.columns
                        if "package" in c.lower() or "app" in c.lower()
                        and c != ts_col), None)
        if pkg_col:
            u = df.groupby("day")[pkg_col].nunique().reset_index()
            u.columns = ["day", "app_unique"]
            agg = agg.merge(u, on="day", how="left")
        agg["uid"] = uid
        rows.append(agg)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


# ─── Merge ────────────────────────────────────────────────────────────────────

def build_daily_df() -> pd.DataFrame:
    print("[1/5] Loading EMA labels …")
    df = load_ema_labels()
    for name, loader in [("call",  load_call_features),
                         ("sms",   load_sms_features),
                         ("wifi",  load_wifi_features),
                         ("apps",  load_app_features)]:
        print(f"[{name}] Loading {name} features …")
        feat = loader()
        if not feat.empty:
            df = df.merge(feat, on=["uid","day"], how="left")
    print(f"[INFO] Daily df: {df.shape}")
    return df


# ─── Supervised (next-day prediction, 7-day window) ───────────────────────────

SENSOR_COLS = ["call_count","call_dur_total","call_dur_mean",
               "sms_count",
               "wifi_unique_ap","wifi_level_mean","wifi_level_std","wifi_scan_count",
               "app_events","app_unique"]


def build_supervised(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    feat_cols = [c for c in SENSOR_COLS if c in df.columns]

    for uid, grp in df.groupby("uid"):
        grp = grp.sort_values("day").reset_index(drop=True)
        for i in range(WINDOW, len(grp) - 1):
            win    = grp.iloc[i - WINDOW: i]
            target = grp.iloc[i + 1]
            row    = {"uid": uid, "day": int(grp.iloc[i]["day"])}

            for col in feat_cols:
                vals = win[col].dropna().values
                row[f"{col}_mean"] = np.nanmean(vals) if len(vals) else np.nan
                row[f"{col}_std"]  = np.nanstd(vals)  if len(vals) else np.nan
                row[f"{col}_last"] = float(vals[-1])   if len(vals) else np.nan

            for lbl in LABEL_COLS:
                row[f"lag_{lbl}"] = (float(grp.iloc[i][lbl])
                                     if pd.notna(grp.iloc[i][lbl]) else np.nan)
            for lbl in LABEL_COLS:
                row[f"target_{lbl}"] = (float(target[lbl])
                                        if pd.notna(target[lbl]) else np.nan)
            rows.append(row)

    result = pd.DataFrame(rows)
    result = result.dropna(subset=[f"target_{l}" for l in LABEL_COLS], how="all")
    print(f"[INFO] Supervised: {len(result)} samples × {result.shape[1]} cols")
    return result


if __name__ == "__main__":
    daily = build_daily_df()
    daily.to_csv(OUT_DIR / "raw_daily.csv", index=False)
    print(f"[SAVED] data/processed/raw_daily.csv")

    sup = build_supervised(daily)
    sup.to_csv(OUT_DIR / "supervised.csv", index=False)
    print(f"[SAVED] data/processed/supervised.csv")
