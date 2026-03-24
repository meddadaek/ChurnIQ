"""
ChurnIQ — bootstrap_models.py
==============================
Trains the EXACT two-stage pipeline from the Kaggle notebook:
  Stage 1 : Ridge  (captures linear patterns)
  Stage 2 : XGBoost  (non-linear corrections + ridge_pred as feature)

Feature pipeline (7 steps from notebook):
  1. Frequency Encoding
  2. Arithmetic Interactions
  3. Service Counts
  4. ORIG_proba  (target-probability from reference data)
  5. Distribution Features  (percentile ranks, z-scores vs churner/non-churner)
  6. Quantile Distance Features
  7. Digit Features  +  Bi-gram / Tri-gram N-gram Target Encoding

Saved artifacts in models/:
  pipeline_config.pkl    all encoding look-up tables + scaler
  ridge_final.pkl        final Ridge model
  xgb_final.pkl          final XGBoost model
  features_numeric.pkl   numeric feature column list (for SHAP)
  background.csv         200 encoded rows for SHAP background

Run once before starting the server:
    python bootstrap_models.py
"""
from __future__ import annotations

import pickle, warnings, time
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

MODEL_DIR = Path("models")
MODEL_DIR.mkdir(parents=True, exist_ok=True)

# ─────────────────────────────────────────────────────────────
# Constants  (identical in app.py)
# ─────────────────────────────────────────────────────────────
RANDOM_SEED  = 42
N_FOLDS      = 20
INNER_FOLDS  = 5
RIDGE_ALPHA  = 10.0

CATS = [
    "gender", "SeniorCitizen", "Partner", "Dependents", "PhoneService",
    "MultipleLines", "InternetService", "OnlineSecurity", "OnlineBackup",
    "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies",
    "Contract", "PaperlessBilling", "PaymentMethod",
]
NUMS = ["tenure", "MonthlyCharges", "TotalCharges"]
SERVICE_COLS = [
    "PhoneService", "MultipleLines", "OnlineSecurity", "OnlineBackup",
    "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies",
]
TOP_CATS_FOR_NGRAM = [
    "Contract", "InternetService", "PaymentMethod",
    "OnlineSecurity", "TechSupport", "PaperlessBilling",
]
TOP4         = TOP_CATS_FOR_NGRAM[:4]
STATS        = ["std", "min", "max"]
BIGRAM_COLS  = [f"BG_{c1}_{c2}"      for c1, c2     in combinations(TOP_CATS_FOR_NGRAM, 2)]
TRIGRAM_COLS = [f"TG_{c1}_{c2}_{c3}" for c1, c2, c3 in combinations(TOP4, 3)]
NGRAM_COLS   = BIGRAM_COLS + TRIGRAM_COLS
NUM_AS_CAT   = [f"CAT_{c}" for c in NUMS]
TE_COLUMNS       = NUM_AS_CAT + CATS
TE_NGRAM_COLUMNS = NGRAM_COLS

XGB_PARAMS = dict(
    n_estimators       = 3000,
    learning_rate      = 0.02,
    max_depth          = 5,
    subsample          = 0.81,
    colsample_bytree   = 0.82,
    min_child_weight   = 4,
    reg_alpha          = 1.5,
    reg_lambda         = 1.3,
    gamma              = 0.4,
    eval_metric        = "auc",
    random_state       = RANDOM_SEED,
    early_stopping_rounds = 150,
    verbosity          = 0,
)

# ─────────────────────────────────────────────────────────────
# Helper functions
# ─────────────────────────────────────────────────────────────
def pctrank_against(values: np.ndarray, reference: np.ndarray) -> np.ndarray:
    ref_sorted = np.sort(reference)
    return (np.searchsorted(ref_sorted, values) / len(ref_sorted)).astype("float32")

def zscore_against(values: np.ndarray, reference: np.ndarray) -> np.ndarray:
    mu, sigma = np.mean(reference), np.std(reference)
    if sigma == 0:
        return np.zeros(len(values), dtype="float32")
    return ((values - mu) / sigma).astype("float32")

# ─────────────────────────────────────────────────────────────
# Feature engineering  (7 steps)
# ─────────────────────────────────────────────────────────────
def engineer_features(df_in: pd.DataFrame, orig_ref: pd.DataFrame) -> pd.DataFrame:
    df = df_in.copy()

    # 1. Frequency Encoding
    for col in NUMS:
        freq = pd.concat([orig_ref[col], df[col]]).value_counts(normalize=True)
        df[f"FREQ_{col}"] = df[col].map(freq).fillna(0).astype("float32")

    # 2. Arithmetic Interactions
    df["charges_deviation"]      = (df["TotalCharges"] - df["tenure"] * df["MonthlyCharges"]).astype("float32")
    df["monthly_to_total_ratio"] = (df["MonthlyCharges"] / (df["TotalCharges"] + 1)).astype("float32")
    df["avg_monthly_charges"]    = (df["TotalCharges"] / (df["tenure"] + 1)).astype("float32")

    # 3. Service Counts
    df["service_count"] = (df[SERVICE_COLS] == "Yes").sum(axis=1).astype("float32")
    df["has_internet"]  = (df["InternetService"] != "No").astype("float32")
    df["has_phone"]     = (df["PhoneService"] == "Yes").astype("float32")

    # 4. ORIG_proba
    for col in CATS + NUMS:
        tmp = orig_ref.groupby(col)["Churn"].mean()
        df[f"ORIG_proba_{col}"] = df[col].map(tmp).fillna(0.5).astype("float32")

    # 5. Distribution Features
    orig_ch_tc  = orig_ref.loc[orig_ref["Churn"] == 1, "TotalCharges"].values
    orig_nc_tc  = orig_ref.loc[orig_ref["Churn"] == 0, "TotalCharges"].values
    orig_all_tc = orig_ref["TotalCharges"].values
    orig_is_mc  = orig_ref.groupby("InternetService")["MonthlyCharges"].mean()

    tc = df["TotalCharges"].values
    df["pctrank_nonchurner_TC"] = pctrank_against(tc, orig_nc_tc)
    df["pctrank_churner_TC"]    = pctrank_against(tc, orig_ch_tc)
    df["pctrank_orig_TC"]       = pctrank_against(tc, orig_all_tc)
    df["zscore_churn_gap_TC"]   = (np.abs(zscore_against(tc, orig_ch_tc)) -
                                    np.abs(zscore_against(tc, orig_nc_tc))).astype("float32")
    df["zscore_nonchurner_TC"]  = zscore_against(tc, orig_nc_tc)
    df["pctrank_churn_gap_TC"]  = (pctrank_against(tc, orig_ch_tc) -
                                    pctrank_against(tc, orig_nc_tc)).astype("float32")
    df["resid_IS_MC"]           = (df["MonthlyCharges"] -
                                    df["InternetService"].map(orig_is_mc).fillna(0)).astype("float32")

    df["cond_pctrank_IS_TC"] = 0.0
    for val in orig_ref["InternetService"].unique():
        m = df["InternetService"] == val
        ref = orig_ref.loc[orig_ref["InternetService"] == val, "TotalCharges"].values
        if len(ref) > 0 and m.sum() > 0:
            df.loc[m, "cond_pctrank_IS_TC"] = pctrank_against(df.loc[m, "TotalCharges"].values, ref)

    df["cond_pctrank_C_TC"] = 0.0
    for val in orig_ref["Contract"].unique():
        m = df["Contract"] == val
        ref = orig_ref.loc[orig_ref["Contract"] == val, "TotalCharges"].values
        if len(ref) > 0 and m.sum() > 0:
            df.loc[m, "cond_pctrank_C_TC"] = pctrank_against(df.loc[m, "TotalCharges"].values, ref)

    # 6. Quantile Distance
    for q_label, q_val in [("q25", 0.25), ("q50", 0.50), ("q75", 0.75)]:
        ch_q = np.quantile(orig_ch_tc, q_val)
        nc_q = np.quantile(orig_nc_tc, q_val)
        df[f"dist_To_ch_{q_label}"]   = np.abs(df["TotalCharges"] - ch_q).astype("float32")
        df[f"dist_To_nc_{q_label}"]   = np.abs(df["TotalCharges"] - nc_q).astype("float32")
        df[f"qdist_gap_To_{q_label}"] = (df[f"dist_To_nc_{q_label}"] -
                                          df[f"dist_To_ch_{q_label}"]).astype("float32")

    # 7. Digit Features
    t_str = df["tenure"].astype(str)
    df["tenure_first_digit"] = t_str.str[0].astype(int)
    df["tenure_last_digit"]  = t_str.str[-1].astype(int)
    df["tenure_mod10"]  = df["tenure"] % 10
    df["tenure_mod12"]  = df["tenure"] % 12

    df["mc_first_digit"] = df["MonthlyCharges"].astype(str).str[0].astype(int)
    df["mc_mod10"]       = np.floor(df["MonthlyCharges"]) % 10
    df["mc_fractional"]  = (df["MonthlyCharges"] - np.floor(df["MonthlyCharges"])).astype("float32")

    df["tc_mod100"]     = np.floor(df["TotalCharges"]) % 100
    df["tc_fractional"] = (df["TotalCharges"] - np.floor(df["TotalCharges"])).astype("float32")

    # N-gram composite strings
    for c1, c2 in combinations(TOP_CATS_FOR_NGRAM, 2):
        df[f"BG_{c1}_{c2}"] = df[c1].astype(str) + "_" + df[c2].astype(str)
    for c1, c2, c3 in combinations(TOP4, 3):
        df[f"TG_{c1}_{c2}_{c3}"] = (df[c1].astype(str) + "_" +
                                      df[c2].astype(str) + "_" +
                                      df[c3].astype(str))
    # Num-as-Cat strings
    for col in NUMS:
        df[f"CAT_{col}"] = df[col].astype(str)

    return df

# ─────────────────────────────────────────────────────────────
# Target encoding helper
# ─────────────────────────────────────────────────────────────
def apply_te(X_tr: pd.DataFrame, y_tr: np.ndarray,
             X_val: pd.DataFrame, skf_inner: StratifiedKFold
             ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Inner-fold TE for cats + n-grams → numeric-only DataFrames."""
    X_tr  = X_tr.copy()
    X_val = X_val.copy()

    te_buf = {f"TE1_{col}_{s}": np.zeros(len(X_tr))
              for col in TE_COLUMNS for s in STATS}
    for in_tr_idx, in_va_idx in skf_inner.split(X_tr, y_tr):
        X_tr2 = X_tr.iloc[in_tr_idx]
        y_s   = pd.Series(y_tr[in_tr_idx], index=X_tr2.index)
        for col in TE_COLUMNS:
            grp = pd.concat([X_tr2[col], y_s], axis=1)
            grp.columns = [col, "y"]
            stats = grp.groupby(col)["y"].agg(STATS)
            for s in STATS:
                mapped = X_tr.iloc[in_va_idx][col].map(stats[s]).fillna(0).values.astype("float32")
                te_buf[f"TE1_{col}_{s}"][in_va_idx] = mapped
    for k, v in te_buf.items():
        X_tr[k] = v.astype("float32")

    y_tr_s = pd.Series(y_tr, index=X_tr.index)
    for col in TE_COLUMNS:
        grp = pd.concat([X_tr[col], y_tr_s], axis=1)
        grp.columns = [col, "y"]
        stats = grp.groupby(col)["y"].agg(STATS)
        for s in STATS:
            X_val[f"TE1_{col}_{s}"] = X_val[col].map(stats[s]).fillna(0).astype("float32")

    ng_buf = {f"TE_ng_{col}": np.full(len(X_tr), 0.5) for col in TE_NGRAM_COLUMNS}
    for in_tr_idx, in_va_idx in skf_inner.split(X_tr, y_tr):
        X_tr2 = X_tr.iloc[in_tr_idx]
        y_s   = pd.Series(y_tr[in_tr_idx], index=X_tr2.index)
        for col in TE_NGRAM_COLUMNS:
            grp = pd.concat([X_tr2[col], y_s], axis=1)
            grp.columns = [col, "y"]
            ng_te  = grp.groupby(col)["y"].mean()
            mapped = X_tr.iloc[in_va_idx][col].map(ng_te).fillna(0.5).values.astype("float32")
            ng_buf[f"TE_ng_{col}"][in_va_idx] = mapped
    for k, v in ng_buf.items():
        X_tr[k] = v.astype("float32")

    for col in TE_NGRAM_COLUMNS:
        grp = pd.concat([X_tr[col], y_tr_s], axis=1)
        grp.columns = [col, "y"]
        ng_te = grp.groupby(col)["y"].mean()
        X_val[f"TE_ng_{col}"] = X_val[col].map(ng_te).fillna(0.5).astype("float32")

    drop = set(TE_COLUMNS + TE_NGRAM_COLUMNS + CATS + NUMS + ["Churn"])
    X_tr_n  = X_tr.drop(columns=[c for c in drop if c in X_tr.columns],  errors="ignore").select_dtypes(include=[np.number]).fillna(0)
    X_val_n = X_val.drop(columns=[c for c in drop if c in X_val.columns], errors="ignore").select_dtypes(include=[np.number]).fillna(0)
    common  = [c for c in X_tr_n.columns if c in X_val_n.columns]
    return X_tr_n[common], X_val_n[common]

# ─────────────────────────────────────────────────────────────
# Dataset generator  (realistic Telco churn)
# ─────────────────────────────────────────────────────────────
def generate_dataset(n: int = 7043, seed: int = RANDOM_SEED) -> pd.DataFrame:
    np.random.seed(seed)
    def cat(ch, sz, p=None): return np.random.choice(ch, sz, p=p)
    tenure_raw  = np.random.randint(0, 73, n)
    monthly_raw = np.random.uniform(18, 119, n).round(2)
    internet    = cat(["DSL", "Fiber optic", "No"], n, p=[0.34, 0.44, 0.22])
    contracts   = cat(["Month-to-month", "One year", "Two year"], n, p=[0.55, 0.21, 0.24])
    payment     = cat(["Electronic check", "Mailed check",
                        "Bank transfer (automatic)", "Credit card (automatic)"], n)
    online_sec  = cat(["Yes", "No", "No internet service"], n, p=[0.28, 0.50, 0.22])
    tech_supp   = cat(["Yes", "No", "No internet service"], n, p=[0.29, 0.49, 0.22])
    df = pd.DataFrame({
        "gender":           cat(["Male", "Female"], n),
        "SeniorCitizen":    np.random.choice([0, 1], n, p=[0.84, 0.16]),
        "Partner":          cat(["Yes", "No"], n),
        "Dependents":       cat(["Yes", "No"], n, p=[0.30, 0.70]),
        "tenure":           tenure_raw,
        "PhoneService":     cat(["Yes", "No"], n, p=[0.90, 0.10]),
        "MultipleLines":    cat(["Yes", "No", "No phone service"], n, p=[0.42, 0.48, 0.10]),
        "InternetService":  internet,
        "OnlineSecurity":   online_sec,
        "OnlineBackup":     cat(["Yes", "No", "No internet service"], n, p=[0.34, 0.44, 0.22]),
        "DeviceProtection": cat(["Yes", "No", "No internet service"], n, p=[0.34, 0.44, 0.22]),
        "TechSupport":      tech_supp,
        "StreamingTV":      cat(["Yes", "No", "No internet service"], n, p=[0.38, 0.40, 0.22]),
        "StreamingMovies":  cat(["Yes", "No", "No internet service"], n, p=[0.39, 0.39, 0.22]),
        "Contract":         contracts,
        "PaperlessBilling": cat(["Yes", "No"], n, p=[0.59, 0.41]),
        "PaymentMethod":    payment,
        "MonthlyCharges":   monthly_raw,
        "TotalCharges":     (tenure_raw * monthly_raw + np.random.uniform(-50, 50, n)).clip(0).round(2),
    })
    churn_p = (
        0.05 + 0.22*(df["Contract"]=="Month-to-month") + 0.14*(df["InternetService"]=="Fiber optic")
        - 0.18*(df["tenure"]/72) + 0.10*(df["PaymentMethod"]=="Electronic check")
        + 0.08*(df["PaperlessBilling"]=="Yes") + 0.06*(df["SeniorCitizen"]==1)
        - 0.12*(df["OnlineSecurity"]=="Yes") - 0.12*(df["TechSupport"]=="Yes")
        + 0.05*(df["MultipleLines"]=="Yes") - 0.04*(df["Partner"]=="Yes")
        + np.random.normal(0, 0.04, n)
    ).clip(0.02, 0.95)
    df["Churn"] = np.random.binomial(1, churn_p).astype(int)
    return df

# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────
def main() -> None:
    t0 = time.time()
    print("\n" + "="*65)
    print("  ChurnIQ — Pipeline Training")
    print("  Two-Stage Ridge → XGBoost  |  N-gram Target Encoding")
    print("="*65)

    print("\n[1/5] Generating dataset…")
    df_all   = generate_dataset()
    orig_ref, train_raw = train_test_split(
        df_all, test_size=0.80, stratify=df_all["Churn"], random_state=RANDOM_SEED)
    print(f"      orig_ref: {len(orig_ref)}  |  train: {len(train_raw)}  |  churn: {train_raw['Churn'].mean():.3f}")

    print("\n[2/5] Engineering features…")
    train_fe = engineer_features(train_raw, orig_ref)
    y        = train_fe["Churn"].values
    print(f"      Engineered shape: {train_fe.shape}")

    skf    = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_SEED)
    skf_in = StratifiedKFold(n_splits=INNER_FOLDS, shuffle=True, random_state=RANDOM_SEED)

    ridge_oof   = np.zeros(len(train_fe))
    xgb_oof     = np.zeros(len(train_fe))
    fold_scores = []
    best_iters  = []

    print(f"\n[3/5] {N_FOLDS}-fold cross-validation…")
    for i, (tr_idx, va_idx) in enumerate(skf.split(train_fe, y)):
        X_tr_fe = train_fe.iloc[tr_idx].copy()
        X_va_fe = train_fe.iloc[va_idx].copy()
        y_tr = y[tr_idx]; y_va = y[va_idx]

        X_tr_n, X_va_n = apply_te(X_tr_fe, y_tr, X_va_fe, skf_in)

        # Stage 1 — Ridge
        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_tr_n); X_va_s = scaler.transform(X_va_n)
        ridge  = Ridge(alpha=RIDGE_ALPHA)
        ridge.fit(X_tr_s, y_tr)
        rp_tr  = 1 / (1 + np.exp(-ridge.predict(X_tr_s)))
        rp_va  = 1 / (1 + np.exp(-ridge.predict(X_va_s)))
        ridge_oof[va_idx] = rp_va

        # Stage 2 — XGBoost
        X_tr_x = X_tr_n.copy(); X_tr_x["ridge_pred"] = rp_tr
        X_va_x = X_va_n.copy(); X_va_x["ridge_pred"] = rp_va
        xgb_m  = XGBClassifier(**XGB_PARAMS)
        xgb_m.fit(X_tr_x, y_tr, eval_set=[(X_va_x, y_va)], verbose=False)
        preds  = xgb_m.predict_proba(X_va_x)[:, 1]
        xgb_oof[va_idx] = preds
        fold_auc = roc_auc_score(y_va, preds)
        fold_scores.append(fold_auc)
        best_iters.append(xgb_m.best_iteration)
        print(f"      Fold {i+1:2d}/{N_FOLDS}: Ridge={roc_auc_score(y_va,rp_va):.4f} | XGB={fold_auc:.5f} | best_iter={xgb_m.best_iteration}")

    ridge_auc = roc_auc_score(y, ridge_oof)
    xgb_auc   = roc_auc_score(y, xgb_oof)
    print(f"\n      Ridge overall AUC : {ridge_auc:.5f}")
    print(f"      XGB   overall AUC : {xgb_auc:.5f}  ±{np.std(fold_scores):.5f}")
    print(f"      Ridge → XGB lift  : {xgb_auc-ridge_auc:+.5f}")

    print("\n[4/5] Training final models on full data…")
    X_all_n, _ = apply_te(train_fe, y, train_fe.head(1).copy(), skf_in)
    scaler_f   = StandardScaler()
    X_all_s    = scaler_f.fit_transform(X_all_n)
    ridge_f    = Ridge(alpha=RIDGE_ALPHA)
    ridge_f.fit(X_all_s, y)
    rp_all     = 1 / (1 + np.exp(-ridge_f.predict(X_all_s)))
    X_all_x    = X_all_n.copy(); X_all_x["ridge_pred"] = rp_all
    n_est_final = int(np.mean(best_iters) * 1.05) + 50
    xgb_params_final = {k: v for k, v in XGB_PARAMS.items()
                        if k not in ("early_stopping_rounds", "n_estimators")}
    xgb_f = XGBClassifier(n_estimators=n_est_final, **xgb_params_final)
    xgb_f.fit(X_all_x, y, verbose=False)
    numeric_cols = list(X_all_x.columns)
    print(f"      Final XGB n_estimators: {n_est_final}")

    print("\n[5/5] Building inference config + saving…")
    # Compute TE look-ups from full training data
    train_fe2 = engineer_features(train_raw, orig_ref)
    y2s = pd.Series(train_fe2["Churn"].values, index=train_fe2.index)

    te_stats: dict = {}
    for col in TE_COLUMNS:
        grp = pd.concat([train_fe2[col], y2s], axis=1)
        grp.columns = [col, "y"]
        te_stats[col] = {s: grp.groupby(col)["y"].agg(s).to_dict() for s in STATS}

    ng_te_stats: dict = {}
    for col in TE_NGRAM_COLUMNS:
        grp = pd.concat([train_fe2[col], y2s], axis=1)
        grp.columns = [col, "y"]
        ng_te_stats[col] = grp.groupby(col)["y"].mean().to_dict()

    dist_stats = {
        "orig_ch_tc":  orig_ref.loc[orig_ref["Churn"]==1, "TotalCharges"].values,
        "orig_nc_tc":  orig_ref.loc[orig_ref["Churn"]==0, "TotalCharges"].values,
        "orig_all_tc": orig_ref["TotalCharges"].values,
        "orig_is_mc":  orig_ref.groupby("InternetService")["MonthlyCharges"].mean().to_dict(),
        "orig_ref":    orig_ref,
        "freq": {col: pd.concat([orig_ref[col], train_raw[col]]).value_counts(normalize=True).to_dict()
                 for col in NUMS},
    }
    for col in CATS + NUMS:
        dist_stats[f"orig_proba_{col}"] = orig_ref.groupby(col)["Churn"].mean().to_dict()

    pipeline_config = {
        "te_stats":     te_stats,
        "ng_te_stats":  ng_te_stats,
        "dist_stats":   dist_stats,
        "scaler":       scaler_f,
        "numeric_cols": numeric_cols,
        "cv_ridge_auc": ridge_auc,
        "cv_xgb_auc":   xgb_auc,
        "cv_xgb_std":   float(np.std(fold_scores)),
        "fold_scores":  fold_scores,
        "n_folds":      N_FOLDS,
    }

    with open(MODEL_DIR / "pipeline_config.pkl", "wb") as f: pickle.dump(pipeline_config, f)
    with open(MODEL_DIR / "ridge_final.pkl",      "wb") as f: pickle.dump(ridge_f, f)
    with open(MODEL_DIR / "xgb_final.pkl",        "wb") as f: pickle.dump(xgb_f, f)
    with open(MODEL_DIR / "features_numeric.pkl", "wb") as f: pickle.dump(numeric_cols, f)

    # Background for SHAP
    bg_raw  = train_raw.sample(min(200, len(train_raw)), random_state=42)
    bg_fe   = engineer_features(bg_raw, orig_ref)
    bg_y    = bg_fe["Churn"].values
    for col in TE_COLUMNS:
        for s in STATS:
            bg_fe[f"TE1_{col}_{s}"] = bg_fe[col].map(te_stats[col][s]).fillna(0).astype("float32")
    for col in TE_NGRAM_COLUMNS:
        bg_fe[f"TE_ng_{col}"] = bg_fe[col].map(ng_te_stats[col]).fillna(0.5).astype("float32")
    drop = set(TE_COLUMNS + TE_NGRAM_COLUMNS + CATS + NUMS + ["Churn"])
    bg_num = bg_fe.drop(columns=[c for c in drop if c in bg_fe.columns], errors="ignore") \
                  .select_dtypes(include=[np.number]).fillna(0)
    bg_xgb = bg_num[[c for c in numeric_cols if c in bg_num.columns]].fillna(0).copy()
    bg_xgb["Churn"] = bg_y
    bg_xgb.to_csv(MODEL_DIR / "background.csv", index=False)

    elapsed = time.time() - t0
    print(f"\n{'='*65}")
    print(f"  Done in {elapsed:.0f}s")
    print(f"  Ridge CV AUC  : {ridge_auc:.5f}")
    print(f"  XGB   CV AUC  : {xgb_auc:.5f}  ±{np.std(fold_scores):.5f}")
    for i, (r_s, x_s) in enumerate(zip(
            [roc_auc_score(y[va] if False else y, ridge_oof) for _ in range(1)],
            fold_scores)):
        pass  # already printed per-fold above
    print(f"  Artifacts  →  {MODEL_DIR.resolve()}/")
    print(f"    pipeline_config.pkl  ridge_final.pkl  xgb_final.pkl")
    print(f"    features_numeric.pkl  background.csv")
    print(f"{'='*65}")
    print("  Run:  python app.py\n")


if __name__ == "__main__":
    main()
