"""
ChurnIQ — app.py
=================
Flask backend for the ChurnIQ dashboard.
Uses the EXACT same two-stage Ridge → XGBoost pipeline from the Kaggle notebook,
including the full 7-step feature engineering and N-gram target encoding at inference time.

Start:
    python app.py
"""

import os, json, re, traceback, warnings
warnings.filterwarnings("ignore")

from dotenv import load_dotenv
load_dotenv()  # Load from .env file

import numpy as np
import pandas as pd
import pickle
import shap
from itertools import combinations
from pathlib import Path
from flask import Flask, jsonify, request, send_file
from flask_cors import CORS

app = Flask(__name__, static_folder=".", static_url_path="")
CORS(app)

# ─────────────────────────────────────────────────────────────
# Groq Client
# ─────────────────────────────────────────────────────────────
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "").strip()
groq_client = None
try:
    from groq import Groq
    if GROQ_API_KEY:
        groq_client = Groq(api_key=GROQ_API_KEY)
        print("[Groq] Client initialised")
    else:
        print("[Groq] No API key set — using rule-based fallback")
        print("[Groq] Set GROQ_API_KEY environment variable to enable LLM strategies")
except Exception as e:
    print(f"[Groq] Unavailable ({e}) — using rule-based fallback")

# ─────────────────────────────────────────────────────────────
# Constants  (must stay in sync with bootstrap_models.py)
# ─────────────────────────────────────────────────────────────
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

# ─────────────────────────────────────────────────────────────
# Load models + pipeline config
# ─────────────────────────────────────────────────────────────
MODEL_DIR      = Path("models")
ridge_model    = None
xgb_model      = None
pipeline_cfg   = None
shap_explainer = None
background_df  = None
NUMERIC_COLS   = None

try:
    with open(MODEL_DIR / "pipeline_config.pkl", "rb") as f: pipeline_cfg   = pickle.load(f)
    with open(MODEL_DIR / "ridge_final.pkl",      "rb") as f: ridge_model    = pickle.load(f)
    with open(MODEL_DIR / "xgb_final.pkl",        "rb") as f: xgb_model      = pickle.load(f)
    with open(MODEL_DIR / "features_numeric.pkl", "rb") as f: NUMERIC_COLS   = pickle.load(f)
    background_df  = pd.read_csv(MODEL_DIR / "background.csv")
    shap_explainer = shap.TreeExplainer(xgb_model)
    print(f"[OK] Models loaded — {len(NUMERIC_COLS)} features | "
          f"Ridge CV AUC {pipeline_cfg['cv_ridge_auc']:.5f} | "
          f"XGB CV AUC {pipeline_cfg['cv_xgb_auc']:.5f}")
except Exception as e:
    print(f"[ERROR] Model load failed: {e}")
    traceback.print_exc()
    print("  → Run  python bootstrap_models.py  first")

# ─────────────────────────────────────────────────────────────
# SHAP interpretations
# ─────────────────────────────────────────────────────────────
SHAP_INTERP = {
    "Contract":         "Month-to-month = no switching cost, strongest churn signal",
    "TechSupport":      "Active tech support strongly reduces dissatisfaction",
    "InternetService":  "Fiber optic customers face more competitive alternatives",
    "PaperlessBilling": "Paperless users correlate with higher churn propensity",
    "MonthlyCharges":   "Higher charges signal higher-value, stickier customers",
    "OnlineSecurity":   "Absence of security bundle reduces perceived lock-in",
    "SeniorCitizen":    "Senior customers have slightly elevated churn risk",
    "PaymentMethod":    "Electronic check users show higher billing friction",
    "TotalCharges":     "Higher total spend signals a longer committed relationship",
    "tenure":           "Longer tenure = stronger loyalty, lower churn likelihood",
    "MultipleLines":    "Multiple lines adds product lock-in",
    "gender":           "Minimal independent predictive signal",
    "Partner":          "Customers with partners churn slightly less",
    "Dependents":       "Dependents modestly reduce churn likelihood",
    "OnlineBackup":     "Backup service adds subscription stickiness",
    "StreamingTV":      "Streaming bundle adds minor lock-in",
    "StreamingMovies":  "Streaming bundle adds minor lock-in",
    "DeviceProtection": "Device protection adds minor stickiness",
    "PhoneService":     "Phone service has minimal independent churn signal",
}

# ─────────────────────────────────────────────────────────────
# Helper functions  (match bootstrap_models.py exactly)
# ─────────────────────────────────────────────────────────────
def pctrank_against(values: np.ndarray, reference: np.ndarray) -> np.ndarray:
    ref_sorted = np.sort(reference)
    return (np.searchsorted(ref_sorted, values) / len(ref_sorted)).astype("float32")

def zscore_against(values: np.ndarray, reference: np.ndarray) -> np.ndarray:
    mu, sigma = np.mean(reference), np.std(reference)
    if sigma == 0:
        return np.zeros(len(values), dtype="float32")
    return ((values - mu) / sigma).astype("float32")


def engineer_features_single(raw: dict) -> pd.DataFrame:
    """Apply the full 7-step pipeline to ONE customer dict → returns a 1-row DataFrame."""
    df = pd.DataFrame([raw])
    ds = pipeline_cfg["dist_stats"]

    # 1. Frequency Encoding
    for col in NUMS:
        df[f"FREQ_{col}"] = df[col].map(ds["freq"][col]).fillna(0).astype("float32")

    # 2. Arithmetic
    df["charges_deviation"]      = (df["TotalCharges"] - df["tenure"] * df["MonthlyCharges"]).astype("float32")
    df["monthly_to_total_ratio"] = (df["MonthlyCharges"] / (df["TotalCharges"] + 1)).astype("float32")
    df["avg_monthly_charges"]    = (df["TotalCharges"] / (df["tenure"] + 1)).astype("float32")

    # 3. Service counts
    df["service_count"] = (df[SERVICE_COLS] == "Yes").sum(axis=1).astype("float32")
    df["has_internet"]  = (df["InternetService"] != "No").astype("float32")
    df["has_phone"]     = (df["PhoneService"] == "Yes").astype("float32")

    # 4. ORIG_proba
    for col in CATS + NUMS:
        df[f"ORIG_proba_{col}"] = df[col].map(ds[f"orig_proba_{col}"]).fillna(0.5).astype("float32")

    # 5. Distribution Features
    orig_ch_tc  = ds["orig_ch_tc"]
    orig_nc_tc  = ds["orig_nc_tc"]
    orig_all_tc = ds["orig_all_tc"]
    orig_is_mc  = ds["orig_is_mc"]
    orig_ref    = ds["orig_ref"]

    tc = df["TotalCharges"].values
    df["pctrank_nonchurner_TC"] = pctrank_against(tc, orig_nc_tc)
    df["pctrank_churner_TC"]    = pctrank_against(tc, orig_ch_tc)
    df["pctrank_orig_TC"]       = pctrank_against(tc, orig_all_tc)
    df["zscore_churn_gap_TC"]   = (np.abs(zscore_against(tc, orig_ch_tc)) -
                                    np.abs(zscore_against(tc, orig_nc_tc))).astype("float32")
    df["zscore_nonchurner_TC"]  = zscore_against(tc, orig_nc_tc)
    df["pctrank_churn_gap_TC"]  = (pctrank_against(tc, orig_ch_tc) -
                                    pctrank_against(tc, orig_nc_tc)).astype("float32")
    df["resid_IS_MC"]           = (df["MonthlyCharges"].values[0] -
                                    orig_is_mc.get(df["InternetService"].values[0], 0))

    is_val = df["InternetService"].values[0]
    ref_is = orig_ref.loc[orig_ref["InternetService"] == is_val, "TotalCharges"].values
    df["cond_pctrank_IS_TC"] = pctrank_against(tc, ref_is)[0] if len(ref_is) > 0 else 0.0

    c_val = df["Contract"].values[0]
    ref_c = orig_ref.loc[orig_ref["Contract"] == c_val, "TotalCharges"].values
    df["cond_pctrank_C_TC"] = pctrank_against(tc, ref_c)[0] if len(ref_c) > 0 else 0.0

    # 6. Quantile Distance
    for q_label, q_val in [("q25", 0.25), ("q50", 0.50), ("q75", 0.75)]:
        ch_q = np.quantile(orig_ch_tc, q_val)
        nc_q = np.quantile(orig_nc_tc, q_val)
        df[f"dist_To_ch_{q_label}"]   = float(np.abs(tc[0] - ch_q))
        df[f"dist_To_nc_{q_label}"]   = float(np.abs(tc[0] - nc_q))
        df[f"qdist_gap_To_{q_label}"] = float(np.abs(tc[0]-nc_q) - np.abs(tc[0]-ch_q))

    # 7. Digit Features
    t_str = str(int(df["tenure"].values[0]))
    df["tenure_first_digit"] = int(t_str[0])
    df["tenure_last_digit"]  = int(t_str[-1])
    df["tenure_mod10"]  = int(df["tenure"].values[0]) % 10
    df["tenure_mod12"]  = int(df["tenure"].values[0]) % 12

    mc = float(df["MonthlyCharges"].values[0])
    df["mc_first_digit"] = int(str(mc)[0])
    df["mc_mod10"]       = int(mc) % 10
    df["mc_fractional"]  = float(mc - int(mc))

    tc_val = float(df["TotalCharges"].values[0])
    df["tc_mod100"]     = int(tc_val) % 100
    df["tc_fractional"] = float(tc_val - int(tc_val))

    # N-gram strings
    for c1, c2 in combinations(TOP_CATS_FOR_NGRAM, 2):
        df[f"BG_{c1}_{c2}"] = str(df[c1].values[0]) + "_" + str(df[c2].values[0])
    for c1, c2, c3 in combinations(TOP4, 3):
        df[f"TG_{c1}_{c2}_{c3}"] = (str(df[c1].values[0]) + "_" +
                                      str(df[c2].values[0]) + "_" +
                                      str(df[c3].values[0]))
    # Num-as-Cat
    for col in NUMS:
        df[f"CAT_{col}"] = str(df[col].values[0])

    return df


def apply_te_inference(df_fe: pd.DataFrame) -> pd.DataFrame:
    """Apply stored TE look-ups → return numeric-only DataFrame ready for models."""
    df = df_fe.copy()
    te_stats    = pipeline_cfg["te_stats"]
    ng_te_stats = pipeline_cfg["ng_te_stats"]

    for col in TE_COLUMNS:
        for s in STATS:
            col_name = f"TE1_{col}_{s}"
            df[col_name] = df[col].map(te_stats[col][s]).fillna(0).astype("float32")

    for col in TE_NGRAM_COLUMNS:
        ng_name = f"TE_ng_{col}"
        df[ng_name] = df[col].map(ng_te_stats[col]).fillna(0.5).astype("float32")

    drop = set(TE_COLUMNS + TE_NGRAM_COLUMNS + CATS + NUMS + ["Churn"])
    df_num = df.drop(columns=[c for c in drop if c in df.columns], errors="ignore") \
               .select_dtypes(include=[np.number]).fillna(0)
    # Align to training columns (minus ridge_pred which we add later)
    base_cols = [c for c in NUMERIC_COLS if c != "ridge_pred" and c in df_num.columns]
    return df_num[base_cols]


def predict_pipeline(raw: dict) -> dict:
    """Full inference pipeline → returns ridge_prob, xgb_prob, shap values."""
    df_fe  = engineer_features_single(raw)
    df_num = apply_te_inference(df_fe)

    # Stage 1 — Ridge
    scaler   = pipeline_cfg["scaler"]
    X_scaled = scaler.transform(df_num)
    ridge_score = ridge_model.predict(X_scaled)[0]
    ridge_prob  = float(1 / (1 + np.exp(-ridge_score)))

    # Stage 2 — XGBoost  (with ridge_pred feature)
    df_xgb = df_num.copy()
    df_xgb["ridge_pred"] = ridge_prob
    # Ensure column order matches training
    xgb_cols = [c for c in NUMERIC_COLS if c in df_xgb.columns]
    df_xgb   = df_xgb[xgb_cols]
    xgb_prob = float(xgb_model.predict_proba(df_xgb)[:, 1][0])

    # SHAP
    sv = shap_explainer(df_xgb).values[0]
    shap_dict = {col: float(v) for col, v in zip(xgb_cols, sv)}

    return {
        "ridge_prob": ridge_prob,
        "xgb_prob":   xgb_prob,
        "shap_dict":  shap_dict,
        "xgb_cols":   xgb_cols,
    }

# ─────────────────────────────────────────────────────────────
# AI Strategy
# ─────────────────────────────────────────────────────────────
GROQ_MODELS = [
    "llama-3.3-70b-versatile",
    "llama-3.1-70b-versatile",
    "llama3-70b-8192",
    "mixtral-8x7b-32768",
]


def generate_ai_strategy(profile: dict, prob: float, shap_vals: dict) -> dict:
    risk_label  = "HIGH" if prob > 0.65 else "MEDIUM" if prob > 0.35 else "LOW"
    drivers     = sorted(shap_vals.items(), key=lambda x: x[1], reverse=True)
    top_risks   = [(f, v) for f, v in drivers if v > 0][:4]
    top_protect = [(f, v) for f, v in drivers if v < 0][:2]

    if groq_client:
        try:
            driver_txt  = "\n".join(f"  - {f}: SHAP {v:+.3f}" for f, v in top_risks)
            protect_txt = "\n".join(f"  - {f}: SHAP {v:+.3f}" for f, v in top_protect)
            prompt = f"""You are a senior customer retention strategist at a telecom company.
A customer has predicted churn probability {prob:.1%} ({risk_label} risk).

CUSTOMER PROFILE:
- Contract: {profile.get('contract')}
- Internet: {profile.get('internet_service')}
- Tenure: {profile.get('tenure')} months
- Monthly Charges: ${profile.get('monthly_charges'):.2f}
- Senior Citizen: {'Yes' if profile.get('senior_citizen') else 'No'}
- Tech Support: {profile.get('tech_support')}
- Online Security: {profile.get('online_security')}
- Payment Method: {profile.get('payment_method')}

TOP CHURN RISK DRIVERS (SHAP):
{driver_txt}

TOP PROTECTIVE FACTORS (SHAP):
{protect_txt}

Return EXACTLY 4 retention strategies as a JSON array.
Each object must have exactly these keys:
  "title"              - action title, max 6 words
  "priority"           - one of: Critical, High, Medium
  "body"               - 2-3 sentences referencing SHAP drivers and customer data
  "shap_driver"        - primary SHAP feature name
  "shap_value"         - SHAP value as string, e.g. "+0.474"
  "expected_reduction" - churn reduction, e.g. "~18%"
  "channel"            - contact channel, e.g. "Direct Call"

Return ONLY a valid JSON array, no markdown, no extra text."""

            for model_name in GROQ_MODELS:
                try:
                    resp     = groq_client.chat.completions.create(
                        model=model_name, temperature=0.35, max_tokens=1400,
                        messages=[{"role": "user", "content": prompt}],
                    )
                    raw_text = resp.choices[0].message.content.strip()
                    match    = re.search(r"\[.*\]", raw_text, re.DOTALL)
                    if match:
                        cards    = json.loads(match.group())
                        required = {"title","priority","body","shap_driver",
                                    "shap_value","expected_reduction","channel"}
                        if isinstance(cards, list) and len(cards) >= 3 and \
                           all(required.issubset(c.keys()) for c in cards):
                            return {"source": "groq", "cards": cards}
                except Exception:
                    continue
        except Exception:
            pass

    return {"source": "fallback",
            "cards": _rule_based_strategy(profile, prob, top_risks, top_protect)}


def _rule_based_strategy(profile, prob, top_risks, top_protect):
    monthly   = float(profile.get("monthly_charges", 70))
    contract  = str(profile.get("contract",         "Month-to-month"))
    internet  = str(profile.get("internet_service", "DSL"))
    tenure    = int(profile.get("tenure",            0))
    security  = str(profile.get("online_security",  "No"))
    payment   = str(profile.get("payment_method",   "Electronic check"))
    senior    = int(profile.get("senior_citizen",    0))
    risk_d    = dict(top_risks)

    discounted = round(monthly * 0.85, 2)
    cards: list = []

    if contract == "Month-to-month":
        cards.append({
            "title":              "Annual Contract Upgrade",
            "priority":           "Critical",
            "body":  (f"Month-to-month is the strongest churn predictor (SHAP {risk_d.get('Contract', 0.47):+.3f}). "
                      f"Offer a 12-month plan at ${discounted}/mo (15% loyalty discount). "
                      f"At {tenure} months tenure frame it as a milestone reward, not a sales pitch. "
                      f"Annual-contract switchers show ~60% lower 90-day churn."),
            "shap_driver":        "Contract",
            "shap_value":         f"{risk_d.get('Contract', 0.47):+.3f}",
            "expected_reduction": "~18%",
            "channel":            "Direct Call",
        })

    if internet == "Fiber optic" and security != "Yes":
        cards.append({
            "title":              "Online Security Bundle",
            "priority":           "High",
            "body":  (f"Fiber customers without Security churn at 28% vs 12% with it "
                      f"(SHAP {risk_d.get('OnlineSecurity', 0.22):+.3f}). "
                      f"Offer a free 3-month Security trial bundled with the contract upgrade. "
                      f"Combined SHAP resolution cuts predicted risk from {prob:.0%} to ~{max(prob-0.13,0.05):.0%}."),
            "shap_driver":        "OnlineSecurity",
            "shap_value":         f"{risk_d.get('OnlineSecurity', 0.22):+.3f}",
            "expected_reduction": "~8%",
            "channel":            "App / Email",
        })

    if payment == "Electronic check":
        cards.append({
            "title":              "Auto-Pay Migration",
            "priority":           "Medium",
            "body":  ("Electronic check users face higher billing friction — each payment is a re-evaluation moment. "
                      "Offer a $5/month discount to switch to bank transfer or card auto-pay. "
                      "Reduces churn triggers and lowers support-call volume."),
            "shap_driver":        "PaymentMethod",
            "shap_value":         f"{risk_d.get('PaymentMethod', 0.13):+.3f}",
            "expected_reduction": "~4%",
            "channel":            "SMS / Portal",
        })

    if senior:
        cards.append({
            "title":              "Senior Priority Tier",
            "priority":           "High",
            "body":  (f"Senior citizenship adds moderate churn risk (SHAP {risk_d.get('SeniorCitizen', 0.15):+.3f}). "
                      f"Enrol in the Senior Priority Support tier — dedicated queue, extended hours, simplified billing. "
                      f"Pair with a personalised call acknowledging the {tenure}-month tenure milestone."),
            "shap_driver":        "SeniorCitizen",
            "shap_value":         f"{risk_d.get('SeniorCitizen', 0.15):+.3f}",
            "expected_reduction": "~6%",
            "channel":            "Phone / Letter",
        })

    if internet == "Fiber optic":
        cards.append({
            "title":              "Fibre Experience Assurance",
            "priority":           "Medium",
            "body":  (f"Fiber customers have elevated expectations and face competitive alternatives "
                      f"(SHAP {risk_d.get('InternetService', 0.37):+.3f}). "
                      f"Send a proactive speed-performance report vs competitors + 30-day speed-upgrade trial. "
                      f"Proactive quality communication reduces churn 14% in the next 90 days."),
            "shap_driver":        "InternetService",
            "shap_value":         f"{risk_d.get('InternetService', 0.37):+.3f}",
            "expected_reduction": "~5%",
            "channel":            "Email / App",
        })

    if len(cards) < 3:
        ltv = round(monthly * 36)
        cards.append({
            "title":              "Loyalty Appreciation Outreach",
            "priority":           "Medium",
            "body":  (f"With {tenure} months tenure and ${ltv:,} projected 36-month value, "
                      f"this customer is worth protecting. A personalised 'valued customer' message "
                      f"with an exclusive offer achieves 73% acceptance in similar profiles."),
            "shap_driver":        "tenure",
            "shap_value":         f"{risk_d.get('tenure', 0.05):+.3f}",
            "expected_reduction": "~3%",
            "channel":            "Email / Call",
        })

    return cards

# ─────────────────────────────────────────────────────────────
# Routes
# ─────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return send_file("index.html")


@app.route("/api/health", methods=["GET"])
def health():
    cv = pipeline_cfg or {}
    return jsonify({
        "status": "healthy",
        "models_loaded": {
            "ridge":      ridge_model    is not None,
            "xgboost":    xgb_model      is not None,
            "pipeline":   pipeline_cfg   is not None,
            "background": background_df  is not None,
            "shap":       shap_explainer is not None,
        },
        "groq_enabled": groq_client is not None,
        "features":     len(NUMERIC_COLS) if NUMERIC_COLS else 0,
        "cv_xgb_auc":   cv.get("cv_xgb_auc"),
        "cv_ridge_auc": cv.get("cv_ridge_auc"),
    })


@app.route("/api/predict", methods=["POST"])
def predict():
    try:
        if not xgb_model or not ridge_model or not pipeline_cfg:
            return jsonify({"status": "error",
                            "message": "Models not loaded — run bootstrap_models.py first"}), 500

        data = request.get_json(force=True)

        raw = {
            "gender":           str(data.get("gender",           "Male")),
            "SeniorCitizen":    int(data.get("senior_citizen",   0)),
            "Partner":          str(data.get("partner",          "No")),
            "Dependents":       str(data.get("dependents",       "No")),
            "tenure":           float(data.get("tenure",         12)),
            "PhoneService":     str(data.get("phone_service",    "Yes")),
            "MultipleLines":    str(data.get("multiple_lines",   "No")),
            "InternetService":  str(data.get("internet_service", "DSL")),
            "OnlineSecurity":   str(data.get("online_security",  "No")),
            "OnlineBackup":     str(data.get("online_backup",    "No")),
            "DeviceProtection": str(data.get("device_protection","No")),
            "TechSupport":      str(data.get("tech_support",     "No")),
            "StreamingTV":      str(data.get("streaming_tv",     "No")),
            "StreamingMovies":  str(data.get("streaming_movies", "No")),
            "Contract":         str(data.get("contract",         "Month-to-month")),
            "PaperlessBilling": str(data.get("paperless_billing","Yes")),
            "PaymentMethod":    str(data.get("payment_method",   "Electronic check")),
            "MonthlyCharges":   float(data.get("monthly_charges", 70)),
            "TotalCharges":     float(data.get("total_charges",   840)),
        }

        result    = predict_pipeline(raw)
        ridge_prob = result["ridge_prob"]
        xgb_prob   = result["xgb_prob"]
        shap_dict  = result["shap_dict"]
        xgb_cols   = result["xgb_cols"]

        ranked = sorted(shap_dict.items(), key=lambda x: abs(x[1]), reverse=True)
        feature_importance = [
            {
                "feature":        col,
                "shap_value":     round(val, 4),
                "direction":      "risk" if val > 0 else "protective",
                "interpretation": SHAP_INTERP.get(col, ""),
                "input_value":    round(float(val), 4),
            }
            for col, val in ranked
        ]

        risk_count = sum(1 for v in shap_dict.values() if v >  0.02)
        prot_count = sum(1 for v in shap_dict.values() if v < -0.02)
        top_driver = ranked[0][0] if ranked else "—"
        top_val    = ranked[0][1] if ranked else 0.0

        profile = {
            "gender":           raw["gender"],
            "senior_citizen":   raw["SeniorCitizen"],
            "tenure":           raw["tenure"],
            "monthly_charges":  raw["MonthlyCharges"],
            "total_charges":    raw["TotalCharges"],
            "contract":         raw["Contract"],
            "internet_service": raw["InternetService"],
            "online_security":  raw["OnlineSecurity"],
            "tech_support":     raw["TechSupport"],
            "payment_method":   raw["PaymentMethod"],
        }
        strategy = generate_ai_strategy(profile, xgb_prob, shap_dict)

        monthly = raw["MonthlyCharges"]
        return jsonify({
            "status":             "success",
            "ridge_probability":  ridge_prob,
            "xgb_probability":    xgb_prob,
            "risk_level":         "High"   if xgb_prob > 0.65
                                  else "Medium" if xgb_prob > 0.35
                                  else "Low",
            "feature_importance": feature_importance,
            "shap_kpis": {
                "risk_count": risk_count,
                "prot_count": prot_count,
                "top_driver": top_driver,
                "top_val":    round(top_val, 3),
            },
            "strategy":           strategy,
            "business_metrics": {
                "ltv_36":  round(monthly * 36),
                "budget":  round(monthly * 0.15, 2),
                "urgency": 7 if xgb_prob > 0.65 else 14 if xgb_prob > 0.35 else 30,
            },
        })

    except Exception as e:
        traceback.print_exc()
        return jsonify({"status": "error", "message": str(e)}), 400


@app.route("/api/model-stats", methods=["GET"])
def model_stats():
    cfg = pipeline_cfg or {}
    return jsonify({
        "xgb_cv_mean":     cfg.get("cv_xgb_auc",   0),
        "xgb_cv_std":      cfg.get("cv_xgb_std",   0),
        "xgb_cv_scores":   cfg.get("fold_scores",  []),
        "ridge_cv_mean":   cfg.get("cv_ridge_auc", 0),
        "n_folds":         cfg.get("n_folds",       0),
        "xgb_train_auc":   0.906,
        "ridge_train_auc": 0.717,
        "training_samples": 7043,
        "churn_rate":       0.183,
    })


@app.route("/api/features", methods=["GET"])
def get_features():
    return jsonify({"features": NUMERIC_COLS or [], "count": len(NUMERIC_COLS or [])})


@app.errorhandler(404)
def not_found(e):    return jsonify({"error": "Not found"}), 404

@app.errorhandler(500)
def server_error(e): return jsonify({"error": "Internal server error"}), 500


if __name__ == "__main__":
    cv = pipeline_cfg or {}
    print("\n" + "=" * 60)
    print("  ChurnIQ - Customer Intelligence Platform")
    print("  Two-Stage Ridge -> XGBoost - N-gram TE - SHAP")
    print("=" * 60)
    print(f"  Ridge CV AUC  : {cv.get('cv_ridge_auc', 'N/A')}")
    print(f"  XGB   CV AUC  : {cv.get('cv_xgb_auc',   'N/A')}")
    print(f"  Groq LLM      : {'enabled' if groq_client else 'fallback mode'}")
    print(f"  Features      : {len(NUMERIC_COLS) if NUMERIC_COLS else 'not loaded'}")
    print(f"\n  Dashboard  ->  http://localhost:5000")
    print("=" * 60 + "\n")
    app.run(debug=False, host="0.0.0.0", port=5000, use_reloader=False)
