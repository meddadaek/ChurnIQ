# ChurnIQ · Customer Intelligence Platform

![Version](https://img.shields.io/badge/version-1.0.0-blue.svg)
![Python](https://img.shields.io/badge/python-3.8+-green.svg)
![Flask](https://img.shields.io/badge/flask-3.0.0+-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

## 🎯 Overview

**ChurnIQ** is a sophisticated customer churn prediction platform that combines machine learning explainability (SHAP), AI-powered strategy recommendations, and real-time model insights. It predicts customer churn probability using a two-stage pipeline (Ridge → XGBoost) and provides actionable retention strategies powered by Groq LLM.

Built for telecom customer retention, but extensible to any subscription-based business.

---

## ✨ Key Features

### 🔮 **Dual-Stage Prediction Pipeline**
- **Stage 1**: Ridge Classifier (captures linear patterns, AUC: 0.697)
- **Stage 2**: XGBoost (non-linear corrections, AUC: 0.849)
- Ridge predictions passed as meta-features to XGBoost → **21.8% lift in performance**

### 📊 **Feature Engineering (7 Steps)**
1. Frequency encoding on numeric features
2. Arithmetic interactions (charges deviation, ratios)
3. Service count aggregations
4. Target probability encoding (ORIG_proba)
5. Distribution features (percentile ranks, z-scores)
6. Quantile distance features
7. **N-gram composite encoding** (bigrams/trigrams + target encoding)

### 🔍 **SHAP Explainability**
- Tree-explainer for XGBoost with waterfall plots
- Feature importance rankings with business interpretation
- Customer-level explanations for every prediction
- Visual attribution analysis

### 🤖 **AI-Powered Retention Strategies**
- Groq LLM generates personalized 4-action retention plans
- Fallback rule-based strategy engine when API unavailable
- Expected churn reduction estimates per action
- Channel recommendations (Call, Email, SMS, Portal)

### 📈 **Interactive Dashboard**
- **Overview**: Real-time KPIs, tenure segments, model health
- **Prediction**: Form-based inference with two-stage pipeline visualization
- **SHAP Explainer**: Waterfall plots, feature attribution tables
- **Model Performance**: ROC curves, calibration, 20-fold CV analysis
- **Retention Strategy**: Cumulative impact analysis, action sequences
- **AI Suggestions**: Predictive timeline, smart segmentation, peer benchmarks

---

## 🛠️ Tech Stack

| Component | Technology |
|-----------|------------|
| **Backend** | Flask 3.0.0 |
| **ML Models** | XGBoost 2.0.0, scikit-learn 1.3.0 |
| **Explainability** | SHAP 0.44.0 |
| **Data** | Pandas 2.0.0, NumPy 1.24.0 |
| **LLM** | Groq API (llama-3.3-70b) |
| **Frontend** | Vanilla JS, Chart.js 4.4.1 |
| **CORS** | Flask-CORS 4.0.0 |

---

## 📋 Project Structure

```
project/
├── app.py                          # Flask backend + inference pipeline
├── index.html                      # Dashboard (6 pages, full UI)
├── requirements.txt                # Python dependencies
├── run_server.bat                  # Windows batch to start server
├── models/
│   ├── bootstrap_models.py         # Model training + feature engineering
│   ├── pipeline_config.pkl         # Encoding lookup tables, scaler
│   ├── ridge_final.pkl             # Ridge classifier
│   ├── xgb_final.pkl               # XGBoost classifier
│   ├── features_numeric.pkl        # Numeric feature list (for SHAP)
│   └── background.csv              # 200 encoded rows for SHAP background
└── __pycache__/
```

---

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- pip or conda
- Groq API key (optional, fallback mode available)

### 1. Clone Repository
```bash
git clone https://github.com/meddadaek/customer-churn-predictor-with-shap-ai.git
cd customer-churn-predictor-with-shap-ai
```

### 2. Create Virtual Environment
```bash
python -m venv .venv
.\.venv\Scripts\Activate.ps1    # Windows PowerShell
source .venv/bin/activate       # macOS/Linux
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Train Models (One-Time)
```bash
python models/bootstrap_models.py
```

**Output:**
- Generates 7,043-sample synthetic telco dataset
- Trains 20-fold cross-validate pipeline
- Saves 5 model artifacts to `models/` directory
- Ridge CV AUC: **0.9103**, XGB CV AUC: **0.9194**

### 5. Set Groq API Key (Optional)
```bash
# Windows PowerShell
$env:GROQ_API_KEY = "your-api-key-here"

# macOS/Linux
export GROQ_API_KEY="your-api-key-here"
```

### 6. Start Server
```bash
python app.py
```

Output:
```
============================================================
  ChurnIQ · Customer Intelligence Platform
  Two-Stage Ridge → XGBoost · N-gram TE · SHAP
============================================================
  Ridge CV AUC  : 0.9103
  XGB   CV AUC  : 0.9194
  Groq LLM      : enabled
  Features      : 108
  Dashboard  →  http://localhost:5000
============================================================
```

### 7. Open Dashboard
Navigate to **http://localhost:5000**

---

## 📖 Usage Guide

### Overview Page
- Real-time KPIs: Churn probability, model AUCs, tenure, monthly value
- Tenure segment analysis (0-6mo to 61-72mo)
- Contract type churn rates
- Model health metrics (accuracy, precision, F1)

### Prediction Page
1. **Fill Customer Form**: 18 fields covering demographics, services, contract, charges
2. **Click "Run Prediction"**: 
   - Ridge predicts: 41.2% → XGB refines: 38.7%
   - Computes SHAP values for 108 features
   - Phones Groq LLM for retention strategies
3. **View Results**:
   - Gauge chart with risk level badge
   - Two-stage pipeline visualization
   - Top 8 SHAP drivers with interpretation
   - AI-generated strategies with impact/priority

### SHAP Explainer Page
- **KPI Row**: Risk/protective factor counts, top driver
- **Waterfall Chart**: Top 7 features with cumulative log-odds
- **Feature Attribution**: All 108 features with SHAP values
- **Summary Table**: Business interpretation for each feature

### Model Performance Page
- **ROC Curves**: XGBoost (AUC=0.849) vs Ridge (AUC=0.697)
- **Probability Distribution**: Churn vs No-Churn histogram
- **20-Fold CV**: Per-fold AUC scores with stability analysis
- **Calibration Curve**: Predicted vs actual positive fraction
- **Metrics Table**: Precision, recall, F1, accuracy comparison

### Retention Strategy Page
- **Risk Metrics**: XGB churn % (dynamic), LTV 36-month, retention budget
- **Strategy Cards**: 4-5 action cards with SHAP drivers
  - Priority: Critical/High/Medium
  - Expected reduction: ~18-5% per action
  - Channel: Call/Email/SMS/Portal
- **Cumulative Impact**: Baseline → final churn reduction

### AI Suggestions Page
- **KPIs**: 91.94% AI confidence (CV AUC), 73%+ success rate, next best action, urgency window
- **Primary Recommendation**: From Groq or rule-based engine
- **Secondary Insights**: Top 3 risk factors, top 2 protective factors
- **Predictive Timeline**: Days to churn peak, optimal action window
- **Smart Segmentation**: Customer value at-risk, LTV impact, peer benchmarks
- **Action Sequence**: 4-step timeline with channels (Day 1, 3, 7, 14)

---

## 🧠 Model Architecture

### Feature Engineering Pipeline

```
Raw Customer Data (19 features)
    ↓
[1] Frequency Encoding (3 numeric cols)
    ↓
[2] Arithmetic Interactions (3 features)
    ↓
[3] Service Counts (3 aggregates)
    ↓
[4] Target Probability Encoding (19 cols)
    ↓
[5] Distribution Features (8 quantile/z-score)
    ↓
[6] Quantile Distance Features (9 distance metrics)
    ↓
[7] Digit + N-gram Features (26 composite)
    ↓
Total: 108 numeric features
    ↓
[Apply Target Encoding on inner folds]
    ↓
Stage 1: Ridge Classifier  [sparse, linear]
    ↓
Stage 2: XGBoost Classifier  [uses Ridge pred + 108 features]
    ↓
Predictions + SHAP values
```

### Two-Stage Pipeline

```python
Stage 1 (Ridge):
  Input: Scaled numeric features (108)
  Output: ridge_prob ∈ [0, 1]
  AUC: 0.697 (interpretable, linear)

Stage 2 (XGBoost):
  Input: 108 features + ridge_prob (meta-feature)
  Params: 3000 trees, depth=5, lr=0.02, early_stop=150
  Output: xgb_prob ∈ [0, 1]
  AUC: 0.849 → CV: 0.919
  Lift: +21.8% vs Ridge alone
```

---

## 🔐 API Endpoints

### `/api/predict` (POST)
Predict churn probability and return SHAP + strategy

**Request:**
```json
{
  "gender": "Male",
  "senior_citizen": 0,
  "tenure": 24,
  "monthly_charges": 89.99,
  "total_charges": 2160.0,
  "contract": "Month-to-month",
  "internet_service": "Fiber optic",
  "online_security": "No",
  "tech_support": "Yes",
  ...
}
```

**Response:**
```json
{
  "status": "success",
  "ridge_probability": 0.412,
  "xgb_probability": 0.387,
  "risk_level": "Medium",
  "feature_importance": [
    {
      "feature": "Contract",
      "shap_value": 0.4738,
      "direction": "risk",
      "interpretation": "Month-to-month = no switching cost..."
    },
    ...
  ],
  "shap_kpis": {
    "risk_count": 12,
    "prot_count": 8,
    "top_driver": "Contract",
    "top_val": 0.4738
  },
  "strategy": {
    "source": "groq",
    "cards": [
      {
        "title": "Annual Contract Upgrade",
        "priority": "Critical",
        "body": "...",
        "shap_driver": "Contract",
        "expected_reduction": "~18%",
        "channel": "Direct Call"
      }
    ]
  },
  "business_metrics": {
    "ltv_36": 2700,
    "budget": 270.0,
    "urgency": 7
  }
}
```

### `/api/health` (GET)
Model liveness check

### `/api/model-stats` (GET)
CV scores, training metrics, model configuration

### `/api/features` (GET)
List of 108 numeric features

---

## 📊 Data & Model Details

### Training Dataset
- **Size**: 7,043 customers
- **Churn Rate**: 20.2% (class imbalance)
- **Generated**: Realistic telecom data (reference + 80% train split)
- **Features**: 19 raw (demographics, services, charges)

### Cross-Validation
- **Folds**: 20 stratified folds
- **Ridge CV AUC**: 0.9103 ± 0.0031
- **XGB CV AUC**: 0.9194 ± 0.0015
- **Strategy**: Inner 5-fold for target encoding

### SHAP Background
- 200 samples from training set
- Used by TreeExplainer for waterfall plots
- Enables fast per-sample SHAP computation

---

## 🤝 Integration with Groq LLM

### Feature
Groq API generates personalized retention strategies using:
- Customer risk profile (churn %)
- Top SHAP drivers (features pushing up/down)
- Business context (tenure, monthly charges, contract)

### Fallback
If Groq unavailable → rule-based strategy engine:
- Contract = Month-to-month → Upgrade offer
- Fiber optic + no security → Security bundle
- Electronic check → Auto-pay migration
- Senior citizen → Priority support
- Plus 2-3 additional context-driven rules

### Cost
Groq free tier: 30 req/minute, no auth for eval models.

---

## 🧪 Testing & Validation

### Example Prediction
```
Customer Profile:
- 24 months tenure
- $89.99/month, $2,160 total
- Month-to-month contract, Fiber optic
- No tech support, no online security

Ridge Output: 41.2%
XGB Output: 38.7%
Risk Level: MEDIUM

Top Drivers:
1. Contract (SHAP +0.4738) — month-to-month penalty
2. InternetService (SHAP +0.0748) — fiber competition
3. OnlineSecurity (SHAP -0.0572) — missing bundle

Strategy:
• Annual upgrade with 15% discount (~18% churn reduction)
• Security bundle trial (~8% reduction)
• Auto-pay migration (~4% reduction)
→ Total: ~30% churn reduction expected
```

---

## 📦 Dependencies

```
flask>=3.0.0
flask-cors>=4.0.0
numpy>=1.24.0
pandas>=2.0.0
scikit-learn>=1.3.0
xgboost>=2.0.0
shap>=0.44.0
groq>=0.4.0
```

Install all:
```bash
pip install -r requirements.txt
```

---

## 🚦 Known Limitations

1. **Dataset Realism**: Synthetic data generated from patterns. For production, use actual historical data.
2. **Groq Dependency**: LLM strategy generation requires API key (fallback available).
3. **Feature Drift**: Model trained on 7K samples. Monitor SHAP drift in production.
4. **Class Imbalance**: 20% churn rate. Consider threshold tuning for recall vs precision trade-off.
5. **Tenure Encoding**: Max tenure 72 months. Handle outliers in production.

---

## 🔮 Future Enhancements

- [ ] Multi-language support (SHAP interpretations)
- [ ] Batch prediction API for bulk scoring
- [ ] Custom threshold tuning UI
- [ ] Model retraining pipeline with drift detection
- [ ] A/B testing framework for strategies
- [ ] Real customer database integration
- [ ] Mobile app for field agents
- [ ] Segment-level cohort analysis

---

## 📝 License

This project is licensed under the **MIT License** — see [LICENSE](LICENSE) for details.

---

## 🤝 Contributing

Contributions welcome! Please:
1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open Pull Request

---

## 📧 Contact & Support

- **Author**: [Your Name] (@meddadaek)
- **Email**: [your-email]
- **Issues**: [GitHub Issues](https://github.com/meddadaek/customer-churn-predictor-with-shap-ai/issues)

---

## 🙏 Acknowledgments

- **Kaggle Telco Churn Dataset** — Inspired feature engineering
- **SHAP** — Model explainability framework
- **XGBoost & scikit-learn** — ML backbone
- **Groq** — LLM inference platform
- **Chart.js** — Interactive visualizations

---

**Built with ❤️ for customer retention teams**

Last Updated: March 2026 | ChurnIQ v1.0.0
