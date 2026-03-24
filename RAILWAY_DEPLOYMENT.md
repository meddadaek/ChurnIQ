# Railway Deployment Guide

## Quick Start

1. Go to https://railway.app and sign up
2. Create new project → Connect GitHub → Select this repo
3. Railway auto-detects Python and builds automatically
4. Add `GROQ_API_KEY` environment variable (see below)
5. Deploy and view your app!

## Step 1: Create Railway Account
- Visit https://railway.app
- Sign up (GitHub recommended for easy authorization)
- Create a new project

## Step 2: Deploy from GitHub

### Option A: Web Dashboard (Easiest)
1. Click "New Project" → "Deploy from GitHub repo"
2. Authorize Railway to access your GitHub
3. Select: `customer-churn-predictor-with-shap-ai`
4. Railway auto-detects Python, installs `requirements.txt`
5. Wait for build complete (~5-10 minutes)

### Option B: Railway CLI
```bash
npm install -g @railway/cli
railway login
cd c:\Users\DELL\Desktop\project
railway init
railway up
```

## Step 3: Configure Environment Variables

**Required**: Set Groq API key for AI recommendations:

1. Open Railway dashboard → Select your service
2. Go to "Variables" tab
3. Add new variable:
   ```
   Key:   GROQ_API_KEY
   Value: [Your Groq API key from console.groq.com]
   ```
4. Click "Update"
5. Service restarts automatically

## Step 4: Access Your App

After deployment completes:
- Railway provides a unique URL (e.g., `https://customer-churn-service.up.railway.app`)
- Open the URL in browser
- Dashboard loads with all 6 pages:
  - Overview (predictions & KPIs)
  - Prediction (input form)
  - SHAP Analysis (feature importance)
  - Performance Metrics (model validation)
  - Strategy (business insights)
  - AI Suggestions (Groq-powered)

## API Endpoints

```
POST   /api/predict        - Make churn prediction
GET    /api/health         - Health check
GET    /api/model-stats    - Model performance metrics
GET    /api/features       - Feature list
GET    /                   - Dashboard UI
```

## Environment Variables

| Variable | Required | Source |
|----------|----------|--------|
| `GROQ_API_KEY` | Yes* | console.groq.com |
| `PYTHONUNBUFFERED` | No | Auto-set to `1` |
| `FLASK_ENV` | No | Auto-set to `production` |

*Required for AI recommendations. App still works without it (fallback mode).

## Production Configuration

### Scaling
- **Workers**: 4 (configured in Procfile)
- **Timeout**: 120 seconds (for model predictions)
- **Port**: Auto-assigned by Railway

### Resources
- **RAM**: 512MB-1GB recommended
- **CPU**: Shared, auto-scales

### Monitoring
- View logs: Railway dashboard → "Logs" tab
- Monitor metrics: CPU, memory, requests
- Set alerts for errors/restarts

## Troubleshooting

### Build Fails
**Check logs**: Dashboard → Logs tab → search for errors
**Common issues**:
- Missing Python version (uses 3.11 by default)
- Large dependency files
- Model pickle files not committed

**Solution**: Ensure `models/*.pkl` files are in git

### App Crashes
**Issue**: Memory exhausted
**Solution**: Upgrade Railway plan or optimize model loading

**Issue**: Timeout errors
**Solution**: Timeout already 120s. If persists, check Groq API limits

### Missing Models
**Error**: `Models not found`
**Fix**: Verify in GitHub:
```
models/
  ├── background.csv
  ├── bootstrap_models.py
  ├── features_numeric.pkl
  ├── pipeline_config.pkl
  ├── ridge_final.pkl
  └── xgb_final.pkl
```

### API Key Not Working
1. Verify key is set: Dashboard → Variables
2. Check key format: Should start with `gsk_`
3. Ensure no extra spaces
4. Restart service after adding key

## Performance Notes

- **First prediction**: ~2-3 seconds (model loading)
- **Subsequent predictions**: <500ms
- **SHAP calculations**: ~1-2 seconds per prediction
- **Groq API calls**: ~3-5 seconds (depends on model traffic)

## Costs

- **Free tier**: 
  - Deployments: Unlimited
  - Resource credits: ~5$/month
  - Sufficient for light usage

- **Pro tier**: 
  - More resource credits
  - Priority support
  - $10/month minimum

## Next Steps

✅ Code is ready on GitHub
⏳ Create Railway account: https://railway.app
⏳ Connect GitHub repository
⏳ Add GROQ_API_KEY variable
⏳ Deploy and test at Railway URL

Your app will be live within minutes!

## Support

- Railway docs: https://docs.railway.app
- Groq API docs: https://console.groq.com/docs
- Issues? Check Railway logs for detailed error messages
