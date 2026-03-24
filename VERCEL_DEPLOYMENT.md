# Vercel Deployment Guide

## Quick Start

1. Go to https://vercel.com and sign up
2. Connect your GitHub repository  
3. Add `GROQ_API_KEY` environment variable
4. Click Deploy - done!

## Step 1: Create Vercel Account

- Visit https://vercel.com
- Click **"Sign Up"** (use GitHub for easiest auth)
- Authorize Vercel to access your GitHub

## Step 2: Deploy from GitHub

### Option A: Web Dashboard (Easiest - Recommended)

1. Go to https://vercel.com/new
2. Click **"Import Git Repository"**
3. Paste your repo URL:
   ```
   https://github.com/meddadaek/customer-churn-predictor-with-shap-ai
   ```
4. Click **"Import"**
5. Vercel auto-detects Python from `vercel.json`
6. Continue to environment variables (see Step 3)

### Option B: CLI Deployment

```bash
# Install Vercel CLI
npm install -g vercel

# Go to project
cd c:\Users\DELL\Desktop\project

# Deploy
vercel

# Follow the interactive prompts
```

## Step 3: Set Environment Variables

**Critical**: Add your Groq API key:

1. In Vercel dashboard, select your project
2. Go to **"Settings"** → **"Environment Variables"**
3. Click **"Add New"**
4. Enter:
   ```
   Name:  GROQ_API_KEY
   Value: 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
   ```
5. Select **"All Environments"** (Production, Preview, Development)
6. Click **"Add"**
7. Click **"Redeploy"** to apply changes

## Step 4: Access Your App

After deployment completes:

- Vercel provides a unique URL (`.vercel.app` domain)
- Click the URL or view in "Deployments" tab
- Dashboard loads instantly with all 6 pages
- UI: Overview, Prediction, SHAP, Performance, Strategy, AI Suggestions

## API Endpoints Available

```
POST   /api/predict        - Make churn prediction
GET    /api/health         - Health check  
GET    /api/model-stats    - Model performance metrics
GET    /api/features       - Feature list
GET    /                   - Dashboard UI
```

## Configuration Details

### Vercel.json
- **Runtime**: Python 3.11
- **Max Lambda Size**: 250MB (for large model files)
- **Routes**: All requests → `app.py`

### Requirements.txt
- Flask 3.0+
- All ML dependencies (XGBoost, scikit-learn, etc.)
- Groq API client
- python-dotenv for env vars
- gunicorn for production WSGI

## Environment Variables

| Variable | Required | Source |
|----------|----------|--------|
| `GROQ_API_KEY` | Yes* | console.groq.com |
| `PYTHONUNBUFFERED` | No | Auto-set to `1` |

*App works without it (fallback mode), but AI recommendations disabled

## Production Details

### Cold Starts
- **First request after deploy**: 10-20 seconds (model loading)
- **Subsequent**: <1 second (cached)

### Performance
- **Prediction**: 1-2 seconds
- **SHAP analysis**: 2-3 seconds
- **Groq API call**: 3-5 seconds total

### Limits
- **Timeout**: 60 seconds (sufficient for predictions)
- **Max package size**: 250MB (model files ~150MB)
- **Memory**: 1GB (per Vercel standard)

## Monitoring & Logs

1. Go to **"Deployments"** tab
2. Click latest deployment
3. View **"Logs"** for real-time errors
4. Check **"Analytics"** for traffic/performance

## Redeployment

Any push to GitHub auto-triggers redeploy:

```bash
git add .
git commit -m "Update description"
git push origin main
# Vercel automatically deploys!
```

## Troubleshooting

### Build Fails
**Error**: `Module not found`
**Solution**: Verify all packages in `requirements.txt`

**Error**: `Python version mismatch`
**Solution**: `vercel.json` specifies Python 3.11 - should be fine

### App Crashes at Runtime
**Error**: `OOM (Out of Memory)`
**Cause**: Model loading uses ~1GB
**Solution**: Vercel's free/pro plans sufficient

**Error**: Lambda timeout (> 60s)
**Cause**: Groq API slow or prediction hanging
**Solution**: Already set timeout. Check Groq API status

### Models Not Loading
**Error**: `FileNotFoundError: models/xgb_final.pkl`
**Solution**: Ensure all `.pkl` files committed to GitHub

### API Key Issues
**Issue**: Groq calls fail
**Steps**:
1. Verify key in Vercel dashboard → Variables
2. Check key starts with `gsk_`
3. Redeploy after adding key
4. Check logs for auth errors

## Costs

**Vercel Pricing**:
- **Hobby (Free)**: 
  - 100 GB-hours/month (free tier)
  - Limited to 60s execution time
  - Great for demo/testing

- **Pro ($20/month)**:
  - Unlimited excecution time
  - 1TB bandwidth
  - Priority support
  - Recommended for production

## Deployment Comparison

| Feature | Vercel | Railway |
|---------|--------|---------|
| GitHub sync | ✅ Auto | ✅ Auto |
| Environment vars | ✅ Easy UI | ✅ Easy UI |
| Cold start | ~15s | ~5s |
| Scaling | Serverless | Traditional |
| Free tier | ✅ Yes (60s limit) | ✅ Yes |
| Custom domain | ✅ Yes | ✅ Yes |

## Next Steps

✅ Code ready on GitHub  
⏳ Visit https://vercel.com/new  
⏳ Import your GitHub repo  
⏳ Add GROQ_API_KEY variable  
⏳ Click Deploy  
⏳ Done! Your app is live  

**Deployment time**: ~3-5 minutes

## Support

- Vercel docs: https://vercel.com/docs
- Python on Vercel: https://vercel.com/docs/concepts/runtimes/python
- Groq API: https://console.groq.com/docs
- Check logs in Vercel dashboard for errors

---

**Your Repository**: https://github.com/meddadaek/customer-churn-predictor-with-shap-ai

Deploy to Vercel: https://vercel.com/new
