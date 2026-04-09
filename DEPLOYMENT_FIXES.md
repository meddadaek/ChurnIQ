# ChurnIQ Deployment Fixes & Optimization Guide

## Overview of Changes

This document outlines the critical fixes applied to resolve the "Prediction failed: Application failed to respond" error and performance improvements made to the ChurnIQ application.

## Root Causes Identified

### 1. **Port Mismatch in Docker**
- **Issue**: `Dockerfile` exposed port 8000 but the application was binding to port 5000
- **Impact**: Container networking failures on platforms like Railway/Vercel
- **Fix**: Updated `EXPOSE 5000` to match the gunicorn binding port

### 2. **Excessive Worker Count**
- **Issue**: 4 Gunicorn workers with heavy ML models (XGBoost, SHAP) caused memory exhaustion
- **Impact**: Application crashes or timeouts on limited-resource platforms
- **Fix**: Reduced workers from 4 to 2 with `--worker-class sync` for better memory efficiency

### 3. **Insufficient Timeout**
- **Issue**: 120-second timeout was too short for SHAP calculations (1-2s per prediction)
- **Impact**: Long-running predictions were terminated mid-process
- **Fix**: Increased timeout to 180 seconds (3 minutes)

### 4. **Loose Dependency Pinning**
- **Issue**: Requirements used only minimum versions (e.g., `flask>=3.0.0`)
- **Impact**: Environment drift and compatibility issues across deployments
- **Fix**: Added upper-bound constraints (e.g., `flask>=3.0.0,<4.0.0`)

## Changes Made

### Dockerfile
```dockerfile
# Before
EXPOSE 8000
CMD ["gunicorn", "--workers", "4", "--bind", "0.0.0.0:5000", "--timeout", "120", "app:app"]

# After
EXPOSE 5000
CMD ["gunicorn", "--workers", "2", "--worker-class", "sync", "--bind", "0.0.0.0:5000", "--timeout", "180", "--access-logfile", "-", "--error-logfile", "-", "app:app"]
```

**Improvements:**
- Fixed port mismatch
- Reduced workers to 2 (optimal for ML models on limited resources)
- Increased timeout to 180 seconds
- Added logging for better debugging

### Procfile
```procfile
# Before
web: gunicorn --workers 4 --bind 0.0.0.0:$PORT --timeout 120 app:app

# After
web: gunicorn --workers 2 --worker-class sync --bind 0.0.0.0:$PORT --timeout 180 --access-logfile - --error-logfile - app:app
```

**Improvements:**
- Consistent with Docker configuration
- Better resource utilization on Railway

### requirements.txt
Added upper-bound version constraints:
```
flask>=3.0.0,<4.0.0
flask-cors>=4.0.0,<5.0.0
numpy>=1.24.0,<2.0.0
pandas>=2.0.0,<3.0.0
scikit-learn>=1.3.0,<2.0.0
xgboost>=2.0.0,<3.0.0
shap>=0.44.0,<1.0.0
groq>=0.4.0,<1.0.0
python-dotenv>=1.0.0,<2.0.0
gunicorn>=21.0.0,<22.0.0
```

### vercel.json
Added memory and timeout configuration:
```json
{
  "config": {
    "runtime": "python3.11",
    "maxLambdaSize": "250mb",
    "memory": 1024,
    "maxDuration": 60
  }
}
```

### app.py
Enhanced error handling in `/api/predict`:
- Changed error status code from 500 to 503 for "Models not loaded"
- Added validation for empty JSON payloads

## Deployment Instructions

### Railway Deployment
1. Push changes to your repository
2. Railway will automatically detect the `Procfile` and rebuild
3. Verify deployment with: `curl https://<your-railway-url>/api/health`

### Vercel Deployment
1. Push changes to your repository
2. Vercel will rebuild using `vercel.json` configuration
3. Verify deployment with: `curl https://<your-vercel-url>/api/health`

### Docker Local Testing
```bash
docker build -t churniq .
docker run -p 5000:5000 -e GROQ_API_KEY="" churniq
curl http://localhost:5000/api/health
```

## Performance Expectations

After these fixes:
- **Cold start**: ~3-5 seconds (model loading)
- **Prediction request**: ~1-2 seconds (Ridge + XGBoost)
- **SHAP explanation**: +1-2 seconds
- **Total response time**: ~2-4 seconds for most requests

## Monitoring Recommendations

1. **Memory Usage**: Monitor that each worker stays below 500MB
2. **Request Timeout**: Set alerts if requests exceed 150 seconds
3. **Error Rate**: Track 503 errors (models not loaded) and 500 errors (runtime failures)
4. **Response Time**: Aim for p95 under 5 seconds

## Future Optimization Opportunities

1. **Model Caching**: Implement Redis caching for frequent predictions
2. **Async Workers**: Consider `--worker-class gthread` with threading for I/O-bound operations
3. **Model Quantization**: Reduce model size to speed up loading
4. **Batch Predictions**: Add `/api/predict-batch` endpoint for multiple predictions
5. **SHAP Caching**: Cache SHAP explanations for similar customer profiles

## Troubleshooting

### "Application failed to respond"
- Check that models are loaded: `curl https://<url>/api/health`
- Verify timeout is sufficient for your platform
- Check platform logs for memory exhaustion messages

### Slow predictions
- Verify SHAP explainer is initialized
- Check if Groq API is configured (optional but improves response time)
- Consider reducing feature count if needed

### 503 Service Unavailable
- Models failed to load during startup
- Check that all pickle files exist in `models/` directory
- Verify file permissions are correct

## References

- [Gunicorn Configuration](https://docs.gunicorn.org/en/stable/settings.html)
- [Flask Deployment Best Practices](https://flask.palletsprojects.com/en/2.3.x/deploying/)
- [Railway Deployment Guide](https://docs.railway.app/)
- [Vercel Python Runtime](https://vercel.com/docs/functions/serverless-functions/python)
