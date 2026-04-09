"""
ChurnIQ Health Check Utility
=============================
Provides comprehensive health status and diagnostics for the application.
"""

import os
import sys
import json
import psutil
from pathlib import Path
from datetime import datetime


def check_models():
    """Verify all required model files exist and are accessible."""
    model_dir = Path("models")
    required_files = [
        "pipeline_config.pkl",
        "ridge_final.pkl",
        "xgb_final.pkl",
        "features_numeric.pkl",
        "background.csv",
    ]
    
    status = {
        "models_present": True,
        "files": {},
        "total_size_mb": 0,
    }
    
    for filename in required_files:
        filepath = model_dir / filename
        if filepath.exists():
            size_mb = filepath.stat().st_size / (1024 * 1024)
            status["files"][filename] = {
                "exists": True,
                "size_mb": round(size_mb, 2),
            }
            status["total_size_mb"] += size_mb
        else:
            status["files"][filename] = {"exists": False, "size_mb": 0}
            status["models_present"] = False
    
    status["total_size_mb"] = round(status["total_size_mb"], 2)
    return status


def check_environment():
    """Check environment variables and configuration."""
    status = {
        "python_version": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
        "flask_env": os.getenv("FLASK_ENV", "not set"),
        "groq_api_key": "configured" if os.getenv("GROQ_API_KEY") else "not configured",
        "pythonunbuffered": os.getenv("PYTHONUNBUFFERED", "not set"),
    }
    return status


def check_system_resources():
    """Check available system resources."""
    try:
        memory = psutil.virtual_memory()
        cpu_count = psutil.cpu_count()
        
        status = {
            "cpu_count": cpu_count,
            "memory_total_mb": round(memory.total / (1024 * 1024), 2),
            "memory_available_mb": round(memory.available / (1024 * 1024), 2),
            "memory_percent": memory.percent,
            "memory_status": "warning" if memory.percent > 80 else "ok",
        }
    except Exception as e:
        status = {"error": str(e)}
    
    return status


def check_dependencies():
    """Verify all required Python packages are installed."""
    required_packages = [
        "flask",
        "flask_cors",
        "numpy",
        "pandas",
        "sklearn",
        "xgboost",
        "shap",
        "groq",
        "dotenv",
        "gunicorn",
    ]
    
    status = {"packages": {}}
    
    for package in required_packages:
        try:
            module = __import__(package)
            version = getattr(module, "__version__", "unknown")
            status["packages"][package] = {
                "installed": True,
                "version": str(version),
            }
        except ImportError:
            status["packages"][package] = {
                "installed": False,
                "version": None,
            }
    
    return status


def generate_health_report():
    """Generate a comprehensive health report."""
    report = {
        "timestamp": datetime.utcnow().isoformat(),
        "models": check_models(),
        "environment": check_environment(),
        "system_resources": check_system_resources(),
        "dependencies": check_dependencies(),
        "overall_status": "healthy",
    }
    
    # Determine overall status
    if not report["models"]["models_present"]:
        report["overall_status"] = "unhealthy"
    elif report["system_resources"].get("memory_status") == "warning":
        report["overall_status"] = "degraded"
    elif any(not pkg["installed"] for pkg in report["dependencies"]["packages"].values()):
        report["overall_status"] = "degraded"
    
    return report


if __name__ == "__main__":
    report = generate_health_report()
    print(json.dumps(report, indent=2))
