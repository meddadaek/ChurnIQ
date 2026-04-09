# Use official Python runtime as base image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Expose port
EXPOSE 5000

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV FLASK_ENV=production

# Run gunicorn with optimized worker count for ML models
CMD ["gunicorn", "--workers", "2", "--worker-class", "sync", "--bind", "0.0.0.0:5000", "--timeout", "180", "--access-logfile", "-", "--error-logfile", "-", "app:app"]
