FROM python:3.11-slim
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
EXPOSE 8000
ENV PYTHONUNBUFFERED=1
ENV FLASK_ENV=production
CMD ["gunicorn", "--workers", "2", "--worker-class", "sync", "--bind", "0.0.0.0:8000", "--timeout", "180", "--access-logfile", "-", "--error-logfile", "-", "app:app"]
