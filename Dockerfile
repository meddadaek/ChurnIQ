FROM python:3.11-slim
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends gcc && rm -rf /var/lib/apt/lists/*
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
EXPOSE 8080
ENV PYTHONUNBUFFERED=1
ENV FLASK_ENV=production
CMD ["gunicorn", "--workers", "1", "--worker-class", "sync", "--bind", "0.0.0.0:8080", "--timeout", "300", "--access-logfile", "-", "--error-logfile", "-", "app:app"]
