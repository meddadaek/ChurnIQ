web: gunicorn --workers 1 --worker-class sync --bind 0.0.0.0:$PORT --timeout 300 --access-logfile - --error-logfile - app:app
