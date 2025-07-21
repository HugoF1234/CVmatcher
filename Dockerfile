FROM python:3.10-slim

# Empêche Python de bufferiser les logs
ENV PYTHONUNBUFFERED True

WORKDIR /app

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Expose le port par défaut attendu par Cloud Run
EXPOSE 8080

# Lancer Gunicorn avec le bon module/app
CMD ["gunicorn", "-c", "gunicorn.conf.py", "main:app"]
