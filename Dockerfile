# Dockerfile : pour Cloud Run
FROM python:3.10-slim

# Installer dépendances système (ex: pour numpy, faiss, flask, etc.)
RUN apt-get update && apt-get install -y \
    build-essential \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Créer dossier de travail
WORKDIR /app

# Copier le code
COPY . .

# Installer les dépendances Python
RUN pip install --no-cache-dir -r requirements.txt

# Exposer le port Flask
ENV PORT=8080
EXPOSE 8080

# Démarrer l'app Flask
CMD ["python", "main.py"]
