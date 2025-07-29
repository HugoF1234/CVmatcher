# Étape 1: Utiliser une image Python 3.9 slim comme base
FROM python:3.9-slim

# Définir l'environnement pour éviter que Python ne génère des fichiers .pyc
ENV PYTHONDONTWRITEBYTECODE 1
# Définir l'environnement pour que les logs ne soient pas mis en buffer
ENV PYTHONUNBUFFERED 1

# Définir le répertoire de travail
WORKDIR /app

# Créer un utilisateur non-root pour des raisons de sécurité
RUN addgroup --system app && adduser --system --group app

# Installer git et les dépendances système nécessaires
RUN apt-get update && apt-get install -y \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copier requirements et installer les dépendances
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Créer un dossier cache pour Hugging Face
ENV SENTENCE_TRANSFORMERS_HOME=/app/models
RUN mkdir -p /app/models && chown -R app:app /app/models

# Télécharger et stocker localement le modèle all-MiniLM-L6-v2
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2').save('/app/models/all-MiniLM-L6-v2')"

# Copier le reste du code de l'application
COPY . .

# Passer à l'utilisateur non-root
USER app

# Lancer gunicorn (tu peux ajuster en fonction de ton conf)
CMD ["gunicorn", "-c", "gunicorn.conf.py", "main:app"]
