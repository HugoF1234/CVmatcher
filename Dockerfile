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

# Mettre à jour pip et installer les dépendances
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# --- Configuration du cache pour sentence-transformers ---
# On définit une variable d'environnement pour que le cache soit dans un dossier accessible en écriture
ENV SENTENCE_TRANSFORMERS_HOME=/app/.cache
# Créer le dossier de cache et donner les permissions
RUN mkdir -p /app/.cache && chown -R app:app /app/.cache

# Copier le reste du code de l'application
COPY . .

# Changer le propriétaire des fichiers pour l'utilisateur non-root
RUN chown -R app:app /app

# Changer d'utilisateur
USER app

# Exposer le port que Cloud Run utilisera
# Cloud Run injecte la variable d'environnement PORT, par défaut 8080
EXPOSE 8080

# Commande pour lancer l'application avec Gunicorn
CMD exec gunicorn --bind :$PORT --config gunicorn.conf.py main:app
