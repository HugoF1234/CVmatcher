# gunicorn.conf.py
import os

# Configuration du serveur
bind = f"0.0.0.0:{os.environ.get('PORT', 10000)}"
workers = 1  # Un seul worker pour éviter les conflits de modèles ML
worker_class = "sync"
worker_connections = 1000

# Timeouts optimisés
timeout = 60  # 60 secondes pour les requêtes
keepalive = 5
max_requests = 500
max_requests_jitter = 100

# Optimisations mémoire
preload_app = True  # Charger l'app avant de fork les workers
worker_tmp_dir = "/dev/shm"  # Utiliser la RAM pour les fichiers temporaires

# Logging
accesslog = "-"
errorlog = "-"
loglevel = "info"
access_log_format = '%(h)s %(l)s %(u)s %(t)s "%(r)s" %(s)s %(b)s "%(f)s" "%(a)s" %(D)s'

# Redémarrage automatique
worker_timeout = 60
graceful_timeout = 30
