## CVmatcher 

CVmatcher scrute un dossier Google Drive, extrait et enrichit des CVs avec Gemini, les stocke dans MongoDB, et les rend recherchables via un index FAISS. Le tout servi par Flask, derrière Nginx, dans de jolis conteneurs.

### En bref (mais technique)
- Ingestion incrémentale: on ne retrait pas toute la base, on ajoute juste les nouveaux CVs
- Cohérence forte: MongoDB, FAISS et Google Drive restent synchronisés (fonction de « sync »)
- Recherche vectorielle: Sentence-Transformers + FAISS (cosine/IP) pour matcher une requête en langage naturel
- Reranking optionnel: Gemini 2.0 Flash réordonne les top profils et explique le « pourquoi »
- UI simple: recherche, détails du profil, likes, téléchargement PDF

### Pile technique
- Backend: Flask + Gunicorn
- Stockage: MongoDB Atlas (collections `CVExtractionCollection`, `faiss_index`, `seen_cvs`)
- Vectorisation: Sentence-Transformers (`paraphrase-MiniLM-L3-v2`) via `HF_TOKEN`
- Similarité: FAISS (IndexFlatIP)
- Enrichissement: Google Gemini 2.0 Flash (`GEMINI_API_KEY`)
- Fichiers: Google Drive API (watcher + download on demand)
- Reverse proxy: Nginx
- Conteneurs: Docker + Docker Compose

### Architecture (vue d’ensemble)
1) Watcher (`app/watcher.py`)
   - Liste les PDFs d’un dossier Drive
   - Télécharge un PDF, extrait le texte, appelle Gemini pour structurer les données
   - Insère en base (MongoDB) et ajoute le vecteur dans FAISS incrémentalement
   - Marque le PDF « vu » seulement si l’insertion MongoDB a réussi
2) Recherche (`app/routes.py`)
   - Requête utilisateur → recherche FAISS → récupération des documents MongoDB
   - Optionnel: reranking par Gemini en 20s max (fallback auto si timeout)
3) Cohérence (`sync_faiss_with_db`)
   - Vérifie et ajoute dans FAISS tout CV présent en base mais absent de l’index

### Données principales
- `CVExtractionCollection`: documents structurés (nom, compétences, expériences, biographie, secteur, nomdupdf…)
- `faiss_index`: index + mapping des IDs MongoDB
- `seen_cvs`: tracking des IDs Drive déjà traités

### Lancer le projet (Docker Compose)

1) Copier `.env.example` → `.env` et remplir:
```
GEMINI_API_KEY=...
HF_TOKEN=...
MONGO_URI=...
GOOGLE_TOKEN_JSON={ ... }  # sur une seule ligne
GOOGLE_DRIVE_FOLDER_ID=...
SECRET_KEY=...
```

2) Démarrer
```bash
docker compose up -d --build
```
Accès:
- App via Nginx: http://SERVER_IP
- Santé: `/health`  •  Diagnostic: `/diagnostic`  •  UI Diagnostic: `/diagnostic-ui`

### HTTPS (Let’s Encrypt, webroot)
1) Pointer ton domaine sur l’IP du serveur
2) Générer le certificat avec Certbot (webroot) et monter les certs dans Nginx
3) Activer le bloc `listen 443 ssl;` dans `deploy/nginx/default.conf`

Renouvellement (cron):
```bash
docker compose run --rm certbot renew && docker compose exec nginx nginx -s reload
```

### Endpoints utiles
- `POST /update-cvs`: lance la mise à jour en arrière-plan (thread)
- `GET /update-status`: nombre de CVs et statut FAISS
- `POST /clean-index`: recrée FAISS depuis la base
- `GET /download/<nomdupdf>`: télécharge le PDF original depuis Drive

### Astuces perf & stabilité
- Gunicorn: timeout à 120s pour absorber les pics (voir `gunicorn.conf.py`)
- Mémoire: ne jamais faire `list(collection.find({}))` — itérer sur le curseur
- HF 429: fournir `HF_TOKEN`, le modèle se charge côté container
- Timeouts Gemini: appels encapsulés avec `ThreadPoolExecutor` + fallback FAISS
- Sync: lancer `sync_faiss_with_db` après ingestion pour garantir la cohérence

### Développement local (sans Docker)
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
export GEMINI_API_KEY=... HF_TOKEN=... MONGO_URI=... SECRET_KEY=...
export GOOGLE_TOKEN_JSON='{"type":"service_account",...}'
export GOOGLE_DRIVE_FOLDER_ID=...
gunicorn --bind :8080 --config gunicorn.conf.py main:app
```

### Structure du repo
```
.
├── app/
│   ├── __init__.py
│   ├── routes.py
│   └── utils/
│       ├── drive_utils.py
│       ├── enrich_db.py
│       └── vectorize.py
├── deploy/
│   └── nginx/
│       └── default.conf
├── docker-compose.yml
├── Dockerfile
├── requirements.txt
├── templates/
├── static/
├── config.py
├── main.py
└── README.md
```

### Questions fréquentes
- Pourquoi c’est parfois « long » pour un seul CV ?
  - Drive, extraction PDF, Gemini (structuration + enrichissement), Mongo, FAISS: tout s’enchaîne. Les timeouts sont maîtrisés, et l’ingestion est incrémentale.
- Comment éviter les décalages Mongo/FAISS ?
  - On n’ajoute dans `seen_cvs` qu’après insertion MongoDB. Et on a `sync_faiss_with_db` en fin de run.
- Et si Gemini tombe ?
  - Les appels ont un retry + backoff. En recherche, on retombe sur le score FAISS si le reranking dépasse le timeout.

Bon match! 🧑‍💻🧠📄
