## CVmatcher - Déploiement VPS OVH

### Prérequis
- Ubuntu 22.04 / Debian 12
- Docker + Docker Compose plugin
- Nom de domaine pointant sur le VPS (optionnel mais recommandé)

### Configuration
1) Crée un fichier `.env` à partir de `.env.example` et renseigne:
   - `GEMINI_API_KEY`, `HF_TOKEN`
   - `MONGO_URI` (MongoDB Atlas)
   - `GOOGLE_TOKEN_JSON` (contenu JSON sur une seule ligne)
2) Vérifie `deploy/nginx/default.conf` si tu utilises Nginx en reverse proxy.

### Lancement en local/vps
```bash
docker compose up -d --build
```
- Application: http://SERVER_IP:80 (via Nginx) ou http://SERVER_IP:8000 (port app direct)

### SSL (Let’s Encrypt)
- Installe `certbot` sur l’hôte ou utilise un conteneur webroot; monte les certificats dans `deploy/nginx` et expose 443.

### Maintenance
- Mise à jour images: `docker compose pull && docker compose up -d`
- Logs: `docker compose logs -f app` / `nginx`
- Santé: `/health`, diagnostic: `/diagnostic`

### Structure
```
.
├── app/
│   ├── __init__.py
│   ├── routes.py
│   └── utils/
│       ├── drive_utils.py
│       ├── enrich_db.py
│       └── vectorize.py
├── deploy/
│   └── nginx/
│       └── default.conf
├── docker-compose.yml
├── Dockerfile
├── requirements.txt
├── templates/
├── static/
├── config.py
├── main.py
└── README.md
```
