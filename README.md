cv-matcher/
│
├── app/                      # Code de l'application
│   ├── __init__.py
│   ├── routes.py            # (Contient le contenu de `webapp2.py`)
│   ├── utils/
│   │   ├── drive_utils.py   # (Contient les fonctions de `TelechargementDrive.py`)
│   │   ├── enrich_db.py     # (Contient `RemplissageMongoDB.py` divisé en fonctions)
│   │   ├── vectorize.py     # (Contient `vectorisation.py`)
│
├── static/
│   └── img/
│       └── 9ghfmil2.png     # Logo utilisé
│
├── templates/
│   ├── index.html
│   ├── likes.html           # À créer pour la page de favoris
│   ├── cv_detail.html       # À créer pour la page détail
│
├── credentials/
│   ├── credentials.json     # Pour accès à Google Drive
│   ├── token.json
│
├── faiss_index/
│   ├── cv_index.faiss
│   ├── id_mapping.pkl
│
├── .env                     # Clés API comme GEMINI_API_KEY, Mongo URI
├── requirements.txt
├── Procfile                 # Pour déploiement Render
├── config.py                # Centralise les constantes (Mongo URI, DB, etc.)
├── main.py                  # Lance l’app Flask (appel `app` depuis app/)
└── README.md
