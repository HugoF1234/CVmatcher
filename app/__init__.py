from flask import Flask
import os
from config import SECRET_KEY

# Configuration des chemins
template_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'templates'))
static_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'static'))

# Créer l'application Flask
app = Flask(__name__, 
            template_folder=template_dir,
            static_folder=static_dir)

app.secret_key = SECRET_KEY

# Importer les routes après la création de l'app pour éviter l'importation circulaire
def register_routes():
    from app import routes
    return app

# Enregistrer les routes
register_routes()
