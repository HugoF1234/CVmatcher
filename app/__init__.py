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

# Importer les routes (cela va les enregistrer automatiquement)
from app import routes
