from flask import Flask
import os
from config import SECRET_KEY

template_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'templates'))
static_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'static'))

app = Flask(__name__, 
            template_folder=template_dir,
            static_folder=static_dir)

app.secret_key = SECRET_KEY

from app import routes
