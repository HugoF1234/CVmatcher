    # main.py
    # Ce fichier expose simplement l'application Flask pour Gunicorn.
    from app import app
    import os

    # Cette section n'est exécutée que si on lance "python main.py" en local.
    # Gunicorn l'ignore en production.
    if __name__ == "__main__":
        port = int(os.environ.get("PORT", 8080))
        debug = os.environ.get("FLASK_DEBUG", "0") == "1"
        app.run(host="0.0.0.0", port=port, debug=debug)
