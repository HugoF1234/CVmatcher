    # main.py
    from app import app
    import os

    if __name__ == "__main__":
        # Cette partie est utile pour le développement local,
        # mais Gunicorn ne l'exécutera pas en production.
        port = int(os.environ.get("PORT", 8080))
        # Pour le local, on active le mode debug.
        # Sur Cloud Run, FLASK_DEBUG ne sera pas défini.
        debug = os.environ.get("FLASK_DEBUG") == "1"
        app.run(host="0.0.0.0", port=port, debug=debug)
