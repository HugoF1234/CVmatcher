from app import app
import os
import logging

# Configuration du logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def startup():
    logger.info("🚀 Application démarrée")
    logger.info(f"Port: {os.environ.get('PORT', 10000)}")
    logger.info(f"MONGO_URI défini: {'Oui' if os.environ.get('MONGO_URI') else 'Non'}")
    logger.info(f"GEMINI_API_KEY défini: {'Oui' if os.environ.get('GEMINI_API_KEY') else 'Non'}")

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    debug = os.environ.get("FLASK_ENV") == "development"
    
    startup()
    logger.info(f"🚀 Démarrage sur le port {port}")
    app.run(host="0.0.0.0", port=port, debug=debug)
