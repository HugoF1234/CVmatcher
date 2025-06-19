import os

# === CONFIGURATION CENTRALISÉE ===
MONGO_URI = os.environ.get("MONGO_URI")
DB_NAME = "CVExtraction"
COLLECTION_NAME = "CVExtractionCollection"
FAISS_INDEX_FILE = "faiss_index/cv_index.faiss"
ID_MAPPING_FILE = "faiss_index/id_mapping.pkl"
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "AIzaSyB2w2GCNE6EqvAtcA2Dj9rsvkD2YzFFMfM")
TOP_K = 5
SECRET_KEY = "0180529a5b9c1ec478296df826a91c31"

# Créer le dossier faiss_index s'il n'existe pas
os.makedirs("faiss_index", exist_ok=True)

def get_mongo_client():
    """Retourne un client MongoDB avec gestion d'erreurs SSL"""
    from pymongo import MongoClient
    import logging
    logger = logging.getLogger(__name__)
    
    if not MONGO_URI:
        logger.error("❌ MONGO_URI non défini")
        return None
    
    # Essayer plusieurs configurations
    configs = [
        {},  # Configuration par défaut
        {"tlsAllowInvalidCertificates": True},
        {"ssl": False},
        {"directConnection": True}
    ]
    
    for i, config in enumerate(configs):
        try:
            client = MongoClient(MONGO_URI, **config)
            # Test de connexion
            client.admin.command('ping')
            logger.info(f"✅ MongoDB connecté (config {i+1})")
            return client
        except Exception as e:
            logger.warning(f"⚠️ Config {i+1} échouée: {str(e)[:100]}...")
            continue
    
    logger.error("❌ Impossible de se connecter à MongoDB")
    return None
