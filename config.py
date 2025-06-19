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
    """Retourne un client MongoDB avec gestion d'erreurs SSL spécifique Render"""
    from pymongo import MongoClient
    import logging
    logger = logging.getLogger(__name__)
    
    if not MONGO_URI:
        logger.error("❌ MONGO_URI non défini")
        return None
    
    # Configurations spécifiques pour Render + Atlas
    configs = [
        # Config 1: URI avec tlsAllowInvalidCertificates
        {"tlsAllowInvalidCertificates": True},
        
        # Config 2: Désactiver totalement TLS/SSL  
        {"tls": False, "ssl": False},
        
        # Config 3: Configuration par défaut
        {},
        
        # Config 4: TLS avec timeouts courts
        {"tlsAllowInvalidCertificates": True, "serverSelectionTimeoutMS": 3000},
    ]
    
    for i, config in enumerate(configs, 1):
        try:
            # Nettoyer l'URI des paramètres SSL conflictuels pour certaines configs
            uri = MONGO_URI
            if i == 2:  # Config sans SSL
                # Supprimer les paramètres SSL de l'URI
                uri = MONGO_URI.replace("&ssl=true", "").replace("ssl=true&", "").replace("ssl=true", "")
                uri = uri.replace("&retryWrites=true", "").replace("retryWrites=true&", "").replace("retryWrites=true", "")
            
            logger.info(f"🔄 Tentative connexion MongoDB config {i}")
            client = MongoClient(uri, **config)
            
            # Test rapide de connexion
            client.admin.command('ping')
            logger.info(f"✅ MongoDB connecté avec config {i}")
            return client
            
        except Exception as e:
            logger.warning(f"⚠️ Config {i} échouée: {str(e)[:100]}...")
            continue
    
    logger.error("❌ Toutes les configurations MongoDB ont échoué")
    return None
