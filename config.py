import os

# === CONFIGURATION CENTRALISÉE ===
MONGO_URI = os.environ.get("MONGO_URI")
DB_NAME = "CVExtraction"
COLLECTION_NAME = "CVExtractionCollection"
FAISS_INDEX_FILE = "faiss_index/cv_index.faiss"
ID_MAPPING_FILE = "faiss_index/id_mapping.pkl"
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
HF_TOKEN = os.environ.get("HF_TOKEN")
GOOGLE_DRIVE_FOLDER_ID = os.environ.get("GOOGLE_DRIVE_FOLDER_ID", "")
TOP_K = 5
SECRET_KEY = os.environ.get("SECRET_KEY", "change_me_for_prod")

# Créer le dossier faiss_index s'il n'existe pas
os.makedirs("faiss_index", exist_ok=True)

def get_mongo_client():
    """Retourne un client MongoDB pour Render + Atlas avec ServerApi"""
    from pymongo.mongo_client import MongoClient
    from pymongo.server_api import ServerApi
    import logging
    logger = logging.getLogger(__name__)
    
    if not MONGO_URI:
        logger.error("❌ MONGO_URI non défini")
        return None
    
    try:
        logger.info("🔄 Connexion à MongoDB Atlas avec ServerApi...")
        
        # Configuration avec ServerApi comme dans votre test réussi
        client = MongoClient(
            MONGO_URI, 
            server_api=ServerApi('1'),
            serverSelectionTimeoutMS=10000,
            socketTimeoutMS=20000,
            connectTimeoutMS=20000
        )
        
        # Test de connexion
        client.admin.command('ping')
        logger.info("✅ MongoDB Atlas connecté avec succès")
        return client
        
    except Exception as e:
        logger.error(f"❌ Erreur de connexion MongoDB: {str(e)}")
        return None
