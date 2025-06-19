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
