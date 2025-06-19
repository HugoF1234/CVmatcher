import faiss
import pickle
import numpy as np
from pymongo import MongoClient
from sentence_transformers import SentenceTransformer
from bson import ObjectId
import logging
import base64
from config import MONGO_URI, DB_NAME, COLLECTION_NAME

logger = logging.getLogger(__name__)

def update_faiss_index():
    """Met à jour l'index FAISS et le stocke dans MongoDB"""
    try:
        # Lazy load du model pour éviter les imports au niveau module
        model = SentenceTransformer("all-MiniLM-L6-v2")
        
        from config import get_mongo_client
        client = get_mongo_client()
        
        if not client:
            logger.error("❌ Impossible de se connecter à MongoDB")
            return False
            
        db = client[DB_NAME]
        collection = db[COLLECTION_NAME]
        index_collection = db["faiss_index"]
        
        cvs = list(collection.find({}))
        logger.info(f"🔄 Création index FAISS pour {len(cvs)} CVs")

        if not cvs:
            logger.warning("⚠️ Aucun CV trouvé pour l'indexation")
            return False

        vectors = []
        id_mapping = []

        for cv in cvs:
            # Création du vecteur combiné
            comp = " ".join(cv.get("competences", []))
            bio = cv.get("biographie", "")
            exp = " ".join([e.get("description", "") for e in cv.get("experiences", [])])

            # Encodage avec pondération
            emb_comp = model.encode(comp) if comp else np.zeros(384)
            emb_bio = model.encode(bio) if bio else np.zeros(384)
            emb_exp = model.encode(exp) if exp else np.zeros(384)

            weighted_vector = (
                0.2 * emb_comp +
                0.4 * emb_bio +
                0.6 * emb_exp
            )

            vectors.append(weighted_vector)
            id_mapping.append(str(cv["_id"]))

        # Création de l'index FAISS
        dimension = len(vectors[0])
        index = faiss.IndexFlatL2(dimension)
        index.add(np.array(vectors).astype("float32"))

        # Sérialisation pour stockage MongoDB
        index_bytes = faiss.serialize_index(index)
        index_b64 = base64.b64encode(index_bytes).decode('utf-8')
        
        mapping_bytes = pickle.dumps(id_mapping)
        mapping_b64 = base64.b64encode(mapping_bytes).decode('utf-8')

        # Stockage dans MongoDB
        index_collection.replace_one(
            {"_id": "faiss_data"},
            {
                "_id": "faiss_data",
                "index": index_b64,
                "id_mapping": mapping_b64,
                "vector_count": len(vectors),
                "dimension": dimension
            },
            upsert=True
        )

        logger.info(f"✅ Index FAISS créé et stocké en base ({len(vectors)} profils)")
        return True
        
    except Exception as e:
        logger.error(f"❌ Erreur création index FAISS: {e}")
        return False

def load_faiss_from_mongodb():
    """Charge l'index FAISS depuis MongoDB"""
    try:
        from config import get_mongo_client
        client = get_mongo_client()
        
        if not client:
            logger.error("❌ Impossible de se connecter à MongoDB pour FAISS")
            return None, []
            
        db = client[DB_NAME]
        index_collection = db["faiss_index"]
        
        faiss_doc = index_collection.find_one({"_id": "faiss_data"})
        if not faiss_doc:
            logger.warning("⚠️ Aucun index FAISS trouvé en base")
            return None, []
        
        # Désérialisation
        index_bytes = base64.b64decode(faiss_doc["index"])
        index = faiss.deserialize_index(index_bytes)
        
        mapping_bytes = base64.b64decode(faiss_doc["id_mapping"])
        id_mapping = pickle.loads(mapping_bytes)
        
        logger.info(f"✅ Index FAISS chargé depuis MongoDB ({len(id_mapping)} profils)")
        return index, id_mapping
        
    except Exception as e:
        logger.error(f"❌ Erreur chargement index FAISS: {e}")
        return None, []
