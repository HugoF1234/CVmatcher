import faiss
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
from bson import ObjectId
import logging
import base64
from config import DB_NAME, COLLECTION_NAME

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
            try:
                # Création du texte combiné pour embedding
                texts_to_embed = []
                
                # Compétences
                competences = cv.get("competences", [])
                if competences and isinstance(competences, list):
                    comp_text = " ".join(competences)
                    if comp_text.strip():
                        texts_to_embed.append(comp_text)
                
                # Biographie
                biographie = cv.get("biographie", "")
                if biographie and isinstance(biographie, str) and biographie.strip():
                    texts_to_embed.append(biographie)
                
                # Expériences - titre et description
                experiences = cv.get("experiences", [])
                if experiences and isinstance(experiences, list):
                    for exp in experiences:
                        if isinstance(exp, dict):
                            titre = exp.get("titre", "")
                            entreprise = exp.get("entreprise", "")
                            description = exp.get("description", "")
                            
                            exp_text = " ".join([
                                str(titre) if titre else "",
                                str(entreprise) if entreprise else "",
                                str(description) if description else ""
                            ]).strip()
                            
                            if exp_text:
                                texts_to_embed.append(exp_text)
                
                # Formations
                formations = cv.get("formations", [])
                if formations and isinstance(formations, list):
                    for form in formations:
                        if isinstance(form, dict):
                            diplome = form.get("diplome", "")
                            etablissement = form.get("etablissement", "")
                            form_text = " ".join([
                                str(diplome) if diplome else "",
                                str(etablissement) if etablissement else ""
                            ]).strip()
                            if form_text:
                                texts_to_embed.append(form_text)
                
                # Secteur
                secteur = cv.get("secteur", [])
                if secteur:
                    if isinstance(secteur, list):
                        secteur_text = " ".join(secteur)
                    else:
                        secteur_text = str(secteur)
                    
                    if secteur_text.strip():
                        texts_to_embed.append(secteur_text)
                
                # Combiner tous les textes
                if texts_to_embed:
                    combined_text = " ".join(texts_to_embed)
                    
                    # Encoder le texte combiné
                    vector = model.encode(combined_text)
                    
                    # Normaliser le vecteur
                    vector = vector / np.linalg.norm(vector)
                    
                    vectors.append(vector)
                    id_mapping.append(str(cv["_id"]))
                    
                    logger.debug(f"✅ Vecteur créé pour {cv.get('nom', 'CV sans nom')}")
                else:
                    logger.warning(f"⚠️ Aucun texte à vectoriser pour {cv.get('nom', 'CV sans nom')}")
                    
            except Exception as e:
                logger.error(f"❌ Erreur vectorisation CV {cv.get('nom', 'inconnu')}: {e}")
                continue

        if not vectors:
            logger.error("❌ Aucun vecteur créé")
            return False

        # Création de l'index FAISS
        vectors_array = np.array(vectors).astype("float32")
        dimension = vectors_array.shape[1]
        
        # Utiliser IndexFlatIP (produit scalaire) plutôt que L2 pour les vecteurs normalisés
        index = faiss.IndexFlatIP(dimension)
        index.add(vectors_array)

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
                "dimension": dimension,
                "model_name": "all-MiniLM-L6-v2"
            },
            upsert=True
        )

        logger.info(f"✅ Index FAISS créé et stocké en base ({len(vectors)} profils, dimension {dimension})")
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

def search_similar_cvs(query_text, top_k=5):
    """Recherche des CVs similaires à la requête"""
    try:
        # Charger l'index
        index, id_mapping = load_faiss_from_mongodb()
        
        if index is None or not id_mapping:
            logger.error("❌ Index FAISS non disponible")
            return []
        
        # Encoder la requête
        model = SentenceTransformer("all-MiniLM-L6-v2")
        query_vector = model.encode(query_text)
        query_vector = query_vector / np.linalg.norm(query_vector)  # Normaliser
        query_vector = np.array([query_vector]).astype("float32")
        
        # Recherche
        scores, indices = index.search(query_vector, top_k)
        
        # Retourner les IDs et scores
        results = []
        for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
            if idx < len(id_mapping):
                results.append({
                    "cv_id": id_mapping[idx],
                    "score": float(score),
                    "rank": i + 1
                })
        
        return results
        
    except Exception as e:
        logger.error(f"❌ Erreur recherche vectorielle: {e}")
        return []
