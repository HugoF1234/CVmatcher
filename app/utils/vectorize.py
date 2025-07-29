import faiss
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
from bson import ObjectId
import logging
import base64
import tempfile
import os
from config import DB_NAME, COLLECTION_NAME

logger = logging.getLogger(__name__)

# Debug: Vérifier que le token est disponible
from config import HF_TOKEN
if HF_TOKEN:
    logger.info("✅ HF_TOKEN trouvé dans la configuration")
    # Masquer le token pour la sécurité (afficher seulement les premiers caractères)
    masked_token = HF_TOKEN[:8] + "..." + HF_TOKEN[-4:] if len(HF_TOKEN) > 12 else "***"
    logger.info(f"🔑 Token: {masked_token}")
else:
    logger.warning("⚠️ HF_TOKEN non trouvé dans la configuration")

# Variable globale pour le modèle
_model = None



def get_model():
    """Charge le modèle SentenceTransformer de manière lazy avec retry"""
    global _model
    
    if _model is not None:
        return _model
    
    try:
        logger.info("🔄 Chargement du modèle SentenceTransformer...")
        
        # Utiliser le modèle local téléchargé dans le Dockerfile
        _model = SentenceTransformer("/app/models/all-MiniLM-L6-v2")
        logger.info("✅ Modèle SentenceTransformer chargé avec succès (local)")
        return _model
    except Exception as e:
        logger.error(f"❌ Erreur lors du chargement du modèle local: {e}")
        logger.info("🔄 Tentative avec le modèle distant...")
        
        # Fallback vers le modèle distant si le local échoue
        try:
            _model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", token=HF_TOKEN)
            logger.info("✅ Modèle SentenceTransformer chargé avec succès (distant)")
            return _model
        except Exception as e2:
            logger.error(f"❌ Échec définitif du chargement du modèle: {e2}")
            return None

def reset_model():
    """Réinitialise le modèle (utile pour les tests)"""
    global _model
    _model = None
    logger.info("🔄 Modèle SentenceTransformer réinitialisé")

def update_faiss_index(client=None):
    """Met à jour l'index FAISS et le stocke dans MongoDB"""
    try:
        logger.info("🔄 Début création index FAISS")
        
        # Vérifier que le modèle est disponible
        if get_model() is None:
            logger.error("❌ Modèle SentenceTransformer non disponible")
            return False
        
        # Utiliser le modèle global déjà chargé
        # model = SentenceTransformer("all-MiniLM-L6-v2")  # SUPPRIMÉ
        
        if client is None:
            from config import get_mongo_client
            client = get_mongo_client()
            
        if not client:
            logger.error("❌ Impossible de se connecter à MongoDB")
            return False
            
        db = client[DB_NAME]
        collection = db[COLLECTION_NAME]
        index_collection = db["faiss_index"]
        
        # cvs = list(collection.find({}))
        # logger.info(f"📊 Traitement de {len(cvs)} CVs pour indexation")
        logger.info("📊 Début du traitement des CVs pour indexation (streaming)")
        vectors = []
        id_mapping = []
        processed_count = 0

        for cv in collection.find({}):
            try:
                # Construire le texte à vectoriser
                text_parts = []
                
                # Nom
                nom = cv.get("nom", "")
                if nom and isinstance(nom, str):
                    text_parts.append(nom)
                
                # Compétences
                competences = cv.get("competences", [])
                if competences and isinstance(competences, list):
                    comp_text = " ".join([str(c) for c in competences if c])
                    if comp_text.strip():
                        text_parts.append(comp_text)
                
                # Biographie
                biographie = cv.get("biographie", "")
                if biographie and isinstance(biographie, str) and biographie.strip():
                    text_parts.append(biographie)
                
                # Secteur
                secteur = cv.get("secteur", [])
                if secteur:
                    if isinstance(secteur, list):
                        secteur_text = " ".join([str(s) for s in secteur if s])
                    else:
                        secteur_text = str(secteur)
                    
                    if secteur_text.strip():
                        text_parts.append(secteur_text)
                
                # Expériences (titre + entreprise + description)
                experiences = cv.get("experiences", [])
                if experiences and isinstance(experiences, list):
                    for exp in experiences[:3]:  # Limiter aux 3 premières expériences
                        if isinstance(exp, dict):
                            exp_parts = []
                            
                            titre = exp.get("titre", "")
                            if titre:
                                exp_parts.append(str(titre))
                            
                            entreprise = exp.get("entreprise", "")
                            if entreprise:
                                exp_parts.append(str(entreprise))
                            
                            description = exp.get("description", "")
                            if description and len(str(description)) > 10:  # Ignorer les descriptions trop courtes
                                exp_parts.append(str(description)[:500])  # Limiter à 500 caractères
                            
                            if exp_parts:
                                text_parts.append(" ".join(exp_parts))
                
                # Formations
                formations = cv.get("formations", [])
                if formations and isinstance(formations, list):
                    for form in formations[:2]:  # Limiter aux 2 premières formations
                        if isinstance(form, dict):
                            form_parts = []
                            
                            diplome = form.get("diplome", "")
                            if diplome:
                                form_parts.append(str(diplome))
                            
                            etablissement = form.get("etablissement", "")
                            if etablissement:
                                form_parts.append(str(etablissement))
                            
                            if form_parts:
                                text_parts.append(" ".join(form_parts))
                
                # Créer le texte final
                if text_parts:
                    combined_text = " ".join(text_parts)
                    
                    # Nettoyer le texte
                    combined_text = " ".join(combined_text.split())  # Supprimer espaces multiples
                    combined_text = combined_text[:2000]  # Limiter la taille
                    
                    if len(combined_text) > 10:  # Minimum 10 caractères
                        # Encoder
                        vector = get_model().encode(combined_text)
                        
                        # Vérifier que le vecteur est valide
                        if vector is not None and len(vector) > 0:
                            # Normaliser
                            norm = np.linalg.norm(vector)
                            if norm > 0:
                                vector = vector / norm
                                
                                vectors.append(vector.astype(np.float32))
                                id_mapping.append(str(cv["_id"]))
                                processed_count += 1
                                
                                if processed_count % 10 == 0:
                                    logger.info(f"📊 {processed_count} CVs vectorisés...")
                            else:
                                logger.warning(f"⚠️ Vecteur nul pour {cv.get('nom', 'CV sans nom')}")
                        else:
                            logger.warning(f"⚠️ Échec encodage pour {cv.get('nom', 'CV sans nom')}")
                    else:
                        logger.warning(f"⚠️ Texte trop court pour {cv.get('nom', 'CV sans nom')}")
                else:
                    logger.warning(f"⚠️ Aucun texte extractible pour {cv.get('nom', 'CV sans nom')}")
                    
            except Exception as e:
                logger.error(f"❌ Erreur vectorisation CV {cv.get('nom', 'inconnu')}: {e}")
                continue

        if not vectors:
            logger.error("❌ Aucun vecteur créé")
            return False

        logger.info(f"✅ {len(vectors)} vecteurs créés avec succès")

        # Création de l'index FAISS
        vectors_array = np.array(vectors, dtype=np.float32)
        dimension = vectors_array.shape[1]
        
        logger.info(f"📐 Dimension des vecteurs: {dimension}")
        
        # Utiliser IndexFlatIP pour vecteurs normalisés (produit scalaire)
        index = faiss.IndexFlatIP(dimension)
        index.add(vectors_array)
        
        logger.info(f"🔍 Index FAISS créé avec {index.ntotal} vecteurs")

        # Sérialisation ROBUSTE avec fichier temporaire
        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            temp_path = tmp_file.name
            
        try:
            # Écrire l'index dans un fichier temporaire
            faiss.write_index(index, temp_path)
            
            # Lire le fichier et encoder en base64
            with open(temp_path, 'rb') as f:
                index_bytes = f.read()
            index_b64 = base64.b64encode(index_bytes).decode('utf-8')
            
        finally:
            # Nettoyer le fichier temporaire
            if os.path.exists(temp_path):
                os.unlink(temp_path)
        
        # Sérialiser le mapping
        mapping_bytes = pickle.dumps(id_mapping)
        mapping_b64 = base64.b64encode(mapping_bytes).decode('utf-8')

        # Supprimer l'ancien index si il existe
        index_collection.delete_one({"_id": "faiss_data"})
        
        # Stockage dans MongoDB
        doc_to_insert = {
            "_id": "faiss_data",
            "index": index_b64,
            "id_mapping": mapping_b64,
            "vector_count": len(vectors),
            "dimension": dimension,
            "model_name": "sentence-transformers/all-MiniLM-L6-v2",
            "index_type": "IndexFlatIP"
        }
        
        index_collection.insert_one(doc_to_insert)
        
        logger.info(f"💾 Index FAISS stocké en base ({len(vectors)} profils, dimension {dimension})")
        return True
        
    except Exception as e:
        logger.error(f"❌ Erreur création index FAISS: {e}")
        import traceback
        logger.error(f"Stack trace: {traceback.format_exc()}")
        return False

def load_faiss_from_mongodb(client=None):
    """Charge l'index FAISS depuis MongoDB"""
    try:
        if client is None:
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
        
        logger.info(f"📖 Chargement index FAISS ({faiss_doc.get('vector_count', 0)} vecteurs)")
        
        # Désérialisation ROBUSTE avec fichier temporaire
        index_b64 = faiss_doc.get("index")
        if not index_b64:
            logger.error("❌ Données index manquantes")
            return None, []
        
        # Décoder et écrire dans un fichier temporaire
        index_bytes = base64.b64decode(index_b64)
        
        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            temp_path = tmp_file.name
            tmp_file.write(index_bytes)
        
        try:
            # Charger l'index depuis le fichier temporaire
            index = faiss.read_index(temp_path)
            
        finally:
            # Nettoyer le fichier temporaire
            if os.path.exists(temp_path):
                os.unlink(temp_path)
        
        # Chargement du mapping
        mapping_b64 = faiss_doc.get("id_mapping")
        if not mapping_b64:
            logger.error("❌ Mapping des IDs manquant")
            return None, []
            
        mapping_bytes = base64.b64decode(mapping_b64)
        id_mapping = pickle.loads(mapping_bytes)
        
        logger.info(f"✅ Index FAISS chargé: {len(id_mapping)} profils, dimension {index.d}")
        return index, id_mapping
        
    except Exception as e:
        logger.error(f"❌ Erreur chargement index FAISS: {e}")
        import traceback
        logger.error(f"Stack trace: {traceback.format_exc()}")
        return None, []

def search_similar_cvs(query_text, top_k=5):
    """Recherche des CVs similaires à la requête"""
    try:
        # Vérifier que le modèle est disponible
        if get_model() is None:
            logger.error("❌ Modèle SentenceTransformer non disponible pour la recherche")
            return []
        
        # Charger l'index
        index, id_mapping = load_faiss_from_mongodb()
        
        if index is None or not id_mapping:
            logger.error("❌ Index FAISS non disponible pour la recherche")
            return []
        
        logger.info(f"🔍 Recherche pour: '{query_text}' (top {top_k})")
        
        # Utiliser le modèle global déjà chargé
        # model = SentenceTransformer("all-MiniLM-L6-v2")  # SUPPRIMÉ
        query_vector = get_model().encode(query_text)
        
        # Normaliser
        norm = np.linalg.norm(query_vector)
        if norm > 0:
            query_vector = query_vector / norm
        
        query_vector = np.array([query_vector], dtype=np.float32)
        
        # Recherche
        scores, indices = index.search(query_vector, min(top_k, len(id_mapping)))
        
        # Retourner les résultats
        results = []
        for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
            if idx >= 0 and idx < len(id_mapping):  # Vérifier la validité de l'index
                results.append({
                    "cv_id": id_mapping[idx],
                    "score": float(score),
                    "rank": i + 1
                })
        
        logger.info(f"✅ {len(results)} résultats trouvés")
        return results
        
    except Exception as e:
        logger.error(f"❌ Erreur recherche vectorielle: {e}")
        import traceback
        logger.error(f"Stack trace: {traceback.format_exc()}")
        return []

def clean_faiss_index(client=None):
    """Nettoie l'index FAISS en base (utilitaire de debug)"""
    try:
        if client is None:
            from config import get_mongo_client
            client = get_mongo_client()
        if not client:
            logger.error("❌ Impossible de se connecter à MongoDB")
            return False
            
        db = client[DB_NAME]
        index_collection = db["faiss_index"]
        
        # Supprimer l'ancien index
        result = index_collection.delete_one({"_id": "faiss_data"})
        
        if result.deleted_count > 0:
            logger.info("✅ Index FAISS nettoyé")
            return True
        else:
            logger.info("ℹ️ Aucun index à nettoyer")
            return True
            
    except Exception as e:
        logger.error(f"❌ Erreur nettoyage index: {e}")
        return False

def add_cv_to_faiss_index(cv):
    """Ajoute un CV à l'index FAISS de façon incrémentale (sans tout revectoriser)"""
    import base64
    import pickle
    import numpy as np
    import faiss
    import tempfile
    import os
    from config import get_mongo_client, DB_NAME, COLLECTION_NAME
    logger = logging.getLogger(__name__)

    # Vérifier que le modèle est disponible
    if get_model() is None:
        logger.error("❌ Modèle SentenceTransformer non disponible pour l'ajout incrémental")
        return False

    # Charger l'index et le mapping existants
    index, id_mapping = load_faiss_from_mongodb()
    if index is None or id_mapping is None:
        logger.error("❌ Impossible de charger l'index FAISS existant pour ajout incrémental")
        return False

    # Construire le texte à vectoriser (reprend la logique de update_faiss_index)
    text_parts = []
    nom = cv.get("nom", "")
    if nom and isinstance(nom, str):
        text_parts.append(nom)
    competences = cv.get("competences", [])
    if competences and isinstance(competences, list):
        comp_text = " ".join([str(c) for c in competences if c])
        if comp_text.strip():
            text_parts.append(comp_text)
    biographie = cv.get("biographie", "")
    if biographie and isinstance(biographie, str) and biographie.strip():
        text_parts.append(biographie)
    secteur = cv.get("secteur", [])
    if secteur:
        if isinstance(secteur, list):
            secteur_text = " ".join([str(s) for s in secteur if s])
        else:
            secteur_text = str(secteur)
        if secteur_text.strip():
            text_parts.append(secteur_text)
    experiences = cv.get("experiences", [])
    if experiences and isinstance(experiences, list):
        for exp in experiences[:3]:
            if isinstance(exp, dict):
                exp_parts = []
                titre = exp.get("titre", "")
                if titre:
                    exp_parts.append(str(titre))
                entreprise = exp.get("entreprise", "")
                if entreprise:
                    exp_parts.append(str(entreprise))
                description = exp.get("description", "")
                if description and len(str(description)) > 10:
                    exp_parts.append(str(description)[:500])
                if exp_parts:
                    text_parts.append(" ".join(exp_parts))
    formations = cv.get("formations", [])
    if formations and isinstance(formations, list):
        for form in formations[:2]:
            if isinstance(form, dict):
                form_parts = []
                diplome = form.get("diplome", "")
                if diplome:
                    form_parts.append(str(diplome))
                etablissement = form.get("etablissement", "")
                if etablissement:
                    form_parts.append(str(etablissement))
                if form_parts:
                    text_parts.append(" ".join(form_parts))
    if not text_parts:
        logger.warning(f"⚠️ Aucun texte extractible pour {cv.get('nom', 'CV sans nom')}")
        return False
    combined_text = " ".join(text_parts)
    combined_text = " ".join(combined_text.split())
    combined_text = combined_text[:2000]
    if len(combined_text) <= 10:
        logger.warning(f"⚠️ Texte trop court pour {cv.get('nom', 'CV sans nom')}")
        return False
    # Vectorisation
    vector = get_model().encode(combined_text)
    norm = np.linalg.norm(vector)
    if norm > 0:
        vector = vector / norm
    else:
        logger.warning(f"⚠️ Vecteur nul pour {cv.get('nom', 'CV sans nom')}")
        return False
    vector = np.array([vector], dtype=np.float32)
    # Ajout au FAISS index
    index.add(vector)
    id_mapping.append(str(cv["_id"]))
    # Sauvegarde dans MongoDB
    try:
        from config import get_mongo_client
        client = get_mongo_client()
        if not client:
            logger.error("❌ Impossible de se connecter à MongoDB pour sauvegarde incrémentale")
            return False
        db = client[DB_NAME]
        index_collection = db["faiss_index"]
        # Sérialisation FAISS
        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            temp_path = tmp_file.name
        try:
            faiss.write_index(index, temp_path)
            with open(temp_path, 'rb') as f:
                index_bytes = f.read()
            index_b64 = base64.b64encode(index_bytes).decode('utf-8')
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)
        # Sérialisation mapping
        mapping_bytes = pickle.dumps(id_mapping)
        mapping_b64 = base64.b64encode(mapping_bytes).decode('utf-8')
        # Mise à jour du document en base
        index_collection.update_one(
            {"_id": "faiss_data"},
            {"$set": {
                "index": index_b64,
                "id_mapping": mapping_b64,
                "vector_count": len(id_mapping),
                "dimension": index.d,
                "model_name": "sentence-transformers/all-MiniLM-L6-v2",
                "index_type": "IndexFlatIP"
            }},
            upsert=True
        )
        logger.info(f"✅ Nouveau CV ajouté à l'index FAISS (total: {len(id_mapping)})")
        return True
    except Exception as e:
        logger.error(f"❌ Erreur sauvegarde incrémentale FAISS: {e}")
        return False

def sync_faiss_with_db(client=None):
    """
    Vérifie la cohérence entre la BDD et l'index FAISS.
    Ajoute à FAISS tout CV présent en BDD mais absent du mapping.
    """
    from config import DB_NAME, COLLECTION_NAME
    if client is None:
        from config import get_mongo_client
        client = get_mongo_client()
    if not client:
        logger.error("❌ Impossible de se connecter à MongoDB pour la synchronisation FAISS")
        return False
    db = client[DB_NAME]
    collection = db[COLLECTION_NAME]
    cvs = list(collection.find({}))
    index, id_mapping = load_faiss_from_mongodb(client=client)
    id_mapping_set = set(id_mapping)
    added = 0
    for cv in cvs:
        if str(cv['_id']) not in id_mapping_set:
            logger.warning(f"🟡 CV manquant dans FAISS: {cv.get('nom', 'inconnu')} ({cv['_id']})")
            add_cv_to_faiss_index(cv)
            added += 1
    logger.info(f"✅ Synchronisation FAISS terminée. {added} CVs ajoutés à FAISS.")
    return True
