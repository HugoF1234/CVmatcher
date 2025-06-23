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

def update_faiss_index():
    """Met à jour l'index FAISS et le stocke dans MongoDB"""
    try:
        logger.info("🔄 Début création index FAISS")
        
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
        logger.info(f"📊 Traitement de {len(cvs)} CVs pour indexation")

        if not cvs:
            logger.warning("⚠️ Aucun CV trouvé pour l'indexation")
            return False

        vectors = []
        id_mapping = []
        processed_count = 0

        for cv in cvs:
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
                        vector = model.encode(combined_text)
                        
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
            "model_name": "all-MiniLM-L6-v2",
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
        # Charger l'index
        index, id_mapping = load_faiss_from_mongodb()
        
        if index is None or not id_mapping:
            logger.error("❌ Index FAISS non disponible pour la recherche")
            return []
        
        logger.info(f"🔍 Recherche pour: '{query_text}' (top {top_k})")
        
        # Encoder la requête
        model = SentenceTransformer("all-MiniLM-L6-v2")
        query_vector = model.encode(query_text)
        
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

def clean_faiss_index():
    """Nettoie l'index FAISS en base (utilitaire de debug)"""
    try:
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
