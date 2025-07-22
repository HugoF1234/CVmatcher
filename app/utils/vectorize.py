import base64
import faiss
import pickle
import logging
import time
import os
import numpy as np

from sentence_transformers import SentenceTransformer

# Assurez-vous que DB_NAME, COLLECTION_NAME et get_mongo_client sont bien importés depuis votre fichier config
from config import DB_NAME, COLLECTION_NAME, get_mongo_client

logger = logging.getLogger(__name__)

# Le modèle SentenceTransformer est initialisé une seule fois au niveau du module
# pour être partagé par toutes les fonctions et optimiser les ressources.
try:
    # Utilisation de "paraphrase-MiniLM-L3-v2" comme convenu
    model = SentenceTransformer("paraphrase-MiniLM-L3-v2")
    logger.info("✅ Modèle SentenceTransformer 'paraphrase-MiniLM-L3-v2' chargé au démarrage.")
except Exception as e:
    logger.error(f"❌ Erreur lors du chargement initial du modèle SentenceTransformer: {e}")
    # Si le modèle ne peut pas être chargé, on le met à None pour éviter des erreurs futures.
    # Les fonctions qui en dépendent devront vérifier cette valeur.
    model = None 

def update_faiss_index():
    """
    Crée/met à jour un index FAISS à partir des CVs de MongoDB et le sauvegarde dans MongoDB.
    Inclut des logs détaillés pour le diagnostic des problèmes de vectorisation.
    """
    if model is None:
        logger.error("❌ Impossible de mettre à jour l'index FAISS, le modèle SentenceTransformer n'a pas été chargé.")
        return False

    start_time_total = time.time()
    logger.info("🔄 Début création/mise à jour de l'index FAISS")
    
    try:
        client = get_mongo_client()
        
        if not client:
            logger.error("❌ Impossible de se connecter à MongoDB pour la mise à jour de l'index.")
            return False
            
        db = client[DB_NAME]
        collection = db[COLLECTION_NAME] # La collection où sont stockés les CVs
        index_collection = db["faiss_index"] # La collection où l'index FAISS est sauvegardé

        # Récupération de tous les CVs
        cvs = list(collection.find({}))
        logger.info(f"📊 {len(cvs)} CVs trouvés dans MongoDB pour indexation.")

        if not cvs:
            logger.warning("⚠️ Aucun CV trouvé pour l'indexation. L'index FAISS sera vide.")
            # Créer un index FAISS vide pour éviter des erreurs au chargement si la DB est vide
            empty_index = faiss.IndexFlatL2(model.get_sentence_embedding_dimension())
            empty_id_mapping = []
            
            index_collection.replace_one(
                {"_id": "faiss_data"},
                {"index": base64.b64encode(faiss.serialize_index(empty_index)).decode('utf-8'),
                 "id_mapping": base64.b64encode(pickle.dumps(empty_id_mapping)).decode('utf-8'),
                 "vector_count": 0,
                 "last_updated": time.time()},
                upsert=True
            )
            logger.info("✅ Index FAISS vide sauvegardé dans MongoDB.")
            return True

        vectors = []
        id_mapping = []
        processed_count = 0

        for i, cv in enumerate(cvs):
            cv_id_str = str(cv.get("_id", f"UNKNOWN_ID_{i}")) # ID de secours si _id est manquant
            logger.info(f"📊 Traitement du CV {i + 1}/{len(cvs)} (ID: {cv_id_str})")
            
            try:
                # Récupération sécurisée et concaténation des champs pertinents du CV
                # Chaque .get("", "") assure qu'on a une chaîne vide si le champ est manquant ou None
                # .strip() pour nettoyer les espaces inutiles
                
                nom = (cv.get("nom", "") or "").strip()
                biographie = (cv.get("biographie", "") or "").strip()
                
                # S'assurer que 'competences' est une liste avant de le joindre
                competences = ", ".join([str(c).strip() for c in cv.get("competences", []) if c and isinstance(c, (str, int, float))]).strip() \
                              if isinstance(cv.get("competences"), list) else ""

                # Traitement des expériences
                experiences_text_parts = []
                experiences = cv.get("experiences", [])
                if isinstance(experiences, list):
                    for exp in experiences[:3]: # Limiter aux 3 premières expériences pour concision
                        if isinstance(exp, dict):
                            exp_detail_parts = []
                            titre = (exp.get("titre", "") or "").strip()
                            entreprise = (exp.get("entreprise", "") or "").strip()
                            description = (exp.get("description", "") or "").strip()
                            
                            if titre: exp_detail_parts.append(titre)
                            if entreprise: exp_detail_parts.append(entreprise)
                            # Ajouter la description si elle est significative (plus de 10 caractères) et tronquée
                            if description and len(description) > 10:
                                exp_detail_parts.append(description[:500]) # Tronquer à 500 caractères
                            
                            if exp_detail_parts:
                                experiences_text_parts.append(" ".join(exp_detail_parts))
                experiences_combined = " ".join(experiences_text_parts).strip()


                # Traitement des formations
                formations_text_parts = []
                formations = cv.get("formations", [])
                if isinstance(formations, list):
                    for form in formations[:2]: # Limiter aux 2 premières formations
                        if isinstance(form, dict):
                            form_detail_parts = []
                            diplome = (form.get("diplome", "") or "").strip()
                            etablissement = (form.get("etablissement", "") or "").strip()
                            
                            if diplome: form_detail_parts.append(diplome)
                            if etablissement: form_detail_parts.append(etablissement)

                            if form_detail_parts:
                                formations_text_parts.append(" ".join(form_detail_parts))
                formations_combined = " ".join(formations_text_parts).strip()

                
                # Concaténation finale de tous les éléments textuels
                text_parts_for_embedding = []
                if nom: text_parts_for_embedding.append(f"Nom: {nom}")
                if biographie: text_parts_for_embedding.append(f"Biographie: {biographie}")
                if competences: text_parts_for_embedding.append(f"Compétences: {competences}")
                if experiences_combined: text_parts_for_embedding.append(f"Expériences: {experiences_combined}")
                if formations_combined: text_parts_for_embedding.append(f"Formations: {formations_combined}")
                
                combined_text = ". ".join(text_parts_for_embedding).strip()
                
                # Nettoyage et troncation du texte combiné final
                combined_text = " ".join(combined_text.split()) # Supprime les espaces multiples
                combined_text = combined_text[:512] # Tronquer le texte pour le modèle SentenceTransformer

                logger.debug(f"DEBUG CV {cv_id_str} - Texte combiné (tronqué à 512 chars): '{combined_text}'")
                
                if not combined_text:
                    logger.warning(f"⚠️ CV {cv_id_str} a généré un texte combiné vide. Skipping ce CV.")
                    continue

                # Encodage du texte en vecteur
                vector = model.encode(combined_text)
                
                # S'assurer que le vecteur est valide
                if vector is None or not isinstance(vector, np.ndarray) or vector.size == 0:
                    logger.warning(f"⚠️ Échec de l'encodage pour le CV {cv_id_str}: le vecteur est vide ou invalide.")
                    continue
                
                # Normalisation du vecteur (essentiel pour IndexFlatIP)
                norm = np.linalg.norm(vector)
                if norm > 0:
                    vector = vector / norm
                else:
                    logger.warning(f"⚠️ Vecteur nul (norme 0) pour le CV {cv_id_str}. Skipping ce CV.")
                    continue
                
                vectors.append(vector.astype(np.float32)) # Assurer le type float32 pour FAISS
                id_mapping.append(cv_id_str)
                processed_count += 1
                logger.info(f"✅ CV {cv_id_str} vectorisé avec succès. Total vectorisé: {processed_count}")

            except Exception as e:
                logger.error(f"❌ Erreur critique lors du traitement du CV {cv_id_str}: {e}")
                import traceback
                logger.error(f"Stack trace pour le CV {cv_id_str}:\n{traceback.format_exc()}")
                continue # Continuer avec le prochain CV même en cas d'erreur
        
        if not vectors:
            logger.warning("⚠️ Aucun vecteur n'a pu être créé après le traitement de tous les CVs. L'index sera vide.")
            # Créer un index vide si aucun vecteur n'a été ajouté (tous ont échoué)
            index = faiss.IndexFlatL2(model.get_sentence_embedding_dimension())
        else:
            # Création de l'index FAISS
            vectors_array = np.array(vectors, dtype=np.float32)
            dimension = vectors_array.shape[1]
            logger.info(f"📐 Dimension des vecteurs créés: {dimension}")
            
            # Utilisation de IndexFlatIP pour la similarité par produit scalaire (pour vecteurs normalisés)
            index = faiss.IndexFlatIP(dimension)
            index.add(vectors_array)
        
        logger.info(f"✅ Index FAISS créé avec {index.ntotal} vecteurs (sur {len(cvs)} CVs initialement).")

        # Sérialisation et sauvegarde de l'index et du mapping dans MongoDB
        # Utilisation d'un fichier temporaire pour la sérialisation FAISS (plus robuste)
        temp_path = None
        try:
            with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
                temp_path = tmp_file.name
                faiss.write_index(index, temp_path) # Écriture de l'index dans le fichier temporaire
            
            with open(temp_path, 'rb') as f:
                index_bytes = f.read()
            index_b64 = base64.b64encode(index_bytes).decode('utf-8')
            
        finally:
            if temp_path and os.path.exists(temp_path):
                os.unlink(temp_path) # Nettoyage du fichier temporaire
        
        mapping_bytes = pickle.dumps(id_mapping)
        mapping_b64 = base64.b64encode(mapping_bytes).decode('utf-8')

        # Stockage dans MongoDB - Utilisation de replace_one avec upsert=True pour écraser/créer
        index_collection.replace_one(
            {"_id": "faiss_data"},
            {
                "index": index_b64,
                "id_mapping": mapping_b64,
                "vector_count": len(vectors),
                "dimension": index.d,
                "model_name": "paraphrase-MiniLM-L3-v2", # Enregistrer le nom du modèle utilisé
                "index_type": "IndexFlatIP", # Enregistrer le type d'index FAISS
                "last_updated": time.time() # Horodatage de la dernière mise à jour
            },
            upsert=True # Si le document n'existe pas, il sera créé
        )
        
        logger.info(f"💾 Index FAISS stocké en base avec {len(vectors)} profils, dimension {index.d}.")
        logger.info(f"Total temps de mise à jour index FAISS: {time.time() - start_time_total:.2f} secondes.")
        return True
        
    except Exception as e:
        logger.error(f"❌ Erreur globale lors de la mise à jour de l'index FAISS: {e}")
        import traceback
        logger.error(f"Stack trace de l'erreur globale:\n{traceback.format_exc()}")
        return False

# ----------------- FONCTION LOAD_FAISS_FROM_MONGODB() (GARDÉE TELLE QUELLE AVEC MES LOGS DE TEMPS) -----------------
def load_faiss_from_mongodb():
    """Charge l'index FAISS depuis MongoDB."""
    try:
        from config import get_mongo_client
        start_time_total = time.time()
        client = get_mongo_client()
        
        if not client:
            logger.error("❌ Impossible de se connecter à MongoDB pour charger l'index FAISS.")
            return None, []
            
        db = client[DB_NAME]
        index_collection = db["faiss_index"]
        
        start_time_find = time.time()
        faiss_doc = index_collection.find_one({"_id": "faiss_data"})
        logger.info(f"⏳ Temps de find_one pour l'index FAISS: {time.time() - start_time_find:.2f} s")
        
        if not faiss_doc:
            logger.warning("⚠️ Aucun index FAISS trouvé en base. Retourne un index vide.")
            return None, []
        
        logger.info(f"📖 Chargement de l'index FAISS ({faiss_doc.get('vector_count', 0)} vecteurs).")
        
        index_b64 = faiss_doc.get("index")
        if not index_b64:
            logger.error("❌ Données de l'index FAISS manquantes dans le document MongoDB.")
            return None, []
        
        start_time_decode_b64 = time.time()
        index_bytes = base64.b64decode(index_b64)
        logger.info(f"⏳ Temps de décodage Base64 de l'index: {time.time() - start_time_decode_b64:.2f} s")
        
        temp_path = None
        try:
            start_time_write_tmp = time.time()
            with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
                temp_path = tmp_file.name
                tmp_file.write(index_bytes)
            logger.info(f"⏳ Temps d'écriture du fichier temporaire pour l'index: {time.time() - start_time_write_tmp:.2f} s")
            
            start_time_read_index = time.time()
            index = faiss.read_index(temp_path)
            logger.info(f"⏳ Temps de lecture de l'index FAISS depuis le fichier: {time.time() - start_time_read_index:.2f} s")
            
        finally:
            if temp_path and os.path.exists(temp_path):
                os.unlink(temp_path)
        
        mapping_b64 = faiss_doc.get("id_mapping")
        if not mapping_b64:
            logger.error("❌ Mapping des IDs FAISS manquant dans le document MongoDB.")
            return None, []
            
        start_time_decode_mapping = time.time()
        mapping_bytes = base64.b64decode(mapping_b64)
        id_mapping = pickle.loads(mapping_bytes)
        logger.info(f"⏳ Temps de décodage du mapping d'IDs: {time.time() - start_time_decode_mapping:.2f} s")
        
        logger.info(f"✅ Index FAISS chargé: {len(id_mapping)} profils, dimension {index.d}.")
        logger.info(f"Total temps de chargement FAISS: {time.time() - start_time_total:.2f} secondes.")
        return index, id_mapping
        
    except Exception as e:
        logger.error(f"❌ Erreur lors du chargement de l'index FAISS: {e}")
        import traceback
        logger.error(f"Stack trace de l'erreur de chargement:\n{traceback.format_exc()}")
        return None, []

def search_faiss_index(query_text: str, top_k: int = 5):
    """
    Recherche les CVs les plus similaires à une requête donnée dans l'index FAISS.
    Retourne une liste de tuples (ID du CV, score de similarité).
    """
    if model is None:
        logger.error("❌ Impossible de rechercher dans l'index FAISS, le modèle SentenceTransformer n'a pas été chargé.")
        return []

    # S'assurer que l'index et le mapping sont disponibles
    global faiss_index, faiss_id_mapping
    if faiss_index is None or faiss_id_mapping is None:
        logger.warning("⚠️ Index FAISS non chargé en mémoire. Tentative de rechargement...")
        faiss_index, faiss_id_mapping = load_faiss_from_mongodb()
        if faiss_index is None:
            logger.error("❌ Échec du rechargement de l'index FAISS. Impossible de rechercher.")
            return []

    try:
        start_time = time.time()
        # Vectorisation de la requête
        query_vector = model.encode([query_text])[0]
        # Normalisation de la requête si l'index utilise IndexFlatIP
        norm = np.linalg.norm(query_vector)
        if norm > 0:
            query_vector = query_vector / norm
        else:
            logger.warning("⚠️ Vecteur de requête nul. Impossible de rechercher.")
            return []

        # Recherche dans l'index FAISS
        # D, I = distances, indices
        D, I = faiss_index.search(np.array([query_vector]).astype(np.float32), top_k)
        
        results = []
        # Parcourir les résultats et construire la liste (ID, score)
        for i, distance in zip(I[0], D[0]):
            if i != -1: # -1 indique un résultat non trouvé (padding)
                cv_id = faiss_id_mapping[i]
                results.append((cv_id, float(distance))) # Convertir np.float32 en float Python

        logger.info(f"✅ Recherche FAISS terminée en {time.time() - start_time:.2f} s. {len(results)} résultats trouvés.")
        return results

    except Exception as e:
        logger.error(f"❌ Erreur lors de la recherche dans l'index FAISS: {e}")
        import traceback
        logger.error(f"Stack trace de l'erreur de recherche FAISS:\n{traceback.format_exc()}")
        return []

# Variables globales pour stocker l'index FAISS et le mapping une fois chargés
faiss_index = None
faiss_id_mapping = None

# Initialisation de l'index FAISS au chargement du module
# Ceci sera appelé une seule fois par processus Gunicorn worker
def init_faiss_service():
    global faiss_index, faiss_id_mapping
    logger.info("Initializing FAISS service...")
    faiss_index, faiss_id_mapping = load_faiss_from_mongodb()
    if faiss_index:
        logger.info("FAISS index loaded successfully during startup.")
    else:
        logger.error("Failed to load FAISS index during startup.")

# Appel de l'initialisation au chargement du module
init_faiss_service()
