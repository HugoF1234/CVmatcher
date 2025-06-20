from flask import render_template, request, session, redirect, url_for, jsonify
import faiss
import numpy as np
import os
import pickle
from pymongo import MongoClient
from bson import ObjectId
import google.generativeai as genai
import json
import logging
import threading
from config import *
from app import app

# Configuration du logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# === INITIALISATION GLOBALE ===
index = None
id_mapping = []
client = None
collection = None
gemini_model = None

def init_services():
    """Initialise les services (MongoDB, FAISS, Gemini)"""
    global index, id_mapping, client, collection, gemini_model
    
    # Init MongoDB
    try:
        from config import get_mongo_client
        client = get_mongo_client()
        if client:
            collection = client[DB_NAME][COLLECTION_NAME]
        else:
            collection = None
            logger.error("❌ Impossible de se connecter à MongoDB")
    except Exception as e:
        logger.error(f"❌ Erreur MongoDB: {e}")
        client = None
        collection = None
    
    # Init FAISS depuis MongoDB
    try:
        from app.utils.vectorize import load_faiss_from_mongodb
        index, id_mapping = load_faiss_from_mongodb()
        if index is not None:
            logger.info("✅ Index FAISS chargé depuis MongoDB")
        else:
            logger.warning("⚠️ Aucun index FAISS trouvé en base")
    except Exception as e:
        logger.error(f"⚠️ Erreur chargement FAISS: {e}")
    
    # Init Gemini
    try:
        if GEMINI_API_KEY:
            genai.configure(api_key=GEMINI_API_KEY)
            gemini_model = genai.GenerativeModel("gemini-1.5-pro")
            logger.info("✅ Gemini configuré")
        else:
            logger.error("❌ GEMINI_API_KEY non défini")
    except Exception as e:
        logger.error(f"❌ Erreur Gemini: {e}")

# Initialiser au chargement du module
init_services()

# Ajoutez aussi cette route dans app/routes.py

@app.route("/test-drive")
def test_drive():
    """Test de la connexion Google Drive"""
    try:
        from app.utils.drive_utils import test_drive_connection, connect_to_drive, list_pdfs
        
        result = {"status": "testing"}
        
        # Test de connexion
        if test_drive_connection():
            result["connection"] = "success"
            
            # Test de listage des PDFs
            try:
                service = connect_to_drive()
                folder_id = "16CpxlBPbm8ZMRBH-B7tj5cmn4h3bXCbt"
                pdfs = list_pdfs(service, folder_id)
                
                result["pdfs_found"] = len(pdfs)
                result["pdf_names"] = [pdf['name'] for pdf in pdfs[:5]]  # Première 5
                
            except Exception as e:
                result["pdf_error"] = str(e)
                
        else:
            result["connection"] = "failed"
            
        return jsonify(result)
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500
        
@app.route("/debug")
def debug_info():
    """Route de debug pour diagnostiquer les problèmes"""
    try:
        debug_data = {}
        
        # Test MongoDB - CORRECTION: utiliser collection is not None
        if collection is not None:
            cv_count = collection.count_documents({})
            collections = collection.database.list_collection_names()
            debug_data["mongodb"] = {
                "connected": True,
                "cv_count": cv_count,
                "collections": collections
            }
            
            # Vérifier s'il y a des CVs
            if cv_count > 0:
                sample_cv = collection.find_one({})
                debug_data["sample_cv"] = {
                    "nom": sample_cv.get("nom", "N/A"),
                    "competences_count": len(sample_cv.get("competences", [])),
                    "has_biographie": bool(sample_cv.get("biographie"))
                }
        else:
            debug_data["mongodb"] = {"connected": False}
        
        # Test FAISS - CORRECTION: utiliser index is not None
        if index is not None:
            debug_data["faiss"] = {
                "available": True,
                "entries": len(id_mapping),
                "dimension": index.d if hasattr(index, 'd') else 'unknown'
            }
        else:
            debug_data["faiss"] = {"available": False}
            
            # Vérifier s'il y a un index FAISS stocké en base
            if collection is not None:
                faiss_collection = collection.database["faiss_index"]
                faiss_doc = faiss_collection.find_one({"_id": "faiss_data"})
                debug_data["faiss"]["stored_in_db"] = faiss_doc is not None
                if faiss_doc:
                    debug_data["faiss"]["stored_count"] = faiss_doc.get("vector_count", 0)
        
        # Test Google Drive credentials
        creds_exist = os.path.exists('credentials/credentials.json')
        token_exist = os.path.exists('credentials/token.json')
        debug_data["google_drive"] = {
            "credentials_file": creds_exist,
            "token_file": token_exist
        }
        
        return jsonify(debug_data)
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500
        
@app.route("/", methods=["GET", "POST"])
def home():
    global index, id_mapping, collection, gemini_model
    
    results = []
    if "likes" not in session:
        session["likes"] = []

    if request.method == "POST":
        if index is None or collection is None:
            error_msg = "Services non disponibles. "
            if index is None:
                error_msg += "Index FAISS manquant. "
            if collection is None:
                error_msg += "Base de données non connectée."
            return render_template("index.html", results=[], error=error_msg)

        try:
            # Lazy load SentenceTransformer ici
            from sentence_transformers import SentenceTransformer
            model = SentenceTransformer("all-MiniLM-L6-v2")

            query = request.form.get("prompt", "").strip()
            if not query:
                return render_template("index.html", results=[], error="Veuillez saisir une requête.")

            # Recherche vectorielle
            query_vec = model.encode(query)
            query_vec = np.array([query_vec]).astype("float32")
            faiss.normalize_L2(query_vec)
            D, I = index.search(query_vec, TOP_K)

            # Récupération des CVs
            cv_list = []
            for idx in I[0]:
                if idx >= len(id_mapping): 
                    continue
                cv = collection.find_one({"_id": ObjectId(id_mapping[idx])})
                if cv: 
                    cv_list.append(cv)

            if not cv_list:
                return render_template("index.html", results=[], error="Aucun CV trouvé dans la base.")

            # Reranking avec Gemini
            if gemini_model:
                try:
                    rerank_prompt = (
                        f"Tu es un assistant RH intelligent.\n"
                        f"Voici la requête initiale de l'utilisateur : \"{query}\"\n\n"
                        f"Tu dois d'abord **réinterpréter cette requête** de façon enrichie (synonymes, précisions, contexte).\n"
                        f"Ensuite, évalue pour chacun des CVs ci-dessous :\n"
                        f"- un score de pertinence entre 0 et 100\n"
                        f"- une explication du score (2 phrases max)\n"
                        f"Enfin, retourne uniquement les **3 meilleurs** profils.\n\n"
                        f"Réponds au format JSON, comme ceci :\n"
                        f"[{{\"nom\": \"Nom Prénom\", \"score\": 92, \"raison\": \"...\"}}, ...]\n\n"
                        f"Voici les CVs à évaluer :\n"
                    )
                    
                    for cv in cv_list:
                        rerank_prompt += f"\n---\nNom : {cv.get('nom', 'Inconnu')}\n"
                        rerank_prompt += f"Compétences : {', '.join(cv.get('competences', []))}\n"
                        rerank_prompt += f"Biographie : {cv.get('biographie', '')[:300]}\n"
                        rerank_prompt += f"Expériences :\n"
                        for exp in cv.get("experiences", [])[:2]:
                            rerank_prompt += f"- {exp.get('titre', '')} chez {exp.get('entreprise', '')} ({exp.get('dateDebut', '')} - {exp.get('dateFin', '')})\n"

                    response = gemini_model.generate_content(rerank_prompt)
                    text = response.text.strip()
                    
                    # Nettoyage du JSON
                    if text.startswith("```json"):
                        text = text[7:-3].strip()
                    elif text.startswith("```"):
                        text = text[3:-3].strip()
                    
                    reranked = json.loads(text)
                    
                    # Construction des résultats
                    for r in reranked:
                        cv = collection.find_one({"nom": {"$regex": f"^{r['nom']}$", "$options": "i"}})
                        if cv:
                            cv["score"] = r.get("score", 0)
                            cv["raison"] = r.get("raison", "Non précisée")
                            cv["liked"] = str(cv["_id"]) in session["likes"]
                            results.append(cv)
                            
                except Exception as e:
                    logger.error(f"⚠️ Erreur Gemini reranking: {e}")
                    # Fallback: retourner les résultats FAISS sans reranking
                    for cv in cv_list[:3]:
                        cv["score"] = 85  # Score par défaut
                        cv["raison"] = "Correspondance basée sur l'analyse vectorielle"
                        cv["liked"] = str(cv["_id"]) in session["likes"]
                        results.append(cv)
            else:
                # Pas de Gemini disponible
                for cv in cv_list[:3]:
                    cv["score"] = 85
                    cv["raison"] = "Correspondance basée sur l'analyse vectorielle"
                    cv["liked"] = str(cv["_id"]) in session["likes"]
                    results.append(cv)

        except Exception as e:
            logger.error(f"❌ Erreur dans la recherche: {e}")
            return render_template("index.html", results=[], error=f"Erreur lors de la recherche: {str(e)}")

    return render_template("index.html", results=results)

def run_update_background():
    """Lance la mise à jour en arrière-plan"""
    try:
        from app.watcher import run_watch
        logger.info("🔄 Début de la mise à jour des CVs en arrière-plan")
        success = run_watch()
        
        if success:
            global index, id_mapping
            # Recharger l'index FAISS depuis MongoDB
            from app.utils.vectorize import load_faiss_from_mongodb
            index, id_mapping = load_faiss_from_mongodb()
            logger.info("🔄 Index FAISS rechargé après mise à jour")
        
        logger.info("✅ Mise à jour terminée")
    except Exception as e:
        logger.error(f"❌ Erreur mise à jour CVs en arrière-plan: {e}")

@app.route("/update-cvs", methods=["POST"])
def update_cvs():
    """Lance la mise à jour des CVs de manière asynchrone pour éviter les timeouts"""
    try:
        # Vérifier que MongoDB est connecté
        if collection is None:
            return jsonify({
                "status": "error", 
                "message": "Base de données non connectée"
            }), 503
        
        # Lancer le processus en arrière-plan
        thread = threading.Thread(target=run_update_background)
        thread.daemon = True
        thread.start()
        
        logger.info("🚀 Mise à jour des CVs lancée en arrière-plan")
        
        # Retourner immédiatement une réponse
        return jsonify({
            "status": "started",
            "message": "Mise à jour des CVs lancée en arrière-plan. Rechargez la page dans quelques minutes."
        })
        
    except Exception as e:
        logger.error(f"❌ Erreur lancement mise à jour CVs: {e}")
        return jsonify({
            "status": "error",
            "message": f"Erreur: {str(e)}"
        }), 500

@app.route("/update-status")
def update_status():
    """Endpoint pour vérifier le statut de la mise à jour"""
    try:
        if collection is None:
            return jsonify({"mongodb": False, "faiss": False})
        
        # Compter les CVs dans la base
        cv_count = collection.count_documents({})
        
        # Vérifier FAISS
        faiss_available = index is not None and len(id_mapping) > 0
        
        return jsonify({
            "mongodb": True,
            "cv_count": cv_count,
            "faiss": faiss_available,
            "faiss_entries": len(id_mapping) if faiss_available else 0
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/toggle_like/<cv_id>")
def toggle_like(cv_id):
    if "likes" not in session:
        session["likes"] = []
    if cv_id in session["likes"]:
        session["likes"].remove(cv_id)
    else:
        session["likes"].append(cv_id)
    session.modified = True
    return redirect(request.referrer or url_for('home'))

@app.route("/likes")
def show_likes():
    global collection
    
    if collection is None:
        return render_template("likes.html", results=[])
    
    liked_ids = session.get("likes", [])
    liked_cvs = []
    for cid in liked_ids:
        try:
            cv = collection.find_one({"_id": ObjectId(cid)})
            if cv:
                liked_cvs.append(cv)
        except Exception as e:
            logger.error(f"Erreur récupération CV {cid}: {e}")
    return render_template("likes.html", results=liked_cvs)

@app.route("/cv/<cv_id>")
def show_cv_detail(cv_id):
    global collection
    
    if collection is None:
        return "Base de données non disponible", 503
        
    try:
        cv = collection.find_one({"_id": ObjectId(cv_id)})
        if not cv:
            return "CV non trouvé", 404
        return render_template("cv_detail.html", cv=cv)
    except Exception as e:
        logger.error(f"Erreur affichage CV {cv_id}: {e}")
        return "ID invalide", 400

@app.route("/health")
def health_check():
    """Endpoint de santé pour vérifier le statut des services"""
    try:
        cv_count = collection.count_documents({}) if collection else 0
        status = {
            "mongodb": collection is not None,
            "cv_count": cv_count,
            "faiss": index is not None,
            "faiss_entries": len(id_mapping) if index else 0,
            "gemini": gemini_model is not None
        }
        return jsonify(status)
    except Exception as e:
        return jsonify({"error": str(e)}), 500
