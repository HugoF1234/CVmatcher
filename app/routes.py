from flask import render_template, request, session, redirect, url_for
import faiss
import numpy as np
import os
import pickle
from pymongo import MongoClient
from bson import ObjectId
import google.generativeai as genai
import json
import logging
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
        if MONGO_URI:
            client = MongoClient(MONGO_URI)
            collection = client[DB_NAME][COLLECTION_NAME]
            logger.info("✅ MongoDB connecté")
        else:
            logger.error("❌ MONGO_URI non défini")
    except Exception as e:
        logger.error(f"❌ Erreur MongoDB: {e}")
    
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

@app.route("/update-cvs", methods=["POST"])
def update_cvs():
    global index, id_mapping
    
    try:
        from app.watcher import run_watch
        success = run_watch()
        
        if success:
            # Recharger l'index FAISS depuis MongoDB
            from app.utils.vectorize import load_faiss_from_mongodb
            index, id_mapping = load_faiss_from_mongodb()
            logger.info("🔄 Index FAISS rechargé après mise à jour")
            
        return redirect(url_for('home'))
    except Exception as e:
        logger.error(f"❌ Erreur mise à jour CVs: {e}")
        return redirect(url_for('home'))

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
    status = {
        "mongodb": collection is not None,
        "faiss": index is not None,
        "gemini": gemini_model is not None
    }
    return status
