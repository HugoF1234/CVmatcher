from flask import Flask, render_template, request, session, redirect, url_for
import faiss
import numpy as np
import os
import pickle
from pymongo import MongoClient
from bson import ObjectId
import google.generativeai as genai
import json

app = Flask(__name__)
app.secret_key = "0180529a5b9c1ec478296df826a91c31"

# === CONFIGURATION ===
MONGO_URI = os.environ.get("MONGO_URI")
DB_NAME = "CVExtraction"
COLLECTION_NAME = "CVExtractionCollection"
FAISS_INDEX_FILE = "faiss_index/cv_index.faiss"
ID_MAPPING_FILE = "faiss_index/id_mapping.pkl"
GEMINI_API_KEY = "AIzaSyCsfLrbLkNiJKSKdQsIps3KK47sxLNVCMQ"
TOP_K = 5

# === INIT FAISS (si possible)
index = None
id_mapping = []
if os.path.exists(FAISS_INDEX_FILE) and os.path.exists(ID_MAPPING_FILE):
    try:
        index = faiss.read_index(FAISS_INDEX_FILE)
        with open(ID_MAPPING_FILE, "rb") as f:
            id_mapping = pickle.load(f)
        print("✅ Index FAISS chargé.")
    except Exception as e:
        print("⚠️ Erreur chargement FAISS :", e)
else:
    print("⚠️ Aucun index FAISS trouvé. Cliquez sur 'Mettre à jour les CVs' pour le générer.")

# === AUTRES INIT
client = MongoClient(MONGO_URI)
collection = client[DB_NAME][COLLECTION_NAME]
genai.configure(api_key=GEMINI_API_KEY)
gemini_model = genai.GenerativeModel("gemini-1.5-pro")

@app.route("/", methods=["GET", "POST"])
def home():
    results = []
    if "likes" not in session:
        session["likes"] = []

    if request.method == "POST":
        if index is None:
            return render_template("index.html", results=[], error="Aucun index FAISS disponible. Cliquez sur 'Mettre à jour les CVs'.")

        # Lazy load SentenceTransformer ici
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer("all-MiniLM-L6-v2")

        query = request.form.get("prompt")
        query_vec = model.encode(query)
        query_vec = np.array([query_vec]).astype("float32")
        faiss.normalize_L2(query_vec)
        D, I = index.search(query_vec, TOP_K)

        cv_list = []
        for idx in I[0]:
            if idx >= len(id_mapping): continue
            cv = collection.find_one({"_id": ObjectId(id_mapping[idx])})
            if cv: cv_list.append(cv)

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

        try:
            response = gemini_model.generate_content(rerank_prompt)
            text = response.text.strip()
            if text.startswith("```json"):
                text = text[7:-3].strip()
            elif text.startswith("```"):
                text = text[3:-3].strip()
            reranked = json.loads(text)
        except Exception as e:
            print("⚠️ Erreur Gemini :", e)
            reranked = []

        for r in reranked:
            cv = collection.find_one({"nom": {"$regex": f"^{r['nom']}$", "$options": "i"}})
            if cv:
                cv["score"] = r.get("score", 0)
                cv["raison"] = r.get("raison", "Non précisée")
                cv["liked"] = str(cv["_id"]) in session["likes"]
                results.append(cv)

    return render_template("index.html", results=results)

@app.route("/update-cvs", methods=["POST"])
def update_cvs():
    from app.watcher import run_watch
    run_watch()
    return redirect("/")

@app.route("/toggle_like/<cv_id>")
def toggle_like(cv_id):
    if "likes" not in session:
        session["likes"] = []
    if cv_id in session["likes"]:
        session["likes"].remove(cv_id)
    else:
        session["likes"].append(cv_id)
    session.modified = True
    return redirect(request.referrer or "/")

@app.route("/likes")
def show_likes():
    liked_ids = session.get("likes", [])
    liked_cvs = []
    for cid in liked_ids:
        cv = collection.find_one({"_id": ObjectId(cid)})
        if cv:
            liked_cvs.append(cv)
    return render_template("likes.html", results=liked_cvs)

@app.route("/cv/<cv_id>")
def show_cv_detail(cv_id):
    try:
        cv = collection.find_one({"_id": ObjectId(cv_id)})
        if not cv:
            return "CV non trouvé", 404
        return render_template("cv_detail.html", cv=cv)
    except:
        return "ID invalide", 400

