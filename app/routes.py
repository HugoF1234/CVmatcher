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
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

index = None
id_mapping = []
client = None
collection = None
gemini_model = None

def run_with_timeout(func, timeout_seconds):
    """Exécute une fonction avec un timeout basé sur ThreadPoolExecutor."""
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(func)
        try:
            return future.result(timeout_seconds)
        except FuturesTimeoutError:
            raise TimeoutError("Opération timeout")
        
def reconfigure_gemini():
    """Reconfigure Gemini avec la clé API actuelle"""
    global gemini_model
    try:
        from config import GEMINI_API_KEY
        import google.generativeai as genai
        
        genai.configure(api_key=GEMINI_API_KEY)
        gemini_model = genai.GenerativeModel("gemini-2.0-flash")
        logger.info("✅ Gemini reconfiguré avec la nouvelle clé API")
        return True
    except Exception as e:
        logger.error(f"❌ Erreur reconfiguration Gemini: {e}")
        return False

def init_services():
    """Initialise les services (MongoDB, FAISS, Gemini)"""
    global index, id_mapping, client, collection, gemini_model
    
    try:
        from config import get_mongo_client
        client = get_mongo_client()
        if client is not None:
            collection = client[DB_NAME][COLLECTION_NAME]
        else:
            collection = None
            logger.error("❌ Impossible de se connecter à MongoDB")
    except Exception as e:
        logger.error(f"❌ Erreur MongoDB: {e}")
        client = None
        collection = None
    
    try:
        from app.utils.vectorize import load_faiss_from_mongodb
        index, id_mapping = load_faiss_from_mongodb()
        if index is not None:
            logger.info("✅ Index FAISS chargé depuis MongoDB")
        else:
            logger.warning("⚠️ Aucun index FAISS trouvé en base")
    except Exception as e:
        logger.error(f"⚠️ Erreur chargement FAISS: {e}")
    
    try:
        if GEMINI_API_KEY:
            genai.configure(api_key=GEMINI_API_KEY)
            gemini_model = genai.GenerativeModel("gemini-2.0-flash")
            logger.info("✅ Gemini 2.0 Flash configuré")
        else:
            logger.error("❌ GEMINI_API_KEY non défini")
    except Exception as e:
        logger.error(f"❌ Erreur Gemini: {e}")

init_services()

@app.route("/test-drive")
def test_drive():
    """Test de la connexion Google Drive"""
    try:
        from app.utils.drive_utils import test_drive_connection, connect_to_drive, list_pdfs
        from config import GOOGLE_DRIVE_FOLDER_ID
        
        result = {"status": "testing"}
        
        if test_drive_connection():
            result["connection"] = "success"
            
            try:
                service = connect_to_drive()
                folder_id = GOOGLE_DRIVE_FOLDER_ID
                pdfs = list_pdfs(service, folder_id)
                
                result["pdfs_found"] = len(pdfs)
                result["pdf_names"] = [pdf['name'] for pdf in pdfs[:5]]
                
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
        
        if collection is not None:
            cv_count = collection.count_documents({})
            collections = collection.database.list_collection_names()
            debug_data["mongodb"] = {
                "connected": True,
                "cv_count": cv_count,
                "collections": collections
            }
            
            if cv_count > 0:
                sample_cv = collection.find_one({})
                debug_data["sample_cv"] = {
                    "nom": sample_cv.get("nom", "N/A"),
                    "competences_count": len(sample_cv.get("competences", [])),
                    "has_biographie": bool(sample_cv.get("biographie"))
                }
        else:
            debug_data["mongodb"] = {"connected": False}
        
        if index is not None:
            debug_data["faiss"] = {
                "available": True,
                "entries": len(id_mapping),
                "dimension": index.d if hasattr(index, 'd') else 'unknown'
            }
        else:
            debug_data["faiss"] = {"available": False}
            
            if collection is not None:
                faiss_collection = collection.database["faiss_index"]
                faiss_doc = faiss_collection.find_one({"_id": "faiss_data"})
                debug_data["faiss"]["stored_in_db"] = faiss_doc is not None
                if faiss_doc:
                    debug_data["faiss"]["stored_count"] = faiss_doc.get("vector_count", 0)
        
        creds_exist = os.path.exists('credentials/credentials.json')
        token_exist = os.path.exists('credentials/token.json')
        debug_data["google_drive"] = {
            "credentials_file": creds_exist,
            "token_file": token_exist
        }
        
        return jsonify(debug_data)
        
    except Exception as e:
        logger.error(f"❌ Erreur debug: {e}")
        return jsonify({"error": str(e)}), 500
        

@app.route("/", methods=["GET", "POST"])
def home():
    global index, id_mapping, collection, gemini_model
    
    results = []
    prompt = ""
    if "likes" not in session:
        session["likes"] = []

    if request.method == "POST":
        if collection is None:
            error_msg = "Base de données non connectée."
            return render_template("index.html", results=[], error=error_msg)

        try:
            query = request.form.get("prompt", "").strip()
            if not query:
                return render_template("index.html", results=[], error="Veuillez saisir une requête.")

            logger.info(f"🔍 Recherche: '{query}'")
            
            search_results = []
            try:
                from app.utils.vectorize import search_similar_cvs
                search_results = run_with_timeout(lambda: search_similar_cvs(query, top_k=TOP_K), 15)
                logger.info(f"✅ Recherche vectorielle terminée: {len(search_results)} résultats")
            except TimeoutError:
                logger.warning("⏰ Timeout recherche vectorielle")
                return render_template("index.html", results=[], 
                                     error="Recherche trop longue, veuillez réessayer avec des termes plus simples.")
            except Exception as e:
                logger.error(f"❌ Erreur recherche vectorielle: {e}")
                return render_template("index.html", results=[], 
                                     error="Erreur de recherche. Veuillez réessayer.")

            if not search_results:
                return render_template("index.html", results=[], 
                                     error="Aucun profil trouvé correspondant à votre recherche.")

            cv_list = []
            for result in search_results:
                try:
                    cv = collection.find_one({"_id": ObjectId(result["cv_id"])})
                    if cv:
                        cv["_faiss_score"] = result["score"]
                        cv["_rank"] = result["rank"]
                        cv_list.append(cv)
                except Exception as e:
                    logger.error(f"❌ Erreur récupération CV {result['cv_id']}: {e}")
                    continue

            if not cv_list:
                return render_template("index.html", results=[], 
                                     error="Erreur lors de la récupération des profils.")

            if gemini_model is not None:
                try:
                    logger.info(f"🤖 Reranking Gemini pour {len(cv_list)} CVs")
                    
                    rerank_prompt = (
                        "Tu es un assistant RH expert.\n"
                        f"Requête utilisateur : \"{query}\"\n\n"
                        "Analyse et note chaque profil sur 100 selon sa pertinence pour cette requête.\n"
                        "Retourne les 3 meilleurs profils au format JSON :\n"
                        "[{\"nom\": \"Nom Prénom\", \"score\": 95, \"raison\": \"Explication en 1-2 phrases\"}, ...]\n\n"
                        "Profils à évaluer :\n"
                    )
                    
                    for i, cv in enumerate(cv_list, 1):
                        nom = cv.get('nom', f'Profil {i}')
                        competences = cv.get('competences', [])
                        biographie = cv.get('biographie', '')
                        secteur = cv.get('secteur', '')
                        
                        rerank_prompt += f"\n--- Profil {i} : {nom} ---\n"
                        rerank_prompt += f"Secteur: {secteur}\n"
                        rerank_prompt += f"Compétences: {', '.join(competences[:8])}\n"
                        rerank_prompt += f"Bio: {biographie[:300]}...\n"
                        
                        experiences = cv.get("experiences", [])
                        if experiences:
                            exp = experiences[0]
                            titre = exp.get('titre', '')
                            entreprise = exp.get('entreprise', '')
                            rerank_prompt += f"Expérience: {titre} chez {entreprise}\n"

                    text = run_with_timeout(lambda: gemini_model.generate_content(rerank_prompt).text.strip(), 20)

                    if text.startswith("```json"):
                        text = text[7:-3].strip()
                    elif text.startswith("```"):
                        text = text[3:-3].strip()

                    reranked = json.loads(text)
                    logger.info(f"✅ Gemini reranking réussi: {len(reranked)} profils")

                    for r in reranked:
                        matching_cv = None
                        nom_recherche = r.get('nom', '').lower().strip()

                        for cv in cv_list:
                            nom_cv = cv.get('nom', '').lower().strip()
                            if nom_recherche in nom_cv or nom_cv in nom_recherche:
                                matching_cv = cv
                                break

                        if matching_cv:
                            matching_cv["score"] = r.get("score", 0)
                            matching_cv["raison"] = r.get("raison", "Correspondance basée sur l'IA")
                            matching_cv["liked"] = str(matching_cv["_id"]) in session["likes"]
                            results.append(matching_cv)
                                
                except TimeoutError:
                    logger.warning("⏰ Timeout Gemini reranking, utilisation scores FAISS")
                    for cv in cv_list[:3]:
                        faiss_score = cv.get("_faiss_score", 0.5)
                        cv["score"] = int(faiss_score * 100) if faiss_score <= 1 else int(faiss_score)
                        cv["raison"] = "Correspondance basée sur l'analyse vectorielle"
                        cv["liked"] = str(cv["_id"]) in session["likes"]
                        results.append(cv)
                        
                except Exception as e:
                    logger.error(f"⚠️ Erreur Gemini reranking: {e}")
                    for cv in cv_list[:3]:
                        faiss_score = cv.get("_faiss_score", 0.5)
                        cv["score"] = int(faiss_score * 100) if faiss_score <= 1 else int(faiss_score)
                        cv["raison"] = "Correspondance basée sur l'analyse vectorielle"
                        cv["liked"] = str(cv["_id"]) in session["likes"]
                        results.append(cv)
            else:
                logger.info("📊 Utilisation des scores FAISS (pas de Gemini)")
                for cv in cv_list[:3]:
                    faiss_score = cv.get("_faiss_score", 0.5)
                    cv["score"] = int(faiss_score * 100) if faiss_score <= 1 else int(faiss_score)
                    cv["raison"] = "Correspondance basée sur l'analyse vectorielle"
                    cv["liked"] = str(cv["_id"]) in session["likes"]
                    results.append(cv)

            logger.info(f"🎯 Recherche terminée: {len(results)} résultats pour '{query}'")
            session["prompt"] = query
            session["last_results"] = [str(cv["_id"]) for cv in cv_list]
            prompt = query

        except Exception as e:
            logger.error(f"❌ Erreur dans la recherche: {e}")
            return render_template("index.html", results=[], 
                                 error="Erreur lors de la recherche. Veuillez réessayer.")

    else:
        if "prompt" in session and "last_results" in session:
            prompt = session["prompt"]
            results = []
            for cid in session["last_results"]:
                cv = collection.find_one({"_id": ObjectId(cid)})
                if cv:
                    results.append(cv)

    return render_template("index.html", results=results, prompt=prompt)

def run_update_background():
    try:
        from config import get_mongo_client
        from app.watcher import run_watch
        from app.utils.vectorize import load_faiss_from_mongodb, sync_faiss_with_db

        client = get_mongo_client()
        logger.info("🔄 Début de la mise à jour des CVs en arrière-plan")
        success = run_watch()

        if success:
            global index, id_mapping
            sync_faiss_with_db(client=client)
            index, id_mapping = load_faiss_from_mongodb(client=client)
            logger.info("🔄 Index FAISS rechargé après mise à jour")

        logger.info("✅ Mise à jour terminée")
    except Exception as e:
        logger.error(f"❌ Erreur mise à jour CVs en arrière-plan: {e}")


@app.route("/clean-index", methods=["POST"])
def clean_index():
    """Nettoie et recrée l'index FAISS"""
    try:
        from app.utils.vectorize import clean_faiss_index, update_faiss_index
        
        logger.info("🧹 Début nettoyage index FAISS")
        
        clean_success = clean_faiss_index()
        if not clean_success:
            return jsonify({
                "status": "error",
                "message": "Erreur lors du nettoyage"
            }), 500
        
        logger.info("🔄 Recréation de l'index FAISS")
        create_success = update_faiss_index()
        
        if create_success:
            global index, id_mapping
            from app.utils.vectorize import load_faiss_from_mongodb
            index, id_mapping = load_faiss_from_mongodb()
            
            return jsonify({
                "status": "success",
                "message": f"Index FAISS recréé avec succès ({len(id_mapping) if id_mapping else 0} profils)"
            })
        else:
            return jsonify({
                "status": "error", 
                "message": "Erreur lors de la recréation de l'index"
            }), 500
            
    except Exception as e:
        logger.error(f"❌ Erreur clean-index: {e}")
        return jsonify({
            "status": "error",
            "message": f"Erreur: {str(e)}"
        }), 500
        
@app.route("/diagnostic")
def run_diagnostic():
    """Route de diagnostic complète du système d'embedding"""
    try:
        import json
        import os
        from sentence_transformers import SentenceTransformer
        import numpy as np
        from datetime import datetime
        
        diagnostic_results = {
            "timestamp": str(datetime.now()),
            "tests": {}
        }
        
        try:
            if collection is not None:
                cv_count = collection.count_documents({})
                sample_cv = collection.find_one({}) if cv_count > 0 else None
                
                diagnostic_results["tests"]["mongodb"] = {
                    "status": "success",
                    "cv_count": cv_count,
                    "sample_cv": {
                        "nom": sample_cv.get("nom", "N/A") if sample_cv else "N/A",
                        "has_competences": bool(sample_cv.get("competences")) if sample_cv else False,
                        "has_biographie": bool(sample_cv.get("biographie")) if sample_cv else False,
                        "experiences_count": len(sample_cv.get("experiences", [])) if sample_cv else 0
                    } if sample_cv else None
                }
            else:
                diagnostic_results["tests"]["mongodb"] = {
                    "status": "failed",
                    "error": "Collection non disponible"
                }
        except Exception as e:
            diagnostic_results["tests"]["mongodb"] = {
                "status": "error",
                "error": str(e)
            }
        
        try:
            import os
            from sentence_transformers import SentenceTransformer
            model = SentenceTransformer("all-MiniLM-L6-v2", use_auth_token=os.environ.get("HF_TOKEN"))
            test_text = "développeur Python avec 5 ans d'expérience"
            vector = model.encode(test_text)
            
            diagnostic_results["tests"]["sentence_transformer"] = {
                "status": "success",
                "model_name": "all-MiniLM-L6-v2",
                "vector_dimension": len(vector),
                "test_text": test_text
            }
        except Exception as e:
            diagnostic_results["tests"]["sentence_transformer"] = {
                "status": "error",
                "error": str(e)
            }
        
        try:
            if index is not None and id_mapping:
                diagnostic_results["tests"]["faiss"] = {
                    "status": "success",
                    "entries": len(id_mapping),
                    "dimension": index.d if hasattr(index, 'd') else 'unknown',
                    "index_type": str(type(index))
                }
            else:
                from app.utils.vectorize import load_faiss_from_mongodb
                test_index, test_mapping = load_faiss_from_mongodb()
                
                if test_index is not None:
                    diagnostic_results["tests"]["faiss"] = {
                        "status": "loaded_from_db",
                        "entries": len(test_mapping),
                        "dimension": test_index.d if hasattr(test_index, 'd') else 'unknown'
                    }
                else:
                    diagnostic_results["tests"]["faiss"] = {
                        "status": "not_available",
                        "message": "Index FAISS non trouvé en mémoire ni en base"
                    }
        except Exception as e:
            diagnostic_results["tests"]["faiss"] = {
                "status": "error",
                "error": str(e)
            }
        
        try:
            from app.utils.vectorize import search_similar_cvs
            test_results = search_similar_cvs("développeur Python", top_k=3)
            
            diagnostic_results["tests"]["vector_search"] = {
                "status": "success" if test_results else "no_results",
                "results_count": len(test_results),
                "sample_scores": [r["score"] for r in test_results[:3]] if test_results else []
            }
        except Exception as e:
            diagnostic_results["tests"]["vector_search"] = {
                "status": "error",
                "error": str(e)
            }
        
        try:
            if gemini_model is not None:
                diagnostic_results["tests"]["gemini"] = {
                    "status": "available",
                    "model_configured": True
                }
            else:
                diagnostic_results["tests"]["gemini"] = {
                    "status": "not_configured",
                    "api_key_present": bool(GEMINI_API_KEY)
                }
        except Exception as e:
            diagnostic_results["tests"]["gemini"] = {
                "status": "error",
                "error": str(e)
            }
        
        passed_tests = sum(1 for test in diagnostic_results["tests"].values() 
                          if test.get("status") in ["success", "available", "loaded_from_db"])
        total_tests = len(diagnostic_results["tests"])
        
        diagnostic_results["summary"] = {
            "passed_tests": passed_tests,
            "total_tests": total_tests,
            "success_rate": f"{(passed_tests/total_tests)*100:.1f}%",
            "overall_status": "healthy" if passed_tests >= 4 else "issues_detected"
        }
        
        return jsonify(diagnostic_results)
        
    except Exception as e:
        logger.error(f"❌ Erreur diagnostic: {e}")
        return jsonify({
            "error": str(e),
            "status": "critical_error"
        }), 500

@app.route("/diagnostic-ui")
def diagnostic_ui():
    """Interface web pour le diagnostic"""
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Diagnostic Système</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 40px; }
            .test { margin: 20px 0; padding: 15px; border-radius: 8px; }
            .success { background-color:
            .error { background-color:
            .warning { background-color:
            pre { background:
            button { padding: 10px 20px; background:
        </style>
    </head>
    <body>
        <h1>🔍 Diagnostic du Système CV-Matcher</h1>
        <button onclick="runDiagnostic()">🚀 Lancer le diagnostic</button>
        <div id="results"></div>
        
        <script>
        async function runDiagnostic() {
            document.getElementById('results').innerHTML = '<p>⏳ Diagnostic en cours...</p>';
            
            try {
                const response = await fetch('/diagnostic');
                const data = await response.json();
                
                let html = '<h2>📊 Résultats</h2>';
                
                // Résumé
                const summary = data.summary;
                const statusClass = summary.overall_status === 'healthy' ? 'success' : 'warning';
                html += `<div class="test ${statusClass}">
                    <h3>📈 Résumé Global</h3>
                    <p><strong>Tests réussis:</strong> ${summary.passed_tests}/${summary.total_tests} (${summary.success_rate})</p>
                    <p><strong>Statut:</strong> ${summary.overall_status}</p>
                </div>`;
                
                // Détails des tests
                for (const [testName, result] of Object.entries(data.tests)) {
                    const statusClass = result.status === 'success' || result.status === 'available' ? 'success' : 
                                       result.status === 'error' ? 'error' : 'warning';
                    
                    html += `<div class="test ${statusClass}">
                        <h3>${testName.toUpperCase()}</h3>
                        <p><strong>Statut:</strong> ${result.status}</p>
                        <pre>${JSON.stringify(result, null, 2)}</pre>
                    </div>`;
                }
                
                document.getElementById('results').innerHTML = html;
                
            } catch (error) {
                document.getElementById('results').innerHTML = 
                    `<div class="test error"><h3>❌ Erreur</h3><p>${error.message}</p></div>`;
            }
        }
        </script>
    </body>
    </html>
    """

@app.route("/update-cvs", methods=["POST"])
def update_cvs():
    """Lance la mise à jour des CVs de manière asynchrone pour éviter les timeouts"""
    try:
        if collection is None:
            return jsonify({
                "status": "error", 
                "message": "Base de données non connectée"
            }), 503
        
        thread = threading.Thread(target=run_update_background)
        thread.daemon = True
        thread.start()
        
        logger.info("🚀 Mise à jour des CVs lancée en arrière-plan")
        
        return jsonify({
            "status": "started",
            "message": "Mise à jour des CVs lancée en arrière-plan. Rechargez la page dans quelques minutes."
        }), 200
        
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
        
        cv_count = collection.count_documents({})
        
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
        cv_count = 0
        if collection is not None:
            cv_count = collection.count_documents({})
        
        status = {
            "mongodb": collection is not None,
            "cv_count": cv_count,
            "faiss": index is not None,
            "faiss_entries": len(id_mapping) if index is not None else 0,
            "gemini": gemini_model is not None
        }
        return jsonify(status), 200
    except Exception as e:
        logger.error(f"❌ Erreur health check: {e}")
        return jsonify({"error": str(e)}), 500


from flask import Blueprint

bp = Blueprint('main', __name__)

@bp.route("/")
def index():
    return "CV Matcher is running!"

from flask import send_file, abort
import os

@app.route("/download/<nomdupdf>")
def download_pdf(nomdupdf):
    local_path = f"/tmp/{nomdupdf}"
    if not os.path.exists(local_path):
        from app.utils.drive_utils import connect_to_drive, download_file
        from config import GOOGLE_DRIVE_FOLDER_ID
        service = connect_to_drive()
        from app.utils.drive_utils import list_pdfs
        folder_id = GOOGLE_DRIVE_FOLDER_ID
        pdfs = list_pdfs(service, folder_id)
        file_id = next((pdf['id'] for pdf in pdfs if pdf['name'] == nomdupdf), None)
        if not file_id:
            abort(404, "PDF non trouvé sur Google Drive")
        download_file(service, file_id, local_path)
    if os.path.exists(local_path):
        return send_file(local_path, as_attachment=True)
    else:
        abort(404, "PDF non trouvé")

@app.route("/reset-database", methods=["POST"])
def reset_database():
    """Vide la base de données et la remplit avec les CVs du nouveau dossier"""
    try:
        if collection is None:
            return jsonify({
                "status": "error", 
                "message": "Base de données non connectée"
            }), 503
        
        logger.info("🗑️ Début du reset de la base de données")
        
        delete_result = collection.delete_many({})
        logger.info(f"🗑️ {delete_result.deleted_count} CVs supprimés de la collection principale")
        
        db = collection.database
        seen_collection = db["seen_cvs"]
        seen_delete_result = seen_collection.delete_many({})
        logger.info(f"🗑️ {seen_delete_result.deleted_count} entrées supprimées de seen_cvs")
        
        faiss_collection = db["faiss_index"]
        faiss_delete_result = faiss_collection.delete_many({})
        logger.info(f"🗑️ {faiss_delete_result.deleted_count} entrées supprimées de faiss_index")
        
        thread = threading.Thread(target=run_update_background)
        thread.daemon = True
        thread.start()
        
        logger.info("🚀 Mise à jour lancée avec le nouveau dossier Google Drive")
        
        return jsonify({
            "status": "success",
            "message": f"Base de données vidée ({delete_result.deleted_count} CVs supprimés). Mise à jour en cours avec le nouveau dossier."
        }), 200
        
    except Exception as e:
        logger.error(f"❌ Erreur reset database: {e}")
        return jsonify({
            "status": "error",
            "message": f"Erreur: {str(e)}"
        }), 500