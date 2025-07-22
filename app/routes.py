from flask import render_template, request, session, redirect, url_for, jsonify
import json
import logging
import threading
import time
from datetime import datetime
import signal
from contextlib import contextmanager
import os # Pour vérifier l'existence des fichiers de crédentials

# Imports spécifiques pour FAISS, MongoDB, Gemini
import faiss
import numpy as np
from pymongo import MongoClient
from bson import ObjectId
import google.generativeai as genai

# Importation de vos configurations
from config import get_mongo_client, DB_NAME, COLLECTION_NAME, GEMINI_API_KEY, TOP_K

# Importation des fonctions de vectorisation et de gestion FAISS
# Assurez-vous que ces fonctions existent et sont correctement implémentées dans app/utils/vectorize.py
from app.utils.vectorize import search_faiss_index, update_faiss_index, load_faiss_from_mongodb, clean_faiss_index, get_model_info, test_embedding_model, get_faiss_index_status

# Configuration du logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# === INITIALISATION GLOBALE DES SERVICES ===
# Ces variables globales sont utilisées pour maintenir l'état des connexions
_mongo_client = None
_db_collection = None
_gemini_model = None

# Variable globale pour l'intervalle de surveillance du fichier (en secondes)
WATCH_INTERVAL = 600  # Exemple: toutes les 10 minutes (ou 3600 pour 1h)
last_update_time = 0 # Variable globale pour suivre la dernière mise à jour de l'index FAISS

# Context manager pour le timeout
@contextmanager
def timeout_context(seconds):
    """Context manager pour timeout"""
    def timeout_handler(signum, frame):
        raise TimeoutError("Opération timeout")

    old_handler = signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(seconds)

    try:
        yield
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)

def _initialize_services_on_startup():
    """Initialise les services lors du démarrage de l'application."""
    global _mongo_client, _db_collection, _gemini_model, last_update_time

    # Init MongoDB
    try:
        _mongo_client = get_mongo_client()
        if _mongo_client:
            _db_collection = _mongo_client[DB_NAME][COLLECTION_NAME]
            logger.info("✅ Connexion MongoDB établie.")
        else:
            logger.error("❌ Impossible de se connecter à MongoDB.")
    except Exception as e:
        logger.error(f"❌ Erreur lors de l'initialisation de MongoDB: {e}")

    # Init Gemini
    try:
        if GEMINI_API_KEY:
            genai.configure(api_key=GEMINI_API_KEY)
            _gemini_model = genai.GenerativeModel("gemini-pro") # Recommandé: gemini-pro pour la production
            logger.info("✅ Gemini Pro configuré.")
        else:
            logger.error("❌ GEMINI_API_KEY non défini, Gemini ne sera pas utilisé.")
    except Exception as e:
        logger.error(f"❌ Erreur lors de l'initialisation de Gemini: {e}")

    # Initialiser la dernière mise à jour pour déclencher une mise à jour dès le démarrage
    last_update_time = 0 # Force la première mise à jour de l'index FAISS au démarrage

# Appelez la fonction d'initialisation au chargement du module
_initialize_services_on_startup()


def rerank_with_gemini(query: str, candidates: list):
    """
    Utilise l'API Gemini pour reranker une liste de candidats (CVs) par rapport à une requête.
    Les candidats doivent être sous la forme : [{"id": "...", "text": "...", "original_score": ..., "data": {}}]
    Retourne la liste des candidats avec un score de reranking de Gemini.
    """
    global _gemini_model # S'assurer que _gemini_model est accessible

    if not _gemini_model:
        logger.warning("Modèle Gemini non configuré. Skipping reranking.")
        return []

    if not candidates:
        logger.warning("Aucun candidat fourni pour le reranking Gemini.")
        return []

    logger.info(f"Début du reranking Gemini pour {len(candidates)} candidats.")

    # Construire le prompt pour Gemini
    prompt_parts = [
        f"Vous êtes un assistant d'évaluation de CV. Évaluez la pertinence de chaque CV suivant par rapport à la requête:\nRequête: \"{query}\"\n\n",
        "Pour chaque CV, fournissez un score de pertinence entre 0.0 (non pertinent) et 1.0 (très pertinent).\n",
        "Votre réponse doit être un OBJET JSON valide STRICTEMENT, contenant une clé 'scores' qui est une liste d'objets. Chaque objet de la liste doit avoir 'id' (l'identifiant du CV) et 'score' (le score de pertinence).\n",
        "Ne retournez AUCUN texte supplémentaire, préface, explication, ou formatage autre que le JSON pur.\n",
        "Voici les CVs à évaluer (ID du CV et contenu textuel combiné):\n"
    ]

    for i, cv in enumerate(candidates):
        # Limiter la longueur du texte du CV envoyé à Gemini pour éviter de dépasser la limite de tokens
        truncated_text = cv["text"][:1000] # Limiter à 1000 caractères par exemple
        prompt_parts.append(f"--- CV {i+1} ---\nID: {cv['id']}\nContenu: {truncated_text}\n")

    prompt_parts.append("\nRéponse JSON:")

    full_prompt = "".join(prompt_parts)

    try:
        response = _gemini_model.generate_content(full_prompt)

        logger.info(f"DEBUG Gemini raw response: '{response.text}'")

        # Nettoyer la réponse pour s'assurer que c'est un JSON valide
        json_string = response.text.strip()
        if json_string.startswith("```json"):
            json_string = json_string[len("```json"):].strip()
        if json_string.endswith("```"):
            json_string = json_string[:-len("```")].strip()

        gemini_json_response = json.loads(json_string)

        if not isinstance(gemini_json_response, dict) or "scores" not in gemini_json_response:
            logger.error("❌ La réponse de Gemini ne contient pas la structure 'scores' attendue.")
            return []

        reranked_scores = {item['id']: item['score'] for item in gemini_json_response.get('scores', []) if 'id' in item and 'score' in item}

        results_with_rerank_scores = []
        for cv_data in candidates:
            cv_id = cv_data['id']
            rerank_score = reranked_scores.get(cv_id, cv_data.get("original_score", 0.0))

            results_with_rerank_scores.append({
                "id": cv_id,
                "rerank_score": rerank_score,
                "data": cv_data["data"]
            })

        logger.info(f"✅ Reranking Gemini terminé. {len(results_with_rerank_scores)} candidats rerankés.")
        return results_with_rerank_scores

    except json.JSONDecodeError as e:
        logger.error(f"❌ Erreur de décodage JSON après reranking Gemini: {e}")
        if hasattr(response, 'text') and response.text:
            error_pos = e.pos if hasattr(e, 'pos') else 0
            logger.error(f"Partie du texte Gemini incriminée (autour de l'erreur): '{response.text[max(0, error_pos-50):error_pos+50]}'")
        return []
    except Exception as e:
        logger.error(f"❌ Erreur générale lors du reranking Gemini: {e}")
        import traceback
        logger.error(f"Stack trace du reranking Gemini:\n{traceback.format_exc()}")
        return []


# Importation de l'objet 'app' depuis votre __init__.py ou main Flask app
from app import app # Assurez-vous que 'app' est importable ainsi

@app.route("/test-drive")
def test_drive():
    """Test de la connexion Google Drive"""
    try:
        # Assurez-vous que drive_utils est correctement accessible
        from app.utils.drive_utils import test_drive_connection, connect_to_drive, list_pdfs

        result = {"status": "testing"}

        # Test de connexion
        if test_drive_connection():
            result["connection"] = "success"

            # Test de listage des PDFs
            try:
                service = connect_to_drive()
                # Remplacez ceci par l'ID réel de votre dossier Drive
                folder_id = os.environ.get("GOOGLE_DRIVE_FOLDER_ID", "16CpxlBPbm8ZMRBH-B7tj5cmn4h3bXCbt")
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
    global _db_collection # Utilise la variable globale _db_collection
    try:
        debug_data = {}

        # Test MongoDB
        if _db_collection is not None:
            cv_count = _db_collection.count_documents({})
            collections = _db_collection.database.list_collection_names()
            debug_data["mongodb"] = {
                "connected": True,
                "cv_count": cv_count,
                "collections": collections
            }

            if cv_count > 0:
                sample_cv = _db_collection.find_one({})
                debug_data["sample_cv"] = {
                    "nom": sample_cv.get("nom", "N/A"),
                    "competences_count": len(sample_cv.get("competences", [])),
                    "has_biographie": bool(sample_cv.get("biographie"))
                }
        else:
            debug_data["mongodb"] = {"connected": False}

        # Test FAISS - Utilisez la fonction de statut de vectorize.py
        faiss_status = get_faiss_index_status()
        debug_data["faiss"] = faiss_status

        # Test Google Drive credentials
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
    global _db_collection, _gemini_model # Accès aux variables globales

    results = []
    if "likes" not in session:
        session["likes"] = []

    if request.method == "POST":
        if _db_collection is None:
            error_msg = "Base de données non connectée."
            return render_template("index.html", results=[], error=error_msg)

        try:
            query_text = request.form.get("prompt", "").strip()
            if not query_text:
                return render_template("index.html", results=[], error="Veuillez saisir une requête.")

            logger.info(f"🎯 Nouvelle recherche reçue pour: '{query_text}'")

            # 1. Recherche initiale avec FAISS
            faiss_results = []
            try:
                with timeout_context(15):  # 15 secondes max pour la recherche vectorielle
                    faiss_results = search_faiss_index(query_text, top_k=TOP_K)
                logger.info(f"✅ Recherche vectorielle terminée: {len(faiss_results)} résultats préliminaires.")
            except TimeoutError:
                logger.warning("⏰ Timeout recherche vectorielle")
                return render_template("index.html", results=[],
                                       error="Recherche trop longue, veuillez réessayer avec des termes plus simples.")
            except Exception as e:
                logger.error(f"❌ Erreur recherche vectorielle: {e}")
                return render_template("index.html", results=[],
                                       error="Erreur de recherche. Veuillez réessayer.")

            if not faiss_results:
                return render_template("index.html", results=[],
                                       error="Aucun profil trouvé correspondant à votre recherche.")

            # 2. Récupération des données complètes des CVs depuis MongoDB
            cv_ids_to_fetch = [ObjectId(res[0]) for res in faiss_results]

            cvs_from_db = list(_db_collection.find({"_id": {"$in": cv_ids_to_fetch}}))

            cv_data_map = {str(cv["_id"]): cv for cv in cvs_from_db}

            candidates_for_rerank = []
            for cv_id, faiss_score in faiss_results:
                cv_full_data = cv_data_map.get(cv_id)
                if cv_full_data:
                    # Construire le texte à envoyer à Gemini pour le reranking
                    nom = (cv_full_data.get("nom", "") or "").strip()
                    biographie = (cv_full_data.get("biographie", "") or "").strip()
                    competences = ", ".join([str(c).strip() for c in cv_full_data.get("competences", []) if c and isinstance(c, (str, int, float))]).strip() \
                                  if isinstance(cv_full_data.get("competences"), list) else ""

                    experiences_text_parts = []
                    experiences = cv_full_data.get("experiences", [])
                    if isinstance(experiences, list):
                        for exp in experiences[:3]: # Limiter aux 3 premières expériences pour le prompt Gemini
                            if isinstance(exp, dict):
                                exp_detail_parts = []
                                titre = (exp.get("titre", "") or "").strip()
                                entreprise = (exp.get("entreprise", "") or "").strip()
                                description = (exp.get("description", "") or "").strip()
                                if titre: exp_detail_parts.append(titre)
                                if entreprise: exp_detail_parts.append(entreprise)
                                if description and len(description) > 10: exp_detail_parts.append(description[:500])
                                if exp_detail_parts: experiences_text_parts.append(" ".join(exp_detail_parts))
                    experiences_combined = " ".join(experiences_text_parts).strip()

                    formations_text_parts = []
                    formations = cv_full_data.get("formations", [])
                    if isinstance(formations, list):
                        for form in formations[:2]: # Limiter aux 2 premières formations
                            if isinstance(form, dict):
                                form_detail_parts = []
                                diplome = (form.get("diplome", "") or "").strip()
                                etablissement = (form.get("etablissement", "") or "").strip()
                                if diplome: form_detail_parts.append(diplome)
                                if etablissement: form_detail_parts.append(etablissement)
                                if form_detail_parts: formations_text_parts.append(" ".join(form_detail_parts))
                    formations_combined = " ".join(formations_text_parts).strip()

                    combined_text_for_rerank = ". ".join(filter(None, [
                        f"Nom: {nom}",
                        f"Biographie: {biographie}",
                        f"Compétences: {competences}",
                        f"Expériences: {experiences_combined}",
                        f"Formations: {formations_combined}"
                    ])).strip()
                    combined_text_for_rerank = " ".join(combined_text_for_rerank.split()) # Nettoyage espaces multiples

                    candidates_for_rerank.append({
                        "id": str(cv_id),
                        "text": combined_text_for_rerank,
                        "original_score": faiss_score,
                        "data": cv_full_data
                    })
                else:
                    logger.warning(f"CV avec ID {cv_id} trouvé par FAISS mais non trouvé dans la collection MongoDB. Ignoré pour le reranking.")


            # 3. Reranking avec Gemini (si des candidats sont disponibles et Gemini est configuré)
            if _gemini_model is not None and candidates_for_rerank:
                try:
                    logger.info(f"🤖 Reranking Gemini pour {len(candidates_for_rerank)} CVs")
                    with timeout_context(20):  # 20 secondes max pour Gemini
                        reranked_cvs = rerank_with_gemini(query_text, candidates_for_rerank)

                    # Tri des résultats rerankés
                    reranked_cvs.sort(key=lambda x: x["rerank_score"], reverse=True)

                    # Préparer les résultats finaux pour le template HTML
                    for r_cv in reranked_cvs[:TOP_K]: # Limiter à TOP_K ou moins
                        cv_data_from_rerank = r_cv["data"]
                        cv_data_from_rerank["score"] = int(r_cv["rerank_score"] * 100)
                        cv_data_from_rerank["raison"] = f"Pertinence IA: {r_cv['rerank_score']:.2f}"
                        cv_data_from_rerank["liked"] = str(cv_data_from_rerank["_id"]) in session["likes"]
                        results.append(cv_data_from_rerank)

                    logger.info(f"✅ Gemini reranking réussi: {len(results)} profils affichés.")

                except TimeoutError:
                    logger.warning("⏰ Timeout Gemini reranking. Utilisation des scores FAISS comme fallback.")
                    # Fallback si Gemini timeout
                    for cv_data in candidates_for_rerank:
                        cv_data["data"]["score"] = int(cv_data["original_score"] * 100)
                        cv_data["data"]["raison"] = "Correspondance basée sur l'analyse vectorielle (Gemini Timeout)"
                        cv_data["data"]["liked"] = str(cv_data["data"]["_id"]) in session["likes"]
                        results.append(cv_data["data"])
                    results.sort(key=lambda x: x["score"], reverse=True)
                    results = results[:TOP_K]

                except Exception as e:
                    logger.error(f"❌ Erreur lors du reranking Gemini: {e}. Utilisation des scores FAISS comme fallback.")
                    import traceback
                    logger.error(f"Stack trace Gemini fallback:\n{traceback.format_exc()}")
                    # Fallback si erreur Gemini
                    for cv_data in candidates_for_rerank:
                        cv_data["data"]["score"] = int(cv_data["original_score"] * 100)
                        cv_data["data"]["raison"] = "Correspondance basée sur l'analyse vectorielle (Erreur Gemini)"
                        cv_data["data"]["liked"] = str(cv_data["data"]["_id"]) in session["likes"]
                        results.append(cv_data["data"])
                    results.sort(key=lambda x: x["score"], reverse=True)
                    results = results[:TOP_K]
            else:
                # Pas de Gemini configuré ou pas de candidats pour le reranking
                logger.info("📊 Pas de reranking Gemini. Utilisation des scores FAISS.")
                for cv_data in candidates_for_rerank:
                    cv_data["data"]["score"] = int(cv_data["original_score"] * 100)
                    cv_data["data"]["raison"] = "Correspondance basée sur l'analyse vectorielle"
                    cv_data["data"]["liked"] = str(cv_data["data"]["_id"]) in session["likes"]
                    results.append(cv_data["data"])
                results.sort(key=lambda x: x["score"], reverse=True)
                results = results[:TOP_K]

            logger.info(f"🎯 Recherche terminée: {len(results)} résultats affichés pour '{query_text}'.")
            return render_template("index.html", results=results)

        except Exception as e:
            logger.error(f"❌ Erreur critique dans la route de recherche: {e}")
            import traceback
            logger.error(f"Stack trace de l'erreur de recherche:\n{traceback.format_exc()}")
            return render_template("index.html", results=[],
                                   error="Une erreur interne est survenue lors du traitement de votre requête de recherche."), 500

    return render_template("index.html", results=results) # Pour la requête GET initiale

def run_update_background():
    """Lance la mise à jour en arrière-plan."""
    global last_update_time # Accède à la variable globale

    current_time = time.time()
    if current_time - last_update_time >= WATCH_INTERVAL:
        logger.info("🔄 Tentative de mise à jour de l'index FAISS (via run_watch).")
        success = update_faiss_index() # Cette fonction est dans app.utils.vectorize
        if success:
            logger.info("✅ Index FAISS mis à jour et rechargé pour usage interne de vectorize.")
            last_update_time = current_time
        else:
            logger.error("⚠️ Échec de la mise à jour de l'index FAISS.")
    else:
        logger.info(f"Pas de mise à jour nécessaire, prochaine mise à jour dans {WATCH_INTERVAL - (current_time - last_update_time):.0f} secondes.")


@app.route("/clean-index", methods=["POST"])
def clean_index():
    """Nettoie et recrée l'index FAISS."""
    try:
        logger.info("🧹 Début nettoyage index FAISS.")

        clean_success = clean_faiss_index()
        if not clean_success:
            return jsonify({
                "status": "error",
                "message": "Erreur lors du nettoyage de l'index FAISS."
            }), 500

        logger.info("🔄 Recréation de l'index FAISS.")
        create_success = update_faiss_index()

        if create_success:
            # Pour l'affichage, il faudrait une fonction get_faiss_index_size() dans vectorize.py
            # Ici, on se base sur le succès de l'opération.
            faiss_status = get_faiss_index_status()
            return jsonify({
                "status": "success",
                "message": f"Index FAISS recréé avec succès ({faiss_status.get('entries', 0)} profils)"
            })
        else:
            return jsonify({
                "status": "error",
                "message": "Erreur lors de la recréation de l'index FAISS."
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
    global _db_collection, _gemini_model
    try:
        diagnostic_results = {
            "timestamp": str(datetime.now()),
            "tests": {}
        }

        # Test 1: MongoDB
        try:
            if _db_collection is not None:
                cv_count = _db_collection.count_documents({})
                sample_cv = _db_collection.find_one({}) if cv_count > 0 else None

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
                    "error": "Collection non disponible ou non connectée"
                }
        except Exception as e:
            diagnostic_results["tests"]["mongodb"] = {
                "status": "error",
                "error": str(e)
            }

        # Test 2: SentenceTransformer (via une fonction dans vectorize.py)
        try:
            model_info = get_model_info()
            test_vector = test_embedding_model("test text for embedding")

            diagnostic_results["tests"]["sentence_transformer"] = {
                "status": "success",
                "model_name": model_info.get("model_name", "unknown"),
                "vector_dimension": model_info.get("vector_dimension", "unknown"),
                "test_vector_length": len(test_vector) if test_vector is not None else "N/A"
            }
        except Exception as e:
            diagnostic_results["tests"]["sentence_transformer"] = {
                "status": "error",
                "error": str(e)
            }

        # Test 3: FAISS Index (via une fonction dans vectorize.py)
        try:
            faiss_status = get_faiss_index_status()

            diagnostic_results["tests"]["faiss"] = {
                "status": faiss_status.get("status", "unknown"),
                "entries": faiss_status.get("entries", 0),
                "dimension": faiss_status.get("dimension", "unknown"),
                "index_type": faiss_status.get("index_type", "unknown")
            }
        except Exception as e:
            diagnostic_results["tests"]["faiss"] = {
                "status": "error",
                "error": str(e)
            }

        # Test 4: Recherche vectorielle (déjà dans vectorize.py)
        try:
            test_results = search_faiss_index("développeur Python", top_k=3)

            diagnostic_results["tests"]["vector_search"] = {
                "status": "success" if test_results else "no_results",
                "results_count": len(test_results),
                "sample_scores": [r[1] for r in test_results[:3]] if test_results else [] # r[1] est le score
            }
        except Exception as e:
            diagnostic_results["tests"]["vector_search"] = {
                "status": "error",
                "error": str(e)
            }

        # Test 5: Gemini
        try:
            if _gemini_model is not None:
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

        # Calcul du statut global
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
            .success { background-color: #d4edda; border-left: 5px solid #28a745; }
            .error { background-color: #f8d7da; border-left: 5px solid #dc3545; }
            .warning { background-color: #fff3cd; border-left: 5px solid #ffc107; }
            pre { background: #f8f9fa; padding: 10px; border-radius: 4px; }
            button { padding: 10px 20px; background: #007bff; color: white; border: none; border-radius: 4px; cursor: pointer; }
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
                    const statusClass = result.status === 'success' || result.status === 'available' || result.status === 'loaded_from_db' ? 'success' :
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
    global _db_collection

    try:
        if _db_collection is None:
            return jsonify({
                "status": "error",
                "message": "Base de données non connectée"
            }), 503

        # Lancer le processus en arrière-plan
        thread = threading.Thread(target=run_update_background)
        thread.daemon = True
        thread.start()

        logger.info("🚀 Mise à jour des CVs lancée en arrière-plan")

        return jsonify({
            "status": "started",
            "message": "Mise à jour des CVs lancée en arrière-plan. Vérifiez les logs pour le statut."
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
    global _db_collection
    try:
        if _db_collection is None:
            return jsonify({"mongodb": False, "faiss": False, "cv_count": 0, "faiss_entries": 0})

        cv_count = _db_collection.count_documents({})

        faiss_status = get_faiss_index_status()

        return jsonify({
            "mongodb": True,
            "cv_count": cv_count,
            "faiss": faiss_status.get("status") in ["success", "loaded_from_db", "available"],
            "faiss_entries": faiss_status.get("entries", 0)
        })
    except Exception as e:
        logger.error(f"❌ Erreur update-status: {e}")
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
    global _db_collection

    if _db_collection is None:
        return render_template("likes.html", results=[])

    liked_ids = session.get("likes", [])
    liked_cvs = []
    for cid in liked_ids:
        try:
            cv = _db_collection.find_one({"_id": ObjectId(cid)})
            if cv:
                liked_cvs.append(cv)
        except Exception as e:
            logger.error(f"Erreur récupération CV {cid}: {e}")
    return render_template("likes.html", results=liked_cvs)

@app.route("/cv/<cv_id>")
def show_cv_detail(cv_id):
    global _db_collection

    if _db_collection is None:
        return "Base de données non disponible", 503

    try:
        cv = _db_collection.find_one({"_id": ObjectId(cv_id)})
        if not cv:
            return "CV non trouvé", 404
        return render_template("cv_detail.html", cv=cv)
    except Exception as e:
        logger.error(f"Erreur affichage CV {cv_id}: {e}")
        return "ID invalide", 400

@app.route("/health")
def health_check():
    """Endpoint de santé pour vérifier le statut des services"""
    global _db_collection, _gemini_model
    try:
        faiss_status = get_faiss_index_status()

        cv_count = 0
        if _db_collection is not None:
            cv_count = _db_collection.count_documents({})

        status = {
            "mongodb": _db_collection is not None,
            "cv_count": cv_count,
            "faiss": faiss_status.get("status") in ["success", "loaded_from_db", "available"],
            "faiss_entries": faiss_status.get("entries", 0),
            "gemini": _gemini_model is not None
        }
        return jsonify(status), 200
    except Exception as e:
        logger.error(f"❌ Erreur health check: {e}")
        return jsonify({"error": str(e)}), 500
