import os
import io
import json
import fitz  # PyMuPDF
import re
import time
import google.generativeai as genai
from bson import ObjectId
from googleapiclient.http import MediaIoBaseDownload
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build

# === CONFIGURATION ===
SCOPES = ['https://www.googleapis.com/auth/drive.readonly']
from config import GEMINI_API_KEY

def get_mongo_collection():
    """Récupère la collection MongoDB en utilisant la fonction centralisée"""
    try:
        from config import get_mongo_client, DB_NAME, COLLECTION_NAME
        
        client = get_mongo_client()
        if not client:
            print("❌ Impossible de se connecter à MongoDB")
            return None
            
        db = client[DB_NAME]
        collection = db[COLLECTION_NAME]
        print(f"✅ Connexion MongoDB réussie - Collection: {COLLECTION_NAME}")
        return collection
        
    except Exception as e:
        print(f"❌ Erreur connexion MongoDB: {e}")
        return None

# === CONNEXION GOOGLE DRIVE ===
def connect_to_drive():
    """Utilise la fonction de connexion centralisée"""
    from app.utils.drive_utils import connect_to_drive as connect_drive
    return connect_drive()

def list_pdfs(service, folder_id):
    """Utilise la fonction de listage centralisée"""
    from app.utils.drive_utils import list_pdfs as list_drive_pdfs
    return list_drive_pdfs(service, folder_id)
    
def process_and_insert_cv(filename, collection=None):
    """Traite un CV et l'insère en base de données"""
    try:
        print(f"🔄 Traitement de {filename}...")
        
        # Extraction du texte
        text = extract_text_from_pdf(filename)
        if not text.strip():
            print(f"⚠️ Aucun texte extrait de {filename}")
            return False
        
        # Extraction structurée avec Gemini
        extracted_data = extract_info_with_gemini(text, filename=filename)
        if not extracted_data:
            print(f"⚠️ Extraction Gemini échouée pour {filename}")
            return False
        
        # Enrichissement
        enriched_data = enrich_with_bio_and_sector(extracted_data)
        
        # Insertion en base
        success = insert_into_mongodb(enriched_data, collection=collection)
        
        if success:
            print(f"✅ CV {filename} traité et inséré avec succès")
            # Ajout incrémental à l'index FAISS
            try:
                from app.utils.vectorize import add_cv_to_faiss_index
                faiss_success = add_cv_to_faiss_index(enriched_data)
                if faiss_success:
                    print(f"✅ Ajout incrémental FAISS réussi pour {enriched_data.get('nom', filename)}")
                else:
                    print(f"⚠️ Ajout incrémental FAISS échoué pour {enriched_data.get('nom', filename)}")
            except Exception as e:
                print(f"❌ Erreur ajout incrémental FAISS: {e}")
        else:
            print(f"❌ Échec insertion {filename}")
            
        return success
        
    except Exception as e:
        print(f"❌ Erreur traitement {filename}: {e}")
        return False
    
def download_file(service, file_id, file_name):
    """Utilise la fonction de téléchargement centralisée"""
    from app.utils.drive_utils import download_file as download_drive_file
    return download_drive_file(service, file_id, file_name)

# === EXTRACTION TEXTE DU PDF ===
def extract_text_from_pdf(file_path):
    """Extrait le texte d'un fichier PDF"""
    try:
        doc = fitz.open(file_path)
        text = ""
        for page in doc:
            text += page.get_text()
        doc.close()
        return text
    except Exception as e:
        print(f"❌ Erreur extraction PDF {file_path}: {e}")
        return ""

# === FONCTION AVEC RETRY POUR GEMINI ===
def call_gemini_with_retry(model, prompt, max_retries=3, base_delay=5):
    """Appelle Gemini avec retry automatique en cas de quota exceeded"""
    for attempt in range(max_retries):
        try:
            response = model.generate_content(prompt)
            return response.text.strip()
        except Exception as e:
            error_str = str(e)
            
            # Si c'est une erreur de quota (429)
            if "429" in error_str or "quota" in error_str.lower():
                delay_seconds = base_delay * (2 ** attempt)  # Backoff exponentiel
                print(f"⏱️ Quota dépassé, attente {delay_seconds}s (tentative {attempt + 1}/{max_retries})")
                
                if attempt < max_retries - 1:  # Ne pas attendre après la dernière tentative
                    time.sleep(delay_seconds)
                    continue
            
            # Autres erreurs ou dernière tentative
            print(f"❌ Erreur Gemini (tentative {attempt + 1}/{max_retries}): {e}")
            if attempt == max_retries - 1:
                raise e
    
    return None

# === EXTRACTION STRUCTURÉE AVEC GEMINI ===
def extract_info_with_gemini(cv_text, filename=""):
    """Extrait les informations structurées avec Gemini 2.0 Flash"""
    try:
        genai.configure(api_key=GEMINI_API_KEY)
        model = genai.GenerativeModel('gemini-2.0-flash')  # ✅ Utiliser 2.0 Flash

        prompt = (
            "Voici un CV. Peux-tu me retourner un JSON structuré avec les champs suivants :\n"
            "- nomdupdf (le nom du fichier PDF)\n"
            "- nom\n"
            "- email\n"
            "- competences (liste)\n"
            "- formations (liste d'objets avec diplome, annee, etablissement)\n"
            "- experiences (liste d'objets avec titre, entreprise, type, dateDebut, dateFin, lieu, statut, importance (note sur 10), description (texte de 200 mots environ généré par toi))\n"
            "- langues (liste)\n"
            "- secteur (liste)\n"
            "- biographie (rédigée par toi, décrivant le parcours professionnel du profil)\n"
            "Réponds uniquement avec du JSON valide. Pas de texte autour.\n\n"
            f"Nom du fichier PDF : {filename}\n\n"
            f"Contenu du CV :\n{cv_text[:8000]}"  # Limiter pour éviter les tokens excessifs
        )

        raw_text = call_gemini_with_retry(model, prompt)
        if not raw_text:
            return None

        # Nettoyage du JSON
        if raw_text.startswith("```json"):
            raw_text = raw_text[7:-3].strip()
        elif raw_text.startswith("```"):
            raw_text = raw_text[3:-3].strip()

        # Supprimer les commentaires
        raw_text = re.sub(r"//.*", "", raw_text)
        raw_text = "".join([line for line in raw_text.splitlines() if line.strip() != ""])
        
        # Trouver la dernière accolade fermante
        last_closing = raw_text.rfind("}")
        if last_closing != -1:
            raw_text = raw_text[:last_closing + 1]

        data = json.loads(raw_text)
        print(f"✅ Extraction Gemini réussie pour {filename}")
        return data
        
    except json.JSONDecodeError as e:
        print(f"⚠️ Erreur parsing JSON pour {filename}: {e}")
        print(f"Début de réponse Gemini : {raw_text[:200]}..." if 'raw_text' in locals() else "Pas de réponse")
        return None
    except Exception as e:
        print(f"❌ Erreur Gemini pour {filename}: {e}")
        return None

# === ENRICHISSEMENT AVEC BIOGRAPHIE & SECTEUR ===
def enrich_with_bio_and_sector(cv_data):
    """Enrichit les données CV avec biographie et secteur"""
    try:
        genai.configure(api_key=GEMINI_API_KEY)
        model = genai.GenerativeModel('gemini-2.0-flash')  # ✅ Utiliser 2.0 Flash

        nom = cv_data.get("nom", "")
        experiences = cv_data.get("experiences", [])
        competences = cv_data.get("competences", [])

        context = f"Nom: {nom}\nCompétences: {', '.join(competences)}\n\nExpériences :\n"
        for exp in experiences:
            context += f"- {exp.get('titre', '')} chez {exp.get('entreprise', '')} ({exp.get('dateDebut', '')} - {exp.get('dateFin', '')})\n"

        # Génération de la biographie
        if not cv_data.get("biographie"):
            bio_prompt = f"Voici un résumé de carrière à générer pour ce profil :\n{context}\nÉcris une biographie professionnelle synthétique en 2-3 phrases."
            bio_text = call_gemini_with_retry(model, bio_prompt)
            if bio_text:
                cv_data["biographie"] = bio_text

        # Génération du secteur
        if not cv_data.get("secteur") or isinstance(cv_data["secteur"], str):
            sector_prompt = f"D'après ce profil, liste les secteurs professionnels associés (ex: finance, tech, retail). Donne-les dans une liste séparée par des virgules.\n{context}"
            sector_text = call_gemini_with_retry(model, sector_prompt)
            if sector_text:
                # Convertir en liste si c'est une chaîne
                cv_data["secteur"] = [s.strip() for s in sector_text.split(",")]

        print(f"✅ Enrichissement réussi pour {cv_data.get('nom', 'CV')}")
        return cv_data
        
    except Exception as e:
        print(f"⚠️ Erreur enrichissement: {e}")
        return cv_data

# === INSERTION DANS MONGODB ===
def insert_into_mongodb(data, collection=None):
    """Insère les données dans MongoDB"""
    try:
        if collection is None:
            collection = get_mongo_collection()
        if collection is None:
            return False
            
        existing = collection.find_one({
            "$or": [
                {"nom": data.get("nom")},
                {"nomdupdf": data.get("nomdupdf")}
            ]
        })
        
        if existing:
            print(f"⚠️ CV {data.get('nom', 'Inconnu')} existe déjà, mise à jour...")
            collection.replace_one({"_id": existing["_id"]}, data)
        else:
            collection.insert_one(data)
            
        print(f"✅ Insertion MongoDB réussie pour {data.get('nom', 'CV')}")
        return True
        
    except Exception as e:
        print(f"❌ Erreur insertion MongoDB: {e}")
        return False

# === MAIN ===
if __name__ == '__main__':
    print("🚀 Test d'enrichissement des CVs avec Gemini 2.0 Flash")
    
    try:
        service = connect_to_drive()
        from config import GOOGLE_DRIVE_FOLDER_ID
        folder_id = GOOGLE_DRIVE_FOLDER_ID
        pdfs = list_pdfs(service, folder_id)

        print(f"📁 {len(pdfs)} PDFs trouvés")

        for i, pdf in enumerate(pdfs[:2], 1):  # Test sur les 2 premiers
            print(f"\n--- Test {i}/{min(2, len(pdfs))} ---")
            filename = pdf['name']
            
            # Téléchargement
            if download_file(service, pdf['id'], filename):
                # Traitement
                success = process_and_insert_cv(filename)
                
                # Nettoyage
                try:
                    os.remove(filename)
                    print(f"🗑️ Fichier temporaire {filename} supprimé")
                except:
                    pass
                    
                if not success:
                    print(f"❌ Échec traitement {filename}")
            else:
                print(f"❌ Échec téléchargement {filename}")
                
    except Exception as e:
        print(f"❌ Erreur globale: {e}")
