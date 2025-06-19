import os
import io
import json
import fitz  # PyMuPDF
import re
import google.generativeai as genai
from pymongo import MongoClient
from bson import ObjectId
from googleapiclient.http import MediaIoBaseDownload
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build

# === CONFIGURATION ===
SCOPES = ['https://www.googleapis.com/auth/drive.readonly']
MONGO_URI = "mongodb://localhost:27017"
DB_NAME = "CVExtraction"
COLLECTION_NAME = "CVExtractionCollection"
GEMINI_API_KEY = "AIzaSyBNE6Ak52CMC6aXkMyOBgjfFsm-NHfT6jA"

# === CONNEXION GOOGLE DRIVE ===
def connect_to_drive():
    creds = None
    if os.path.exists('token.json'):
        creds = Credentials.from_authorized_user_file('token.json', SCOPES)
    else:
        flow = InstalledAppFlow.from_client_secrets_file('credentials.json', SCOPES)
        creds = flow.run_local_server(port=0)
        with open('token.json', 'w') as token:
            token.write(creds.to_json())
    return build('drive', 'v3', credentials=creds)

def list_pdfs(service, folder_id):
    query = f"'{folder_id}' in parents and mimeType='application/pdf'"
    files = []
    page_token = None

    while True:
        response = service.files().list(
            q=query,
            spaces='drive',
            fields='nextPageToken, files(id, name)',
            pageToken=page_token
        ).execute()

        files.extend(response.get('files', []))
        page_token = response.get('nextPageToken', None)
        if page_token is None:
            break

    return files
    
def process_and_insert_cv(filename):
    text = extract_text_from_pdf(filename)
    extracted_data = extract_info_with_gemini(text, filename=filename)
    if extracted_data:
        enriched_data = enrich_with_bio_and_sector(extracted_data)
        insert_into_mongodb(enriched_data)
        return True
    return False
    
def download_file(service, file_id, file_name):
    request = service.files().get_media(fileId=file_id)
    with io.FileIO(file_name, 'wb') as fh:
        downloader = MediaIoBaseDownload(fh, request)
        done = False
        while not done:
            _, done = downloader.next_chunk()
    print(f"✅ Téléchargé : {file_name}")

# === EXTRACTION TEXTE DU PDF ===
def extract_text_from_pdf(file_path):
    doc = fitz.open(file_path)
    text = ""
    for page in doc:
        text += page.get_text()
    return text

# === FIX JSON MAL FORMATTÉ ===
def fix_missing_commas(text):
    text = re.sub(r'"\](\s*"[a-zA-Z])', r'"],\1', text)
    text = re.sub(r'(\})\s*("[a-zA-Z])', r'\1, \2', text)
    text = re.sub(r'(\])\s*("[a-zA-Z])', r'\1, \2', text)
    return text

# === EXTRACTION STRUCTURÉE AVEC GEMINI ===
def extract_info_with_gemini(cv_text, filename=""):
    genai.configure(api_key=GEMINI_API_KEY)
    model = genai.GenerativeModel('gemini-1.5-pro')

    prompt = (
        "Voici un CV. Peux-tu me retourner un JSON structuré avec les champs suivants :"
        "- nomdupdf (le nom du fichier PDF)"
        "- nom"
        "- email"
        "- competences (liste)"
        "- formations (liste d'objets avec diplome, annee, etablissement)"
        "- experiences (liste d'objets avec titre, entreprise, type, dateDebut, dateFin, lieu, statut, importance (note sur 10), description (texte de 200 mots environ généré par toi))"
        "- langues (liste)"
        "- secteur (liste)"
        "- biographie (rédigée par toi, décrivant le parcours professionnel du profil)"
        "Réponds uniquement avec du JSON valide. Pas de texte autour."
    )

    full_prompt = f"{prompt}Nom du fichier PDF : {filename}{cv_text}"
    response = model.generate_content(full_prompt)
    raw_text = response.text.strip()

    if raw_text.startswith("```json"):
        raw_text = raw_text[7:-3].strip()
    elif raw_text.startswith("```"):
        raw_text = raw_text[3:-3].strip()

    raw_text = re.sub(r"//.*", "", raw_text)
    raw_text = "".join([line for line in raw_text.splitlines() if line.strip() != ""])
    last_closing = raw_text.rfind("}")
    if last_closing != -1:
        raw_text = raw_text[:last_closing + 1]

    try:
        return json.loads(raw_text)
    except Exception as e:
        print("⚠️ Parsing JSON échoué :", e)
        print("Début de réponse Gemini :", raw_text[:500])
        return None

# === ENRICHISSEMENT AVEC BIOGRAPHIE & SECTEUR ===
def enrich_with_bio_and_sector(cv_data):
    genai.configure(api_key=GEMINI_API_KEY)
    model = genai.GenerativeModel('gemini-1.5-pro')         ##passer au 2.5 si trop d'erreurs de parsing json et prendre une version plus puissante

    nom = cv_data.get("nom", "")
    experiences = cv_data.get("experiences", [])
    competences = cv_data.get("competences", [])

    context = f"Nom: {nom}\nCompétences: {', '.join(competences)}\n\nExpériences :\n"
    for exp in experiences:
        context += f"- {exp.get('titre', '')} chez {exp.get('entreprise', '')} ({exp.get('dateDebut', '')} - {exp.get('dateFin', '')})\n"

    bio_prompt = f"Voici un résumé de carrière à générer pour ce profil :\n{context}\nÉcris une biographie professionnelle synthétique."
    bio_resp = model.generate_content(bio_prompt)
    cv_data["biographie"] = bio_resp.text.strip()

    sector_prompt = f"D'après ce profil, liste les secteurs professionnels associés (ex: finance, tech, retail). Donne-les dans une chaîne séparée par virgules.\n{context}"
    sector_resp = model.generate_content(sector_prompt)
    cv_data["secteur"] = sector_resp.text.strip()

    return cv_data

# === INSERTION DANS MONGODB ===
def insert_into_mongodb(data):
    client = MongoClient(MONGO_URI)
    db = client[DB_NAME]
    collection = db[COLLECTION_NAME]
    collection.insert_one(data)
    print("✅ Insertion MongoDB réussie.")

# === MAIN ===
if __name__ == '__main__':
    service = connect_to_drive()
    folder_id = '16CpxlBPbm8ZMRBH-B7tj5cmn4h3bXCbt'
    pdfs = list_pdfs(service, folder_id)

    for pdf in pdfs:
        filename = pdf['name']
        download_file(service, pdf['id'], filename)

        text = extract_text_from_pdf(filename)
        extracted_data = extract_info_with_gemini(text, filename=filename)

        if extracted_data:
            enriched_data = enrich_with_bio_and_sector(extracted_data)
            insert_into_mongodb(enriched_data)
        else:
            print(f"❌ Extraction échouée pour : {filename}")
