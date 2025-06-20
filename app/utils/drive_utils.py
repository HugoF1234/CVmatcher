import os.path
import io
import json
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload

SCOPES = ['https://www.googleapis.com/auth/drive.readonly']

def connect_to_drive():
    """Connexion à Google Drive avec gestion des credentials sur Render"""
    creds = None
    
    # Chemins possibles pour les credentials
    possible_paths = [
        'token.json',
        'credentials/token.json',
        os.path.join(os.path.dirname(__file__), '..', '..', 'credentials', 'token.json')
    ]
    
    # Chercher le fichier token.json
    token_path = None
    for path in possible_paths:
        if os.path.exists(path):
            token_path = path
            break
    
    if token_path:
        print(f"📁 Token trouvé : {token_path}")
        creds = Credentials.from_authorized_user_file(token_path, SCOPES)
    else:
        print("⚠️ Aucun token trouvé")
    
    # Si pas de credentials valides, essayer d'en créer
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            print("🔄 Rafraîchissement du token...")
            try:
                creds.refresh(Request())
            except Exception as e:
                print(f"❌ Erreur rafraîchissement : {e}")
                creds = None
        
        if not creds:
            print("🔑 Création de nouveaux credentials...")
            
            # Chercher credentials.json
            creds_paths = [
                'credentials.json',
                'credentials/credentials.json',
                os.path.join(os.path.dirname(__file__), '..', '..', 'credentials', 'credentials.json')
            ]
            
            creds_path = None
            for path in creds_paths:
                if os.path.exists(path):
                    creds_path = path
                    break
            
            if not creds_path:
                raise FileNotFoundError("❌ Fichier credentials.json non trouvé")
            
            print(f"📁 Credentials trouvé : {creds_path}")
            flow = InstalledAppFlow.from_client_secrets_file(creds_path, SCOPES)
            
            # Sur Render, on ne peut pas faire run_local_server, donc on va essayer une approche différente
            try:
                creds = flow.run_local_server(port=0)
                
                # Sauvegarder le token pour la prochaine fois
                if token_path:
                    with open(token_path, 'w') as token:
                        token.write(creds.to_json())
                        print(f"💾 Token sauvegardé : {token_path}")
                        
            except Exception as e:
                print(f"❌ Impossible de créer les credentials sur Render : {e}")
                raise
    
    return build('drive', 'v3', credentials=creds)

def test_drive_connection():
    """Test de la connexion Google Drive"""
    try:
        service = connect_to_drive()
        
        # Test simple : lister quelques fichiers
        results = service.files().list(pageSize=5, fields="files(id, name)").execute()
        files = results.get('files', [])
        
        print(f"✅ Connexion Google Drive réussie - {len(files)} fichiers trouvés")
        return True
        
    except Exception as e:
        print(f"❌ Erreur connexion Google Drive : {e}")
        return False

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

def download_file(service, file_id, file_name):
    request = service.files().get_media(fileId=file_id)
    with io.FileIO(file_name, 'wb') as fh:
        downloader = MediaIoBaseDownload(fh, request)
        done = False
        while not done:
            status, done = downloader.next_chunk()
    print(f"✅ Téléchargé : {file_name}")

if __name__ == '__main__':
    # Test de connexion
    if test_drive_connection():
        service = connect_to_drive()
        folder_id = '16CpxlBPbm8ZMRBH-B7tj5cmn4h3bXCbt'
        
        print("🔍 Recherche des fichiers PDF...")
        pdfs = list_pdfs(service, folder_id)

        if not pdfs:
            print("❌ Aucun PDF trouvé dans le dossier.")
        else:
            print(f"📁 {len(pdfs)} PDFs trouvés")
            for pdf in pdfs:
                print(f"  - {pdf['name']}")
    else:
        print("❌ Impossible de se connecter à Google Drive")
