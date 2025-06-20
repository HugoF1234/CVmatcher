import os.path
import io
import json
from google.oauth2.credentials import Credentials
from google.auth.transport.requests import Request
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload

SCOPES = ['https://www.googleapis.com/auth/drive.readonly']

def connect_to_drive():
    """Connexion à Google Drive avec gestion spéciale pour Render"""
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
        try:
            creds = Credentials.from_authorized_user_file(token_path, SCOPES)
        except Exception as e:
            print(f"❌ Erreur lecture token : {e}")
            creds = None
    else:
        print("⚠️ Aucun token trouvé")
    
    # Vérifier et rafraîchir les credentials
    if creds and creds.expired and creds.refresh_token:
        print("🔄 Rafraîchissement du token...")
        try:
            creds.refresh(Request())
            print("✅ Token rafraîchi avec succès")
            
            # Sauvegarder le token rafraîchi
            if token_path:
                with open(token_path, 'w') as token:
                    token.write(creds.to_json())
                    print(f"💾 Token sauvegardé : {token_path}")
                    
        except Exception as e:
            print(f"❌ Erreur rafraîchissement : {e}")
            creds = None
    
    # Si pas de credentials valides
    if not creds or not creds.valid:
        print("❌ Credentials non valides ou expirés")
        
        # Sur Render, on ne peut pas faire l'auth interactive
        # Il faut utiliser des credentials pré-configurés
        raise Exception(
            "Credentials Google Drive expirés ou invalides. "
            "Sur Render, vous devez utiliser des credentials pré-configurés. "
            "Régénérez le token.json localement et redéployez."
        )
    
    return build('drive', 'v3', credentials=creds)

def test_drive_connection():
    """Test de la connexion Google Drive"""
    try:
        service = connect_to_drive()
        
        # Test simple : lister quelques fichiers dans le dossier racine
        results = service.files().list(pageSize=5, fields="files(id, name)").execute()
        files = results.get('files', [])
        
        print(f"✅ Connexion Google Drive réussie - {len(files)} fichiers trouvés")
        return True
        
    except Exception as e:
        print(f"❌ Erreur connexion Google Drive : {e}")
        return False

def list_pdfs(service, folder_id):
    """Liste les PDFs dans un dossier Google Drive"""
    try:
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
    except Exception as e:
        print(f"❌ Erreur listage PDFs : {e}")
        return []

def download_file(service, file_id, file_name):
    """Télécharge un fichier depuis Google Drive"""
    try:
        request = service.files().get_media(fileId=file_id)
        with io.FileIO(file_name, 'wb') as fh:
            downloader = MediaIoBaseDownload(fh, request)
            done = False
            while not done:
                status, done = downloader.next_chunk()
        print(f"✅ Téléchargé : {file_name}")
        return True
    except Exception as e:
        print(f"❌ Erreur téléchargement {file_name} : {e}")
        return False

# Fonction pour régénérer le token (à utiliser localement)
def regenerate_token():
    """À exécuter localement pour régénérer le token"""
    creds_paths = [
        'credentials.json',
        'credentials/credentials.json'
    ]
    
    creds_path = None
    for path in creds_paths:
        if os.path.exists(path):
            creds_path = path
            break
    
    if not creds_path:
        print("❌ Fichier credentials.json non trouvé")
        return False
    
    print(f"📁 Utilisation de : {creds_path}")
    flow = InstalledAppFlow.from_client_secrets_file(creds_path, SCOPES)
    creds = flow.run_local_server(port=0)
    
    # Sauvegarder le token
    with open('credentials/token.json', 'w') as token:
        token.write(creds.to_json())
        print("💾 Token sauvegardé dans credentials/token.json")
    
    return True

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
        print("💡 Exécutez localement : python -c 'from app.utils.drive_utils import regenerate_token; regenerate_token()'")
