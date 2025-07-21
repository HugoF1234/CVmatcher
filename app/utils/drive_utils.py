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
    """Connexion à Google Drive - Version qui marche sur Render/GCP et en local"""
    creds = None
    token_json_str = os.environ.get('GOOGLE_TOKEN_JSON')

    if token_json_str:
        # Priorité 1: Charger depuis la variable d'environnement (pour GCP/Render)
        try:
            token_info = json.loads(token_json_str)
            creds = Credentials.from_authorized_user_info(token_info, SCOPES)
            print("📖 Credentials chargés depuis la variable d'environnement GOOGLE_TOKEN_JSON")
        except Exception as e:
            print(f"❌ Erreur de chargement des credentials depuis l'environnement : {e}")
            # Si l'env var est mal formée, on ne continue pas pour éviter des erreurs ambiguës
            return None
    else:
        # Priorité 2: Charger depuis le fichier token.json (pour le dev local)
        possible_token_paths = [
            'token.json',
            'credentials/token.json',
        ]
        token_path = None
        for path in possible_token_paths:
            if os.path.exists(path):
                token_path = path
                print(f"📁 Token trouvé pour le développement local : {token_path}")
                break
        
        if token_path:
            try:
                creds = Credentials.from_authorized_user_file(token_path, SCOPES)
                print("📖 Credentials chargés depuis le fichier token.json local")
            except Exception as e:
                print(f"❌ Erreur lecture token local : {e}")

    # Si on n'a toujours pas de credentials, c'est une erreur.
    if not creds:
        raise Exception(
            "❌ Credentials Google Drive non trouvés. "
            "Assurez-vous que le secret GOOGLE_TOKEN_JSON est défini sur le serveur "
            "ou que le fichier token.json est présent localement."
        )

    # Vérifier et rafraîchir si nécessaire (commun aux deux méthodes)
    if creds.expired and creds.refresh_token:
        print("🔄 Rafraîchissement du token...")
        try:
            from google.auth.transport.requests import Request
            creds.refresh(Request())
            print("✅ Token rafraîchi avec succès")
        except Exception as e:
            print(f"❌ Erreur de rafraîchissement du token : {e}")
            # On ne bloque pas ici, on laisse la suite décider si les creds sont valides
    
    return build('drive', 'v3', credentials=creds)

def test_drive_connection():
    """Test de la connexion Google Drive"""
    try:
        service = connect_to_drive()
        
        # Test simple : lister quelques fichiers
        results = service.files().list(pageSize=5, fields="files(id, name)").execute()
        files = results.get('files', [])
        
        print(f"✅ Connexion Google Drive réussie - {len(files)} fichiers accessibles")
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

        print(f"🔍 Recherche des PDFs dans le dossier {folder_id}...")

        while True:
            response = service.files().list(
                q=query,
                spaces='drive',
                fields='nextPageToken, files(id, name, size)',
                pageToken=page_token,
                pageSize=50  # Plus de fichiers par page
            ).execute()

            new_files = response.get('files', [])
            files.extend(new_files)
            
            page_token = response.get('nextPageToken', None)
            if page_token is None:
                break

        print(f"📁 {len(files)} PDFs trouvés au total")
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
                if status:
                    progress = int(status.progress() * 100)
                    # Afficher le progrès seulement tous les 25%
                    if progress % 25 == 0:
                        print(f"  📥 {file_name}: {progress}%")
        
        print(f"✅ Téléchargé : {file_name}")
        return True
        
    except Exception as e:
        print(f"❌ Erreur téléchargement {file_name} : {e}")
        return False

if __name__ == '__main__':
    # Test de connexion
    if test_drive_connection():
        service = connect_to_drive()
        folder_id = '16CpxlBPbm8ZMRBH-B7tj5cmn4h3bXCbt'
        
        pdfs = list_pdfs(service, folder_id)

        if pdfs:
            print(f"\n📋 Liste des {len(pdfs)} PDFs :")
            for i, pdf in enumerate(pdfs, 1):
                size = int(pdf.get('size', 0))
                size_mb = size / (1024 * 1024) if size > 0 else 0
                print(f"  {i:2d}. {pdf['name']} ({size_mb:.1f} MB)")
        else:
            print("❌ Aucun PDF trouvé dans le dossier.")
    else:
        print("❌ Impossible de se connecter à Google Drive")
