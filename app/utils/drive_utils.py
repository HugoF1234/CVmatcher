import os.path
import io
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload

SCOPES = ['https://www.googleapis.com/auth/drive.readonly']

def connect_to_drive():
    creds = None
    if os.path.exists('token.json'):
        creds = Credentials.from_authorized_user_file('token.json', SCOPES)
    else:
        flow = InstalledAppFlow.from_client_secrets_file('creditial.json', SCOPES)
        creds = flow.run_local_server(port=0)
        with open('token.json', 'w') as token:
            token.write(creds.to_json())
    return build('drive', 'v3', credentials=creds)

def list_pdfs(service, folder_id):
    query = f"'{folder_id}' in parents and mimeType='application/pdf'"
    results = service.files().list(q=query, pageSize=20, fields="files(id, name)").execute()
    return results.get('files', [])

def download_file(service, file_id, file_name):
    request = service.files().get_media(fileId=file_id)
    with io.FileIO(file_name, 'wb') as fh:
        downloader = MediaIoBaseDownload(fh, request)
        done = False
        while not done:
            status, done = downloader.next_chunk()
    print(f"✅ Téléchargé : {file_name}")

if __name__ == '__main__':
    service = connect_to_drive()

    folder_id = '16CpxlBPbm8ZMRBH-B7tj5cmn4h3bXCbt'


    print("🔍 Recherche des fichiers PDF...")
    pdfs = list_pdfs(service, folder_id)

    if not pdfs:
        print("❌ Aucun PDF trouvé dans le dossier.")
    else:
        for pdf in pdfs:
            download_file(service, pdf['id'], pdf['name'])
