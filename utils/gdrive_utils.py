import os.path
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow

from google.oauth2 import service_account
from googleapiclient.discovery import build

from googleapiclient.http import MediaIoBaseDownload
import io

from tqdm import tqdm

def connect_to_gdrive():
    creds = service_account.Credentials.from_service_account_file(
        'credentials.json',
        scopes=['https://www.googleapis.com/auth/drive.readonly']
    )

    service = build('drive', 'v3', credentials=creds)

    return service

def list_folder_contents(service, folder_id, parent_path=''):
    all_files = []

    query = f"'{folder_id}' in parents and trashed = false"
    results = service.files().list(q=query,
                                   fields="files(id, name, mimeType)").execute()
    items = results.get('files', [])

    for item in items:
        file_path = os.path.join(parent_path, item['name'])

        if item['mimeType'] == 'application/vnd.google-apps.folder':
            # Recurse into subfolder
            all_files.extend(list_folder_contents(service, item['id'], file_path))
        else:
            # Add file to list
            all_files.append({
                'id': item['id'],
                'name': item['name'],
                'mimeType': item['mimeType'],
                'path': file_path
            })

    return all_files

def download_file(service, file, output_base):
    os.makedirs(os.path.dirname(os.path.join(output_base, file['path'])), exist_ok=True)
    file_path = os.path.join(output_base, file['path'])

    export_formats = {
        'application/vnd.google-apps.document': 'application/pdf',
        'application/vnd.google-apps.spreadsheet': 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
        'application/vnd.google-apps.presentation': 'application/pdf',
    }

    if file['mimeType'] in export_formats:
        # Export Google Docs formats
        request = service.files().export_media(
            fileId=file['id'],
            mimeType=export_formats[file['mimeType']]
        )
        file_path += get_file_extension(export_formats[file['mimeType']])
    else:
        # Binary files
        request = service.files().get_media(fileId=file['id'])

    with io.FileIO(file_path, 'wb') as fh:
        downloader = MediaIoBaseDownload(fh, request)
        done = False
        while not done:
            status, done = downloader.next_chunk()

def get_file_extension(mime_type):
    mapping = {
        'application/pdf': '.pdf',
        'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet': '.xlsx',
        'application/vnd.openxmlformats-officedocument.wordprocessingml.document': '.docx',
    }
    return mapping.get(mime_type, '')

def sync_gdrive_db_to_local():
    service = connect_to_gdrive()
    files = list_folder_contents(service, '1_eMgnRaFtt-ZSZD3zfwai3qlpYJ-M5C6')

    for f in tqdm(files, total=len(files), desc="Downloading data files..."):
        download_file(service, f, '.')
    print("Download complete.")