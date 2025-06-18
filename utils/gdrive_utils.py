import os.path
import mimetypes
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow

from google.oauth2 import service_account
from googleapiclient.discovery import build

from googleapiclient.http import MediaIoBaseDownload, MediaFileUpload
import io

from tqdm import tqdm

def connect_to_gdrive():
    creds = service_account.Credentials.from_service_account_file(
        'credentials.json',
        scopes=['https://www.googleapis.com/auth/drive']
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

def get_or_create_folder(service, folder_name, parent_id=None):
  
    query = f"mimeType='application/vnd.google-apps.folder' and name='{folder_name}' and trashed = false"
    if parent_id:
        query += f" and '{parent_id}' in parents"
    else:
        query += " and 'root' in parents"

    response = service.files().list(q=query, fields="files(id, name)").execute()
    files = response.get('files', [])
    
    if files:
        return files[0]['id']
    else:
        folder_metadata = {
            'name': folder_name,
            'mimeType': 'application/vnd.google-apps.folder',
        }
        if parent_id:
            folder_metadata['parents'] = [parent_id]
        folder = service.files().create(body=folder_metadata, fields='id').execute()
        return folder.get('id')



def get_folder_id_by_path(service, path_list, root_parent_id=None):
    connect_to_gdrive()
    parent_id = root_parent_id
    for folder_name in path_list:
        parent_id = get_or_create_folder(service, folder_name, parent_id)
    return parent_id

def create_or_get_folder(service, folder_name, parent_folder_id):
    """Create or find folder in Google Drive."""
    query = (
        f"'{parent_folder_id}' in parents and name = '{folder_name}' "
        "and mimeType = 'application/vnd.google-apps.folder' and trashed = false"
    )
    results = service.files().list(q=query, spaces='drive').execute()
    files = results.get('files', [])
    if files:
        return files[0]['id']  # Folder exists
    else:
        file_metadata = {
            'name': folder_name,
            'mimeType': 'application/vnd.google-apps.folder',
            'parents': [parent_folder_id]
        }
        folder = service.files().create(body=file_metadata, fields='id').execute()
        return folder.get('id')

def upload_file_to_drive(service, local_file_path, drive_folder_id, drive_file_name=None, mimetype=None):
    """
    Uploads a file or all files in a folder to Google Drive.
    If local_file_path is a folder, uploads each file into a created subfolder in Google Drive.
    """

    if os.path.isdir(local_file_path):
        # If it's a directory, create a corresponding folder in GDrive
        folder_name = os.path.basename(local_file_path.rstrip("/"))
        gdrive_subfolder_id = create_or_get_folder(service, folder_name, drive_folder_id)

        # Recursively upload files in this folder
        for root, _, files in os.walk(local_file_path):
            for filename in files:
                file_path = os.path.join(root, filename)
                rel_path = os.path.relpath(file_path, local_file_path)
                print(f"Uploading file: {rel_path} from folder: {local_file_path}")

                file_metadata = {
                    'name': filename,
                    'parents': [gdrive_subfolder_id]
                }

                mimetype = mimetypes.guess_type(file_path)[0] or 'application/octet-stream'
                media = MediaFileUpload(file_path, mimetype=mimetype, resumable=True)

                request = service.files().create(
                    body=file_metadata, media_body=media, fields='id, name'
                )
                response = None
                while response is None:
                    status, response = request.next_chunk()
                    if status:
                        print(f"Upload progress: {int(status.progress() * 100)}%")

                print(f"Uploaded file '{response.get('name')}' with ID: {response.get('id')}")
        return None  # For folders, no single file ID is returned

    else:
        # If it's a single file
        if drive_file_name is None:
            drive_file_name = os.path.basename(local_file_path)
        if mimetype is None:
            mimetype = mimetypes.guess_type(local_file_path)[0] or 'application/octet-stream'

        file_metadata = {
            'name': drive_file_name,
            'parents': [drive_folder_id]
        }
        media = MediaFileUpload(local_file_path, mimetype=mimetype, resumable=True)

        request = service.files().create(body=file_metadata, media_body=media, fields='id, name')
        response = None
        while response is None:
            status, response = request.next_chunk()
            if status:
                print(f"Upload progress: {int(status.progress() * 100)}%")
        print(f"Uploaded file '{response.get('name')}' with ID: {response.get('id')}")
        return response.get('id')