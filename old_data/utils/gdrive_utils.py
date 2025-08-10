import os.path
import mimetypes
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow

from google.oauth2 import service_account
from googleapiclient.discovery import build

from googleapiclient.http import MediaIoBaseDownload, MediaFileUpload
import io

from tqdm import tqdm
from pyspark.sql import SparkSession


import pandas as pd
from datetime import datetime
import tempfile


# Create a SparkSession if you haven't already
spark = SparkSession.builder \
    .appName("ReadParquet") \
    .config("spark.python.worker.reuse", "false") \
    .config("spark.network.timeout", "600s") \
    .getOrCreate()


def connect_to_gdrive():
    creds = service_account.Credentials.from_service_account_file(
        '/opt/airflow/utils/credentials.json',
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
    os.makedirs(os.path.dirname(os.path.join(output_base, file['name'])), exist_ok=True)
    file_path = os.path.join(output_base, file['name'])

    export_formats = {
        'application/vnd.google-apps.document': 'application/pdf',
        'application/vnd.google-apps.spreadsheet': 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
        'application/vnd.google-apps.presentation': 'application/pdf',
    }
        # Get mimeType from API if not present
    if  file['mimeType'] is None:
        print("[INFO] Fetching MIME type via Drive API")
        file_meta = service.files().get(fileId=file['id'], fields='mimeType').execute()
        file['mimeType'] = file_meta['mimeType']

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
    
# overwrite files if exists
def clear_drive_folder(service, folder_id):
    """Delete all files inside a Google Drive folder (non-recursive)."""

    query = f"'{folder_id}' in parents and trashed=false"
    results = service.files().list(q=query, fields="files(id, name)").execute()
    files = results.get('files', [])
    
    for file in files:
        try:
            print(f"Deleting file '{file['name']}' (ID: {file['id']}) from Drive folder.")
            service.files().delete(fileId=file['id']).execute()
        except Exception as e:
            print(f"Error deleting file {file['name']}: {e}")


def upload_file_to_drive(service, local_file_path, drive_folder_id, drive_file_name=None, mimetype=None):
    if os.path.isdir(local_file_path):
        # Directory upload handling
        folder_name = os.path.basename(local_file_path.rstrip("/"))
        gdrive_subfolder_id = create_or_get_folder(service, folder_name, drive_folder_id)

        # CLEAR the target folder before uploading to avoid file (1), file (2), etc.
        clear_drive_folder(service, gdrive_subfolder_id)

        # Recursively upload files in this folder
        for root, _, files in os.walk(local_file_path):
            for filename in files:
                file_path = os.path.join(root, filename)
                rel_path = os.path.relpath(file_path, local_file_path)
                print(f"Uploading file: {rel_path} from folder: {local_file_path}")

                # No need for overwrite_if_exists — folder was cleared
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
        # Single file upload handling
        if drive_file_name is None:
            drive_file_name = os.path.basename(local_file_path)
        if mimetype is None:
            mimetype = mimetypes.guess_type(local_file_path)[0] or 'application/octet-stream'

        # Overwrite existing single file if it exists
        # overwrite_if_exists(service, drive_folder_id, drive_file_name)

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

  
def get_file_by_name(service, folder_id, filename):
    query = f"'{folder_id}' in parents and name = '{filename}' and trashed = false"
    results = service.files().list(q=query, spaces='drive', fields='files(id, name, mimeType)', pageSize=1).execute()
    files = results.get('files', [])
    return files[0] if files else None

def list_parquet_files_in_folder(service, folder_id):
    query = (
        f"'{folder_id}' in parents and trashed = false "
        "and mimeType != 'application/vnd.google-apps.folder'"
    )
    results = service.files().list(
        q=query,
        fields="files(id, name, mimeType)"
    ).execute()
    files = results.get("files", [])
    
    # Strictly return only files that end with ".parquet" (not .crc etc.)
    return [f for f in files if f["name"].endswith(".parquet")]
    
def get_silver_file_update(feature,service,snapshot_date, new_inference_df,spark): #feature example 'job_description'
    parent_root = '1_eMgnRaFtt-ZSZD3zfwai3qlpYJ-M5C6'       
    jd_path = ['datamart','silver',  'online', feature]
    folder_id = get_folder_id_by_path(service, jd_path, parent_root) 
    selected_date = str(snapshot_date.year) + "-" + str(snapshot_date.month) 
    filename    = selected_date + ".parquet"
    file = get_file_by_name(service, folder_id, filename)
    print(filename)
    parquet_file = list_parquet_files_in_folder(service, file["id"])[0] if file else None
    with tempfile.TemporaryDirectory() as tmpdir:
        if parquet_file:
            local_file_path = os.path.join(tmpdir, parquet_file["name"])
            download_file(service, parquet_file, output_base=tmpdir)
            print(f"Downloaded file to: {local_file_path}")
            existing_df = spark.read.parquet(local_file_path)
            combined_df = existing_df.unionByName(new_inference_df)
            combined_df.coalesce(1).write.mode("overwrite").parquet(local_file_path)
        else:
            combined_df = new_inference_df
        
    
    output_path = os.path.join("datamart","silver","online", feature, filename)
    combined_df.write.mode("overwrite").parquet(output_path)
    
    upload_file_to_drive(service, output_path, folder_id)

    print(f"Overwriting combined df has {combined_df.count()} rows.")
    return combined_df

def get_month_list(start_date, end_date):
    return [f"{d.year}-{d.month}" for d in pd.date_range(start=start_date, end=end_date, freq='MS')]

def get_gold_file_if_exist(service,start_date, end_date,spark): #feature example 'job_description'
    parent_root = '1_eMgnRaFtt-ZSZD3zfwai3qlpYJ-M5C6'
    request_files = [[],[]]
    for i, store in enumerate(["feature_store", "label_store"]):       
        jd_path = ['datamart', 'gold', store]
        folder_id = get_folder_id_by_path(service, jd_path, parent_root) 
        start_date = datetime(start_date.year, start_date.month, 1)
        end_date = datetime(end_date.year, end_date.month, 1)
        month_list = get_month_list(start_date, end_date)
        for selected_date in month_list:
            filename = selected_date + ".parquet"
            print(f"Checking for file: {filename}")
            file = get_file_by_name(service, folder_id, filename)
            # If file exists, download it
            # If file does not exist, return None
            if file:
                parquet_file = list_parquet_files_in_folder(service, file["id"])[0]
                local_file_path = os.path.join("tmp",parquet_file["name"])
                # print(f"Downloading file: {local_file_path}")
                os.makedirs(os.path.dirname(local_file_path), exist_ok=True)
                local_output_base = os.path.dirname(local_file_path)
                download_file(service, parquet_file, output_base=local_output_base)
                request_files[i].append(local_file_path)
    if request_files[0]:
        feature_df = spark.read.parquet(*request_files[0])
        label_df = spark.read.parquet(*request_files[1])
        return feature_df, label_df
    

# #test gold
# service = connect_to_gdrive()
# start_date = pd.to_datetime("2021-06-01")
# end_date = pd.to_datetime("2021-09-01") 
# requested_df=get_gold_file_if_exist(service,start_date, end_date,spark)
# print(f"requested_df: {requested_df.count()} rows")



#test silver update 

# Step 1: Connect to Google Drive
service = connect_to_gdrive()

# Step 2: Set up test values
feature = "job_description"
snapshot_date = datetime(2020, 1, 1)

# Step 3: Create test DataFrame
data = [
    {"resume_id": "R001", "prediction": 0.85, "snapshot_date": "2020-01"},
    {"resume_id": "R002", "prediction": 0.73, "snapshot_date": "2020-01"}
]
new_inference_df = spark.createDataFrame(pd.DataFrame(data))

# Step 4: Call your function to append/upload
combined_df = get_silver_file_update(feature, service, snapshot_date, new_inference_df,spark)

# Step 5: Read back (if needed, verify locally or re-download)
print(" Update complete — verify contents on Drive.{combined_df}")

