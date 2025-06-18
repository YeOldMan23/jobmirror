# utils/s3_utils.py
import boto3
from config import AWSConfig

def get_s3_client():
    """Create S3 client using environment variables"""
    config = AWSConfig()
    
    return boto3.client(
        's3',
        aws_access_key_id=config.access_key,
        aws_secret_access_key=config.secret_key,
        region_name=config.region
    )

def upload_to_s3(local_file_path: str, s3_key: str):
    """Upload file to S3 using env config"""
    config = AWSConfig()
    s3_client = get_s3_client()
    
    try:
        s3_client.upload_file(local_file_path, config.bucket_name, s3_key)
        print(f"Uploaded {local_file_path} to s3://{config.bucket_name}/{s3_key}")
    except Exception as e:
        print(f"Error uploading: {e}")