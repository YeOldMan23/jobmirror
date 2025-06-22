# utils/s3_utils.py
import boto3
import os
from .config import AWSConfig
from pyspark.sql import SparkSession

def get_s3_client():
    """Create S3 client using environment variables"""
    config = AWSConfig()
    
    return boto3.client(
        's3',
        aws_access_key_id=config.access_key,
        aws_secret_access_key=config.secret_key,
        region_name=config.region
    )

def upload_to_s3(local_path: str, s3_key: str):
    """Upload file or directory to S3 using env config"""
    config = AWSConfig()
    s3_client = get_s3_client()
    
    try:
        # Check if it's a file or directory
        if os.path.isfile(local_path):
            # Single file upload
            s3_client.upload_file(local_path, config.bucket_name, s3_key)
            print(f"Uploaded {local_path} to s3://{config.bucket_name}/{s3_key}")
        
        elif os.path.isdir(local_path):
            # Directory upload (for parquet)
            for root, dirs, files in os.walk(local_path):
                for file in files:
                    local_file = os.path.join(root, file)
                    # Maintain directory structure in S3
                    relative_path = os.path.relpath(local_file, local_path)
                    s3_file_key = os.path.join(s3_key, relative_path).replace("\\", "/")
                    
                    s3_client.upload_file(local_file, config.bucket_name, s3_file_key)
                    print(f"Uploaded {local_file} to s3://{config.bucket_name}/{s3_file_key}")
        else:
            raise ValueError(f"Path {local_path} is neither a file nor directory")
            
    except Exception as e:
        print(f"Error uploading: {e}")
        raise

def read_parquet_from_s3(spark: SparkSession, key: str):
    """Read Parquet directly from S3 using Spark's S3A connector."""
    config = AWSConfig()
    s3_path = f"s3a://{config.bucket_name}/{key}" 
    
    try:
        return spark.read.parquet(s3_path)
    except Exception as e:
        print(f"Error reading Parquet from S3: {e}")
        return None
    
def list_s3_folders(bucket, prefix):
    """List folders in S3 bucket"""
    s3_client = get_s3_client()
    paginator = s3_client.get_paginator('list_objects_v2')
    result = paginator.paginate(Bucket=bucket, Prefix=prefix, Delimiter='/')
    return [f's3://{bucket}/{cp["Prefix"]}' for page in result for cp in page.get("CommonPrefixes", [])]

