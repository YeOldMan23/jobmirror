# test_aws.py
from config import AWSConfig
import boto3

def test_aws_connection():
    try:
        # Load your config
        aws_config = AWSConfig()
        print(f"Testing connection to bucket: {aws_config.bucket_name}")
        print(f"Region: {aws_config.region}")
        
        # Create S3 client
        s3 = boto3.client(
            's3',
            aws_access_key_id=aws_config.access_key,
            aws_secret_access_key=aws_config.secret_key,
            region_name=aws_config.region
        )
        
        # Test connection by checking if bucket exists
        s3.head_bucket(Bucket=aws_config.bucket_name)
        print("AWS connection successful!")
        print("Bucket is accessible!")
        
        # Optional: List a few objects to confirm read access
        response = s3.list_objects_v2(Bucket=aws_config.bucket_name, MaxKeys=3)
        if 'Contents' in response:
            print(f"Found {len(response['Contents'])} objects in bucket")
        else:
            print("Bucket is empty but accessible")
            
        return True
        
    except Exception as e:
        print(f"AWS connection failed: {e}")
        return False

if __name__ == "__main__":
    test_aws_connection()