# config.py
import os
from dotenv import load_dotenv
from dataclasses import dataclass

# Load .env file
load_dotenv()

@dataclass
class AWSConfig:
    region: str = os.getenv("AWS_REGION")
    bucket_name: str = os.getenv("S3_BUCKET_NAME")
    access_key: str = os.getenv("AWS_ACCESS_KEY_ID")
    secret_key: str = os.getenv("AWS_SECRET_ACCESS_KEY")
    
    def __post_init__(self):
        if not self.bucket_name:
            raise ValueError("S3_BUCKET_NAME not found in environment variables")

@dataclass 
class MlFlowConfig:
    uri: str = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:8080")
    experiment_name: str = os.getenv("MLFLOW_EXPERIMENT_NAME", "default")
    registered_model_name: str = "ml-pipeline-model"
    artifact_path: str = "model"