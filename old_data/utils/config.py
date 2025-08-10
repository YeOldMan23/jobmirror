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

class PathConfig:
    """Configuration for data paths"""
    
    BASE_PATHS = {
        'training': {
            'silver': {
                'combined_resume_jd': '/datamart/silver/combined_resume_jd',
                'resumes': '/datamart/silver/resumes',
                'job_descriptions': '/datamart/silver/job_description',
                'labels': '/datamart/silver/silver/labels'
            },
            'gold': {
                'feature': '/datamart/gold/feature_store',
                'label': '/datamart/gold/label_store'
            }},
        'inference': {
            'silver': {
                'combined_resume_jd': '/datamart/silver/online/combined_resume_jd',
                'resumes': '/datamart/silver/online/resumes',
                'job_descriptions': '/datamart/silver/online/job_description',
                'labels': '/datamart/silver/silver/online/labels'
            },
            'gold': {
                'label': '/datamart/gold/online/label_store',
                'feature': '/datamart/gold/online/feature_store'
                    }
                 }
    }
    
    MONGODB_COLLECTIONS = {
        'training': {
            'resumes': 'bronze_resumes',
            'job_descriptions': 'bronze_job_descriptions',
            'labels': 'bronze_labels'
        },
        'inference': {
            'resumes': 'online_bronze_resumes',
            'job_descriptions': 'online_bronze_job_descriptions',
            'labels': 'online_bronze_labels'
        }
    }