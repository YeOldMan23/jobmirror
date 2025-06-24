import mlflow
from mlflow.tracking import MlflowClient
from pyspark.sql import SparkSession
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_champion_model(model_name="job-fit-classifier"):
    print("Logging start", flush=True)
    mlflow.set_tracking_uri("http://mlflow:5000")
    print("Logging initialized", flush=True)
    client = MlflowClient()
    logger.info("Champion model loading started")
    model_versions = client.search_model_versions(f"name='{model_name}'")

    if not model_versions:
        raise ValueError(f"No versions found for model '{model_name}'")

    # Sort versions by version number (as int) descending
    latest_mv = max(model_versions, key=lambda mv: int(mv.version))
    print(f"Loading model version: {latest_mv.version} from {latest_mv.source}")

    return mlflow.spark.load_model(latest_mv.source)

# def load_shadow_model(model_name="job-fit-classifier"):
#     mlflow.set_tracking_uri("http://mlflow:5000")
#     client = MlflowClient()
#     for mv in client.search_model_versions(f"name='{model_name}'"):
#         if 'shadow' in dict(mv).get('aliases', []):
#             return mlflow.spark.load_model(mv.source)
#     raise ValueError("Shadow model not found.")
