import mlflow
from mlflow.tracking import MlflowClient
from pyspark.sql import SparkSession
import logging

import pyspark.ml
from typing import Tuple

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_champion_model(model_name="job-fit-classifier") -> Tuple[pyspark.ml.PipelineModel, float]:
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
    run_id    = latest_mv.run_id
    print(f"Loading model version: {latest_mv.version} from {latest_mv.source}")
    
    # Retrieve Model
    model = mlflow.spark.load_model(latest_mv.source)

    run = client.get_run(str(run_id))
    params = run.data.params
    
    # Error handling
    if "best_thresh" not in params:
        print("Best Threshold not found, using default of 0.5")
        best_thresh = 0.5
    else:
        best_thresh = float(params["best_thresh"])

    return model, best_thresh

def load_model_shadow(model_name="job-fit-classifier") -> Tuple[pyspark.ml.PipelineModel, float]:
    print("Logging start", flush=True)
    mlflow.set_tracking_uri("http://mlflow:5000")
    print("Logging initialized", flush=True)
    client = MlflowClient()
    logger.info("Shadow model loading started")
    model_versions = client.search_model_versions(f"name='{model_name}' and current_stage='shadow'")
    
    if not model_versions:
        raise ValueError(f"No model versions found for model '{model_name}' in stage 'shadow'")
    
    latest_mv = max(model_versions, key=lambda mv: int(mv.version))
    run_id    = latest_mv.run_id

    model = mlflow.spark.load_model(latest_mv.source)

    print(f"Loading model version: {latest_mv.version} from {latest_mv.source}")

    run = client.get_run(str(run_id))
    params = run.data.params
    
    # Error handling
    if "best_thresh" not in params:
        print("Best Threshold not found, using default of 0.5")
        best_thresh = 0.5
    else:
        best_thresh = float(params["best_thresh"])

    return model, best_thresh