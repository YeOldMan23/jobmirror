import mlflow
from mlflow.tracking import MlflowClient
from pyspark.sql import SparkSession

from utils.s3_utils import upload_to_s3, read_parquet_from_s3

def load_champion_model(model_name="job-fit-classifier"):
    mlflow.set_tracking_uri("http://localhost:8080")
    client = MlflowClient()
    for mv in client.search_model_versions(f"name='{model_name}'"):
        if 'champion' in dict(mv).get('aliases', []):
            return mlflow.sklearn.load_model(mv.source)
    raise ValueError("Champion model not found.")

def read_gold_feature(snapshot_date : datetime, spark : SparkSession):
    selected_date = str(snapshot_date.year) + "-" + str(snapshot_date.month)
    table_dir     = "datamart", "gold", table_name, f"{selected_date}.parquet"
    return read_parquet_from_s3(table_dir)

# Example usage
model = load_champion_model()

##### Load Data #####
df = read_gold_feature(snapshot_date, spark)

df = df.drop(columns="fit")
