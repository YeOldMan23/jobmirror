import mlflow
from mlflow.tracking import MlflowClient

mlflow.set_tracking_uri("http://localhost:8080")
MODEL_NAME = 'job-fit-classifier'

client = MlflowClient()
model_deploy = None

for mv in client.search_model_versions(f"name='{MODEL_NAME}'"):
    if 'champion' in dict(mv).get('aliases', []):
        model_deploy = mv
        break

model = mlflow.sklearn.load_model(model_deploy.source)
