import mlflow
from mlflow.tracking import MlflowClient

def load_champion_model(model_name="job-fit-classifier"):
    mlflow.set_tracking_uri("http://localhost:8080")
    client = MlflowClient()
    for mv in client.search_model_versions(f"name='{model_name}'"):
        if 'champion' in dict(mv).get('aliases', []):
            return mlflow.sklearn.load_model(mv.source)
    raise ValueError("Champion model not found.")

def load_shadow_model(model_name="job-fit-classifier"):
    mlflow.set_tracking_uri("http://localhost:8080")
    client = MlflowClient()
    for mv in client.search_model_versions(f"name='{model_name}'"):
        if 'shadow' in dict(mv).get('aliases', []):
            return mlflow.sklearn.load_model(mv.source)
    raise ValueError("Shadow model not found.")
# Example usage
if __name__ == "__main__":
    model = load_champion_model()
    shadow_model = load_shadow_model()