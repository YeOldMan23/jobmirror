from mlflow.tracking import MlflowClient
import mlflow

# Connect to the MLflow server
mlflow.set_tracking_uri(uri="http://mlflow:5000")
    
def promote_best_model():
    client = MlflowClient("http://mlflow:5000")
    experiment = client.get_experiment_by_name("job-fit-classification")
    runs = client.search_runs(experiment_ids=[experiment.experiment_id], order_by=["metrics.f1 DESC"])

    best_run = runs[0]
    run_id = best_run.info.run_id
    model_uri = f"runs:/{run_id}/job-fit-classification-model"

    # Register model, only a model better will be registered 
    result = mlflow.register_model(model_uri, "job-fit-classifier")

    # Set alias "champion"
    client.set_registered_model_alias("job-fit-classifier", "champion", version=result.version)  #replaces the alias

    # Return model type (from run tags)
    return best_run.data.tags.get("model_type")

def promote_shadow_model(curr_best_model_type):
    client = MlflowClient("http://mlflow:5000")
    experiment = client.get_experiment_by_name("job-fit-classification")
    runs = client.search_runs(experiment_ids=[experiment.experiment_id], order_by=["metrics.f1 DESC"])

    for run in runs:
        print(f"current model:{run.data.tags.get("model_type")}")
        print(f"curr_best_model_type:{curr_best_model_type}")
        if run.data.tags.get("model_type") != curr_best_model_type and run.data.tags.get("model_type") is not None:
            run_id = run.info.run_id
            model_uri = f"runs:/{run_id}/job-fit-classification-model"
            result = mlflow.register_model(model_uri, "job-fit-classifier")
            client.set_registered_model_alias("job-fit-classifier", "shadow", version=result.version)
            print(f"Shadow model promoted: version {result.version}")
            break

if __name__ == "__main__":
    curr_best_model_type = promote_best_model()
    promote_shadow_model(curr_best_model_type)


