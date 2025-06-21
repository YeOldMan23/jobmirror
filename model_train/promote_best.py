from mlflow.tracking import MlflowClient
import mlflow

    
def promote_best_model():
    client = MlflowClient()
    experiment = client.get_experiment_by_name("job-fit-classification")
    runs = client.search_runs(experiment_ids=[experiment.experiment_id], order_by=["metrics.f1 DESC"])

    best_run = runs[0]
    run_id = best_run.info.run_id
    model_uri = f"runs:/{run_id}/model"

    # Register model, only a model better will be registered 
    result = mlflow.register_model(model_uri, "tracking-quickstart")

    # Set alias "champion"
    client.set_registered_model_alias("job-fit-classifier", "champion", version=result.version)

def promote_shadow_model():
    client = MlflowClient()
    experiment = client.get_experiment_by_name("job-fit-classification")
    runs = client.search_runs(experiment_ids=[experiment.experiment_id], order_by=["metrics.f1 DESC"])

    for run in runs:
        if run.data.tags.get("model_type") == "LogisticRegression":
            run_id = run.info.run_id
            model_uri = f"runs:/{run_id}/model"
            result = mlflow.register_model(model_uri, "job-fit-classifier")
            client.set_registered_model_alias("job-fit-classifier", "shadow", version=result.version)
            print(f"Shadow model promoted: version {result.version}")
            break

if __name__ == "__main__":
    promote_best_model()
    promote_shadow_model()


