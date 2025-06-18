from pyspark.sql import SparkSession
from datetime import datetime, timedelta
import os
import mlflow
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, FunctionTransformer
from xgboost import XGBClassifier

import mlflow.pytorch
from mlflow.models.signature import infer_signature
from mlflow.models import infer_signature
from mlflow.tracking import MlflowClient
import optuna


from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import io
import boto3

# Connect to the MLflow server (in this case, we are using our own computer)
mlflow.set_tracking_uri(uri="http://localhost:8080")
# Set the tracking experiment 
mlflow.set_experiment("job-fit-classification")
# Enable MLflow system metrics logging
mlflow.enable_system_metrics_logging()  

spark = SparkSession.builder.getOrCreate()

def process_snapshot_data(**kwargs):
    # Get execution date from Airflow
    exec_date = kwargs['execution_date']  # type: datetime.datetime

    # Define 12-month window
    start_date = exec_date
    end_date = start_date + timedelta(days=365)
    return start_date, end_date

def get_files(spark: SparkSession ):
    # Process snapshot data
    start_date, end_date = process_snapshot_data(execution_date=datetime.now())
    
    feature_root_dir = "/path/to/local/parquet_data"
    all_folders_feature = [f.path for f in os.scandir(feature_root_dir) if f.is_dir()]

    label_root_dir = "/path/to/local/parquet_data"
    all_folders_label = [f.path for f in os.scandir(label_root_dir) if f.is_dir()]
 
    # # if u are using s3
    # feature_root_dir = "s3://your-bucket/path/to/parquet_data"#dummy path, replace with actual S3 path
    # label_root_dir = "s3://your-bucket/path/to/parquet_data"#dummy path, replace with actual S3 path
    # all_folders_feature = list_s3_folders("your-bucket", "path/to/parquet_data/")
    # all_folders_label = list_s3_folders("your-bucket", "path/to/parquet_data/")
    
    valid_paths_feature = []
    valid_paths_label = []
    
    folders = [all_folders_feature, all_folders_label]
    valid_paths = [valid_paths_feature,valid_paths_label]
    dfs =[]
    
    for i in range(len(folders)):
        folder = folders[i]
        for file in folder:
            # Extract snapshot_date from folder path
            try:
                file_date_str = file.split("snapshot_date=")[-1] # assume that the file are named as follow: snapshot_date=2023-01-01/
                file_date = datetime.strptime(file_date_str, "%Y-%m-%d")
                if start_date <= file_date <= end_date:
                    valid_paths[i].append(file)
            except Exception:
                continue  # skip folders without valid format

        # Load only those folders into Spark
        df = spark.read.parquet(*valid_paths[i])
        dfs.append(df)
    return dfs

def register_model_mlflow(run_name, params, model, X_train, X_test, y_train, y_test, model_name): # Ensure it's a DataFrame
    # Start an MLflow run
    with mlflow.start_run(run_name=run_name):


        # Instantiate the Ridge model with parameters
        model = model(**params)
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)

        
        acc = accuracy_score(y_test, predictions)
        prec = precision_score(y_test, predictions, average="weighted") # depends on num classes 
        rec = recall_score(y_test, predictions, average="weighted")
        f1 = f1_score(y_test, predictions, average="weighted")


        # log table
        X_test_pred = X_test.reset_index(drop=True).copy()
        X_test_pred["ground_truth"] = y_test.reset_index(drop=True)
        X_test_pred["predictions"] = predictions

        mlflow.log_table(data=X_test_pred, artifact_file="val.csv")

        metric_eval = {"acc": acc, "prec": prec, "rec": rec, "f1": f1}

        # Log the hyperparameters
        mlflow.log_params(params)

        # Log the loss metric
        mlflow.log_metrics(metric_eval)

        # Set a tag that we can use to remind ourselves what this run was for
        mlflow.set_tag("Training Info", f"Basic {model_name} model for job fit classification")

        # Infer the model signature
        signature = infer_signature(X_train, model.predict(X_train))
        input_example = X_train[:1]

        # Log the model
        model_info = mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="job-fit-classification-model",
            signature=signature,
            input_example=input_example,
            #registered_model_name="tracking-quickstart",
        )

        # Plot correlation matrix
        corr = X_train.corr()
        plt.figure(figsize=(10, 8))
        sns.heatmap(corr, annot=True, cmap='coolwarm')
        plt.title("Correlation Matrix")
        plt.tight_layout()

        buf = io.BytesIO()
        plt.savefig(buf, format="png")
        buf.seek(0)
        mlflow.log_image(buf, "correlation_matrix.png")
        plt.close()

        # Log the dataset
        #mlflow.log_dataset(X_train, "training_data")
        #mlflow.log_dataset(X_test, "test_data")

        dataset_train = mlflow.data.from_pandas(X_train, "training_data")
        mlflow.log_input(dataset_train, context="training")

        dataset_test = mlflow.data.from_pandas(X_test_pred, "test_data",predictions="predictions")
        mlflow.log_input(dataset_test, context="test")
        mlflow.set_tag("model_type", "LogisticRegression")  

        return model, f1
    


def objective_lr(trial, X_train, X_test, y_train, y_test):
    params = {
        "C": trial.suggest_float("C", 1e-4, 1e2, log=True),
        "solver": trial.suggest_categorical("solver", ["lbfgs", "liblinear"]),
        "max_iter": trial.suggest_int("max_iter", 100, 1000),
    }
    _, f1 = register_model_mlflow(f"lr_trial_{trial.number}", params, LogisticRegression, X_train, X_test, y_train, y_test, "LogisticRegression")
    return f1


def run_optuna_lr( X_train, X_test, y_train, y_test):
    def objective(trial): return objective_lr(trial, X_train, X_test, y_train, y_test)
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=20)

if __name__ == "__main__":
    dfs = get_files(spark)
    X = dfs[0]
    y = dfs[1].iloc[:, 0]  # or adjust column based on your schema
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    run_optuna_lr(X_train, X_test, y_train, y_test)
