from pyspark.sql import SparkSession
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
import os
import shutil
import argparse
import pandas as pd

from pyspark.ml.classification import GBTClassifier 
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
import uuid

import mlflow 
from mlflow.models.signature import infer_signature
from mlflow.models import infer_signature
import optuna

from typing import List
from pyspark.sql import SparkSession, DataFrame
from pyspark.ml.feature import VectorAssembler

from utils.gdrive_utils import connect_to_gdrive, get_gold_file_if_exist



# Connect to the MLflow server
mlflow.set_tracking_uri(uri="http://mlflow:5000")
# Set the tracking experiment 
experiment_name = "job-fit-classification"

mlflow.set_experiment(experiment_name)
# Enable MLflow system metrics logging
mlflow.enable_system_metrics_logging()  

experiment = mlflow.get_experiment_by_name("job-fit-classification")
print("Artifact location:", experiment.artifact_location)

spark = SparkSession.builder.getOrCreate()


def process_snapshot_data(**kwargs):
    # Get execution date from Airflow
    exec_date = kwargs['execution_date']  # type: datetime.datetime

    # Define 12-month window
    start_date = exec_date
    end_date = start_date + timedelta(days=365)
    return start_date, end_date


def get_files(spark, feature_file_path, label_file_path) -> List[DataFrame]:
    """
    Get feature and label files from local parquet files for testing
    """
    # Replace with your actual file paths
    feature_file = feature_file_path
    label_file = label_file_path
    
    # Load the feature and label data
    df_features = spark.read.parquet(feature_file)
    df_labels = spark.read.parquet(label_file)
    
    return [df_features, df_labels]


def register_model_mlflow(run_name, params, model, train_df, test_df, model_name,feature_col): # Ensure it's a DataFrame
    # Start an MLflow run
    with mlflow.start_run(run_name=run_name):

        # Instantiate the Ridge model with parameters
        classifier = model(**params, labelCol="fit_label", featuresCol=feature_col)
        model = classifier.fit(train_df)
        predictions = model.transform(test_df).persist()
        _ = predictions.count() # Force model materialization by transforming and counting


        evaluator_acc = MulticlassClassificationEvaluator(labelCol="fit_label", predictionCol="prediction", metricName="accuracy")
        evaluator_prec = MulticlassClassificationEvaluator(labelCol="fit_label", predictionCol="prediction", metricName="weightedPrecision")
        evaluator_rec = MulticlassClassificationEvaluator(labelCol="fit_label", predictionCol="prediction", metricName="weightedRecall")
        evaluator_f1 = MulticlassClassificationEvaluator(labelCol="fit_label", predictionCol="prediction", metricName="f1")

        
        acc = evaluator_acc.evaluate(predictions)
        prec = evaluator_prec.evaluate(predictions)
        rec = evaluator_rec.evaluate(predictions)
        f1 = evaluator_f1.evaluate(predictions)


        metric_eval = {"acc": acc, "prec": prec, "rec": rec, "f1": f1}

        # Log the hyperparameters
        mlflow.log_params(params)

        # Log the loss metric
        mlflow.log_metrics(metric_eval)
        
         # log table
        log_dir = f"/tmp/spark_predictions_log_{uuid.uuid4()}"
        if os.path.exists(log_dir):
            shutil.rmtree(log_dir)

        predictions.select("fit_label", "prediction") \
            .withColumnRenamed("fit_label", "ground_truth") \
            .withColumnRenamed("prediction", "predictions") \
            .coalesce(1) \
            .write \
            .option("header", True) \
            .mode("overwrite") \
            .csv(log_dir)

        # Log the .csv file from the output directory
        csv_file = next((f for f in os.listdir(log_dir) if f.endswith(".csv")), None)
        if csv_file:
            mlflow.log_artifact(os.path.join(log_dir, csv_file), artifact_path="val")

        # Set a tag that we can use to remind ourselves what this run was for
        mlflow.set_tag("Training Info", f"Basic {model_name} model for job fit classification")

        # Infer the model signature
        signature = infer_signature(test_df.select(feature_col), predictions.select("prediction"))
        
        # Log the model
        mlflow.spark.log_model(
            spark_model=model,
            artifact_path="job-fit-classification-model",
            signature=signature,
        )

        # # Log the dataset
        # mlflow.log_dataset(X_train, "training_data")
        # mlflow.log_dataset(X_test, "test_data")

        train_path = f"/tmp/train_spark_{uuid.uuid4()}.parquet"
        train_df.select(feature_col).write.mode("overwrite").parquet(train_path)
        mlflow.log_artifact(train_path, artifact_path="training_data")

        # Save test data with predictions
        test_path = f"/tmp/test_spark_{uuid.uuid4()}.parquet"
        predictions.select(["fit_label", "prediction", feature_col]).write.mode("overwrite").parquet(test_path)
        mlflow.log_artifact(test_path, artifact_path="test_data")

        mlflow.set_tag("model_type", "GBTClassifier")
        return model, f1
    

def run_optuna_xgb( train_df, test_df,feature_col,snapshot_date):
    def objective(trial):
        params = {
            "stepSize": trial.suggest_float("stepSize", 0.01, 0.3),
            "maxDepth": trial.suggest_int("maxDepth", 3, 10),
            "maxBins": trial.suggest_int("maxBins", 32, 500),
            "lossType":'logistic'
        }
        _, f1 = register_model_mlflow(f"xgb_{snapshot_date}_trial_{trial.number}", params, GBTClassifier, train_df, test_df, "GBTClassifier",feature_col)
        return f1

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=20)
    return study


    

if __name__ == "__main__":
    # Setup argparse to parse command-line arguments
    parser = argparse.ArgumentParser(description="run job")
    parser.add_argument("--snapshotdate", type=str, required=True, help="YYYY-MM-DD")    
    args = parser.parse_args()

    service = connect_to_gdrive()
    snapshot_date = datetime.strptime(args.snapshotdate, "%Y-%m-%d")  # adjust format if needed
    start_date = snapshot_date - relativedelta(months=2)
    end_date = snapshot_date
    feature_df, input_df =get_gold_file_if_exist(service,start_date, end_date,spark)
    print(f"requested_df: {feature_df.count()} rows")

    feature_col = ['hard_skills_general_ratio', 'soft_skills_ratio', 'location_preference_match','employment_type_match','work_authorization_match',
    'relevant_yoe','avg_exp_sim','is_freshie']

    input_df = feature_df.join(input_df.select("resume_id", "job_id", "fit_label"), on=["resume_id", "job_id"], how="inner")
    input_df = input_df.select(*feature_col, "fit_label")
    input_df = input_df.fillna(0.0, subset=feature_col)

    assembler = VectorAssembler(inputCols=feature_col, outputCol="features")
    df_transformed = assembler.transform(input_df)

    train_df, test_df = df_transformed.randomSplit([0.8, 0.2], seed=42)
    run_optuna_xgb(train_df, test_df, feature_col = "features", snapshot_date=args.snapshotdate)
    spark.stop()
