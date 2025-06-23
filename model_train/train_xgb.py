from pyspark.sql import SparkSession
from datetime import datetime, timedelta
import os
import sys
import shutil

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, FunctionTransformer
from pyspark.ml.classification import GBTClassifier  # or LogisticRegression, etc.
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from xgboost import XGBClassifier
import boto3
import uuid

import mlflow # only available in MLflow ≥2.4
import mlflow.pytorch
from mlflow.models.signature import infer_signature
from mlflow.models import infer_signature
from mlflow.tracking import MlflowClient
import optuna


from PIL import Image
from typing import List, Tuple, Optional
import logging
from pyspark.sql import SparkSession, DataFrame
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import io

from pyspark.sql.functions import col
from pyspark.sql.types import ArrayType, StringType, FloatType
from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer, CountVectorizer, VectorAssembler
from pyspark.ml.linalg import Vectors, VectorUDT
from pyspark.sql.functions import udf
from pyspark.ml.linalg import DenseVector

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.config import AWSConfig
from utils.s3_utils import get_s3_client, list_s3_folders

# Connect to the MLflow server (in this case, we are using our own computer)
mlflow.set_tracking_uri(uri="http://localhost:8081")
# Set the tracking experiment 
experiment_name = "job-fit-classification-v2"
artifact_path = "file:///C:/Users/shuji/OneDrive/Desktop/School/Machine learning engineering/Project/model_train/mlartifacts"

# Check if experiment exists
client = MlflowClient()
experiment = client.get_experiment_by_name(experiment_name)

if experiment is None:
    experiment_id = client.create_experiment(
        name=experiment_name,
        artifact_location=artifact_path
    )
    print(f"✅ Created new experiment: {experiment_name}")
else:
    experiment_id = experiment.experiment_id
    print(f"ℹ️ Experiment already exists: {experiment_name} (ID: {experiment_id})")

mlflow.set_experiment(experiment_name)
# Enable MLflow system metrics logging
mlflow.enable_system_metrics_logging()  

experiment = mlflow.get_experiment_by_name("job-fit-classification-v2")
print("Artifact location:", experiment.artifact_location)

os.environ["PYSPARK_PYTHON"] = r"C:\Users\shuji\anaconda3.1\envs\job_fit\python.exe"
os.environ["PYSPARK_DRIVER_PYTHON"] = r"C:\Users\shuji\anaconda3.1\envs\job_fit\python.exe"

spark = SparkSession.builder.getOrCreate()

def process_snapshot_data(**kwargs):
    # Get execution date from Airflow
    exec_date = kwargs['execution_date']  # type: datetime.datetime

    # Define 12-month window
    start_date = exec_date
    end_date = start_date + timedelta(days=365)
    return start_date, end_date

# def list_s3_folders(bucket, prefix):
#     s3_client = get_s3_client()
#     # s3 = boto3.client('s3')
#     paginator = s3_client.get_paginator('list_objects_v2')
#     result = paginator.paginate(Bucket=bucket, Prefix=prefix, Delimiter='/')
#     return [f's3://{bucket}/{cp["Prefix"]}' for page in result for cp in page.get("CommonPrefixes", [])]

# def get_files(spark: SparkSession):

#     config = AWSConfig()
#     s3_client = get_s3_client()
#     # Process snapshot data
#     start_date, end_date = process_snapshot_data(execution_date=datetime.now())

#     feature_root_dir = f's3://{config.bucket_name}/features/parquet_data'#dummy path, replace with actual S3 path
#     label_root_dir = f's3://{config.bucket_name}/labels/parquet_data'#dummy path, replace with actual S3 path

#     all_folders_feature = list_s3_folders(config.bucket_name, "features/parquet_data/")
#     all_folders_label = list_s3_folders(config.bucket_name, "labels/parquet_data/")
    
#     valid_paths_feature = []
#     valid_paths_label = []
    
#     folders = [all_folders_feature, all_folders_label]
#     valid_paths = [valid_paths_feature,valid_paths_label]
#     dfs =[]
    
#     for i in range(len(folders)):
#         folder = folders[i]
#         for file in folder:
#             # Extract snapshot_date from folder path
#             try:
#                 file_date_str = file.split("snapshot_date=")[-1] # assume that the file are named as follow: snapshot_date=2023-01-01/
#                 file_date = datetime.strptime(file_date_str, "%Y-%m-%d")
#                 if start_date <= file_date <= end_date:
#                     valid_paths[i].append(file)
#             except Exception:
#                 continue  # skip folders without valid format

#         # Load only those folders into Spark
#         df = spark.read.parquet(*valid_paths[i])
#         dfs.append(df)
#     return dfs


# def get_files(spark: SparkSession, **kwargs) -> List[Optional[DataFrame]]:
#     """
#     Get feature and label files from S3 within date range and load as Spark DataFrames
#     """
#     config = AWSConfig()
#     s3_client = get_s3_client()  # Get once and reuse
    
#     # Process snapshot data
#     start_date, end_date = process_snapshot_data(**kwargs)
    
#     # S3 paths
#     feature_prefix = "features/parquet_data/"
#     label_prefix = "labels/parquet_data/"
    
#     # Get all folders using s3_utils function
#     all_folders_feature = list_s3_folders(config.bucket_name, feature_prefix, s3_client)
#     all_folders_label = list_s3_folders(config.bucket_name, label_prefix, s3_client)
    
#     def filter_folders_by_date(folders: List[str], start_date: datetime, end_date: datetime) -> List[str]:
#         """Filter folders based on snapshot_date within date range"""
#         valid_paths = []
        
#         for folder in folders:
#             try:
#                 if "snapshot_date=" in folder:
#                     date_part = folder.rstrip('/').split("snapshot_date=")[-1]
#                     file_date = datetime.strptime(date_part, "%Y-%m-%d")
                    
#                     if start_date <= file_date <= end_date:
#                         valid_paths.append(folder)
                        
#             except (ValueError, IndexError) as e:
#                 logging.warning(f"Skipping folder with invalid date format: {folder} - {str(e)}")
#                 continue
        
#         return valid_paths
    
#     # Filter folders by date range
#     valid_paths_feature = filter_folders_by_date(all_folders_feature, start_date, end_date)
#     valid_paths_label = filter_folders_by_date(all_folders_label, start_date, end_date)
    
#     logging.info(f"Found {len(valid_paths_feature)} valid feature folders")
#     logging.info(f"Found {len(valid_paths_label)} valid label folders")
    
#     dfs = []
    
#     # Load feature data
#     if valid_paths_feature:
#         try:
#             logging.info("Loading feature data...")
#             df_features = spark.read.parquet(*valid_paths_feature)
#             dfs.append(df_features)
#             logging.info(f"Successfully loaded feature data from {len(valid_paths_feature)} folders")
#         except Exception as e:
#             logging.error(f"Error loading feature data: {str(e)}")
#             dfs.append(None)
#     else:
#         logging.warning("No valid feature folders found in date range")
#         dfs.append(None)
    
#     # Load label data
#     if valid_paths_label:
#         try:
#             logging.info("Loading label data...")
#             df_labels = spark.read.parquet(*valid_paths_label)
#             dfs.append(df_labels)
#             logging.info(f"Successfully loaded label data from {len(valid_paths_label)} folders")
#         except Exception as e:
#             logging.error(f"Error loading label data: {str(e)}")
#             dfs.append(None)
#     else:
#         logging.warning("No valid label folders found in date range")
#         dfs.append(None)
    
#     return dfs


#using dummy files for training 
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

def convert_dense_vector_to_list(row):
    row_dict = row.asDict()
    if isinstance(row_dict.get("features"), DenseVector):
        row_dict["features"] = row_dict["features"].toArray().tolist()
    return row_dict


# def get_files(spark: SparkSession ):

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
        # sample_input = test_df.select(feature_col).limit(5).toPandas()
        # input_example = convert_dense_vector_to_list(sample_input)

        # Also run model.transform to get predictions
        # sample_output = model.transform(test_df).select("prediction").limit(5).toPandas()

        # Infer signature from sample
        # signature = infer_signature(sample_input, sample_output)
        signature = infer_signature(test_df.select("features"), predictions.select("prediction"))
        
        # Log the model
        mlflow.spark.log_model(
            spark_model=model,
            artifact_path="job-fit-classification-model",
            signature=signature,
            # input_example=input_example
        )



        # # Log the dataset
        # mlflow.log_dataset(X_train, "training_data")
        # mlflow.log_dataset(X_test, "test_data")

        train_path = f"/tmp/train_spark_{uuid.uuid4()}.parquet"
        train_df.select(feature_col).write.mode("overwrite").parquet(train_path)
        mlflow.log_artifact(train_path, artifact_path="training_data")

        # Save test data with predictions
        test_path = f"/tmp/test_spark_{uuid.uuid4()}.parquet"
        predictions.select(feature_col + ["fit_label", "prediction"]).write.mode("overwrite").parquet(test_path)
        mlflow.log_artifact(test_path, artifact_path="test_data")

        mlflow.set_tag("model_type", "GBTClassifier")
        return model, f1
    

def run_optuna_xgb( train_df, test_df,feature_col):
    def objective(trial):
        params = {
            # "n_estimators": trial.suggest_int("n_estimators", 50, 1000),
            "stepSize": trial.suggest_float("stepSize", 0.01, 0.3),
            "maxDepth": trial.suggest_int("maxDepth", 3, 10),
            "maxBins": 500,
            "lossType":'logistic'
        }
        _, f1 = register_model_mlflow(f"xgb_trial_{trial.number}", params, GBTClassifier, train_df, test_df, "GBTClassifier",feature_col)
        return f1

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=20)
    return study


    

if __name__ == "__main__":
    # dfs = get_files(spark)
    df = get_files(spark,"../static/2021_6.parquet","../static/2021_6_labels.parquet")
    feature_col = ['hard_skills_general_ratio', 'soft_skills_ratio', 'location_preference_match','employment_type_match','work_authorization_match',
    'relevant_yoe','avg_exp_sim','is_freshie']
    # feature_col = ['resume_id', 'job_id', 'snapshot_date', 'soft_skills_mean_score', 'soft_skills_max_score', 'soft_skills_count', 'soft_skills_ratio', 'hard_skills_general_count', 'hard_skills_general_ratio', 'hard_skills_specific_count', 'hard_skills_specific_ratio', 'work_authorization_match', 'employment_type_match', 'location_preference_match', 'relevant_yoe', 'total_yoe', 'avg_exp_sim', 'max_exp_sim', 'is_freshie']
    # X = df[0].select(*feature_col)
    # y = df[1].select("fit")
    # input_df = X.withColumn("fit", y["fit"])
    print(df[0].show(5))
    df[0].printSchema()

    input_df = df[0].join(df[1].select("resume_id", "job_id", "fit_label"), on=["resume_id", "job_id"], how="inner")
    # input_df = input_df.withColumnRenamed("fit", "fit_label")
    input_df = input_df.select(*feature_col, "fit_label")
    input_df = input_df.fillna(0.0, subset=feature_col)

    assembler = VectorAssembler(inputCols=feature_col, outputCol="features")
    df_transformed = assembler.transform(input_df)

    train_df, test_df = df_transformed.randomSplit([0.8, 0.2], seed=42)
    run_optuna_xgb( train_df, test_df, feature_col = "features")
    spark.stop()
    
    
# mlflow server --host 127.0.0.1 --port 8081 --default-artifact-root "file:///C:/Users/shuji/OneDrive/Desktop/School/Machine learning engineering/Project/model_train/mlartifacts"
