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
from pyspark.sql.functions import udf, col, when 
from pyspark.sql.types import DoubleType

from utils.gdrive_utils import connect_to_gdrive, get_gold_file_if_exist
import mlflow
import matplotlib.pyplot as plt
import tempfile
import os
from sklearn.metrics import f1_score
import numpy as np


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

def tune_threshold(y_true, y_prob):
    thresholds = np.arange(0.0, 1.01, 0.01)
    best_threshold, best_score = 0.5, 0

    for t in thresholds:
        y_pred = (y_prob >= t).astype(int)
        score = f1_score(y_true, y_pred)
        if score > best_score:
            best_score = score
            best_threshold = t

    return best_threshold, best_score

def register_model_mlflow(run_name, params, model, train_df, test_df, oot_df, model_name,feature_col,snapshot_date): # Ensure it's a DataFrame
    # Start an MLflow run
    with mlflow.start_run(run_name=run_name):

        # Instantiate the Ridge model with parameters
        classifier = model(**params, labelCol="fit_label", featuresCol=feature_col)
        model = classifier.fit(train_df)

        feature = train_df.columns
        # Get feature importances from the trained model
        importances = model.featureImportances.toArray()

        columns_to_remove = ['resume_id', 'job_id', 'snapshot_date','fit_label', 'features']

        # Remove the columns if they exist in feature_col
        feature = [col for col in feature if col not in columns_to_remove]

        print(feature)
        print(importances)

        print(len(feature))
        print(len(importances))

        # Combine with feature names
        feature_importance_df = pd.DataFrame({
            'Feature': feature,
            'Importance': importances
        }).sort_values(by="Importance", ascending=False)

        print(feature_importance_df)
        with tempfile.TemporaryDirectory() as tmpdir:
            plt.figure(figsize=(10, 6))
            plt.barh(feature_importance_df['Feature'], feature_importance_df['Importance'])
            plt.xlabel("Feature Importance")
            plt.ylabel("Feature")
            plt.title(" Feature Importances")
            plt.gca().invert_yaxis()  # highest at top
            plt.tight_layout()
            bar_plot_path = os.path.join(tmpdir, "feature_summary_bar.png")
            # --- Bar plot of mean absolute SHAP values ---
            plt.savefig(bar_plot_path, bbox_inches="tight")
            mlflow.log_artifact(bar_plot_path, artifact_path="feature_importance")

        plt.close()
        predictions = model.transform(test_df).persist()
        _ = predictions.count() # Force model materialization by transforming and counting

        get_prob_1 = udf(lambda v: float(v[1]), DoubleType())
        df_prob = predictions.withColumn("prob_1", get_prob_1(col("probability")))
        pdf = df_prob.select("prob_1", "fit_label").toPandas()
        y_prob = pdf["prob_1"].values
        y_true = pdf["fit_label"].values

        best_thresh, best_f1 = tune_threshold(y_true, y_prob)
        print(f"Best threshold = {best_thresh:.2f} with F1 = {best_f1:.4f}")

        mlflow.log_params({"best_thresh":float(best_thresh)})
        
        if isinstance(snapshot_date, datetime):
            formatted_date = snapshot_date.strftime("%Y %m %d")
        else:
            formatted_date = snapshot_date
        mlflow.log_param("snapshot_date",str(formatted_date))

        df_pred = df_prob.withColumn(
            "custom_prediction",
            when(col("prob_1") >= best_thresh, 1).otherwise(0).cast("double")
        )


        evaluator_acc = MulticlassClassificationEvaluator(labelCol="fit_label", predictionCol="custom_prediction", metricName="accuracy")
        evaluator_prec = MulticlassClassificationEvaluator(labelCol="fit_label", predictionCol="custom_prediction", metricName="weightedPrecision")
        evaluator_rec = MulticlassClassificationEvaluator(labelCol="fit_label", predictionCol="custom_prediction", metricName="weightedRecall")
        evaluator_f1 = MulticlassClassificationEvaluator(labelCol="fit_label", predictionCol="custom_prediction", metricName="f1")

        
        acc = evaluator_acc.evaluate(df_pred)
        prec = evaluator_prec.evaluate(df_pred)
        rec = evaluator_rec.evaluate(df_pred)
        f1 = evaluator_f1.evaluate(df_pred)


        metric_eval = {"acc": acc, "prec": prec, "rec": rec, "f1": f1, }

        # Log the hyperparameters
        mlflow.log_params(params)

        mlflow.log_metrics(metric_eval)
        
         # log table
        log_dir = f"/tmp/spark_predictions_log_{uuid.uuid4()}"
        if os.path.exists(log_dir):
            shutil.rmtree(log_dir)

        df_pred.select("fit_label", "custom_prediction") \
            .withColumnRenamed("fit_label", "ground_truth") \
            .withColumnRenamed("custom_prediction", "predictions") \
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
        signature = infer_signature(test_df.select(feature_col), df_pred.select("custom_prediction"))
        
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
        df_pred.select(["fit_label", "custom_prediction", feature_col]).write.mode("overwrite").parquet(test_path)
        mlflow.log_artifact(test_path, artifact_path="test_data")


        # === OOT EVALUATION ===
        oot_predictions = model.transform(oot_df).persist()
        _ = oot_predictions.count()

        get_prob_1 = udf(lambda v: float(v[1]), DoubleType())
        df_prob_oot = oot_predictions.withColumn("prob_1", get_prob_1(col("probability")))

        df_pred_oot = df_prob_oot.withColumn(
            "custom_prediction",
            when(col("prob_1") >= best_thresh, 1).otherwise(0).cast("double")
        )

        oot_acc = evaluator_acc.evaluate(df_pred_oot)
        oot_prec = evaluator_prec.evaluate(df_pred_oot)
        oot_rec = evaluator_rec.evaluate(df_pred_oot)
        oot_f1 = evaluator_f1.evaluate(df_pred_oot)

        oot_metrics = {
            "oot_acc": oot_acc,
            "oot_prec": oot_prec,
            "oot_rec": oot_rec,
            "oot_f1": oot_f1
        }
        
        mlflow.log_metrics(oot_metrics)

        mlflow.set_tag("model_type", "GBTClassifier")
        return model, f1
    

def run_optuna_xgb( train_df, test_df,oot_df,feature_col,snapshot_date):
    def objective(trial):
        params = {
            "stepSize": trial.suggest_float("stepSize", 0.01, 0.3),
            "maxDepth": trial.suggest_int("maxDepth", 3, 10),
            "maxBins": trial.suggest_int("maxBins", 32, 500),
            "lossType":'logistic'
        }
        _, f1 = register_model_mlflow(f"xgb_{snapshot_date}_trial_{trial.number}", params, GBTClassifier, train_df, test_df,oot_df, "GBTClassifier",feature_col,snapshot_date)
        return f1

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=2)
    return study


    

if __name__ == "__main__":
    # Setup argparse to parse command-line arguments
    parser = argparse.ArgumentParser(description="run job")
    parser.add_argument("--snapshotdate", type=str, required=True, help="YYYY-MM-DD")    
    args = parser.parse_args()

    service = connect_to_gdrive()
    snapshot_date = datetime.strptime(args.snapshotdate, "%Y-%m-%d")  # adjust format if needed
    start_date = snapshot_date - relativedelta(months=3) # use 3 months of training used to be 5
    end_date = snapshot_date
    feature_df, input_df =get_gold_file_if_exist(service,start_date, end_date,spark)
    print(f"requested_df: {feature_df.count()} rows")
    print(f"Getting feature inputs: {feature_df.columns}")

    feature_col = ['soft_skills_mean_score', 'soft_skills_max_score', 'soft_skills_count', 'soft_skills_ratio', 'hard_skills_general_count', 'hard_skills_general_ratio', 'hard_skills_specific_count', 'hard_skills_specific_ratio', 'edu_gpa', 'edu_level_match', 'edu_level_score', 'edu_field_match', 'cert_match', 'work_authorization_match', 'employment_type_match', 'location_preference_match', 'relevant_yoe', 'total_yoe', 'avg_exp_sim', 'max_exp_sim', 'is_freshie']

    input_df = feature_df.join(input_df.select("resume_id", "job_id", "fit_label"), on=["resume_id", "job_id"], how="inner")
    input_df = input_df.select(*feature_col, "fit_label")

    input_df = input_df.withColumn("edu_level_match", col("edu_level_match").cast("int")) \
       .withColumn("edu_field_match", col("edu_field_match").cast("int")) \
       .withColumn("cert_match", col("cert_match").cast("int"))
    
    input_df = input_df.fillna(0.0, subset=feature_col)
    print(f"rows:{input_df.count()}")
    input_df = input_df.dropna()
    print(f"rows:{input_df.count()}")

    assembler = VectorAssembler(inputCols=feature_col, outputCol="features")
    df_transformed = assembler.transform(input_df)

    train_df, test_df, oot_df = df_transformed.randomSplit([0.8, 0.1, 0.1], seed=42) #already shuffling
    run_optuna_xgb(train_df, test_df,oot_df, feature_col = "features", snapshot_date=args.snapshotdate)
    spark.stop()
