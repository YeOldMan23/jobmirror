import evidently
from evidently import Report
from evidently.presets import DataDriftPreset

import pandas as pd
import argparse
import datetime
from dateutil.relativedelta import relativedelta
import os 
from pyspark.sql import SparkSession, DataFrame
from pyspark.ml.feature import VectorAssembler
import mlflow
import datetime
from sklearn.metrics import roc_auc_score
from datetime import datetime

import mlflow
from mlflow.tracking import MlflowClient

import logging

def detect_data_drift_without_labels(training_df, prod_df,model_drift_date, run_id,model_type, psi_threshold: float = 0.2) :
    report = Report([DataDriftPreset(method="psi")])
    train_pd = training_df.toPandas()
    prod_pd = prod_df.toPandas()
    text_cols = train_pd.select_dtypes(include=['object', 'string']).columns.tolist()
    print(f"text_cols:{text_cols}")
    print(f"all_cols:{prod_pd.columns}")
    train_pd = train_pd.drop(columns=text_cols)
    prod_pd = prod_pd.drop(columns=text_cols)
    my_eval = report.run(reference_data=train_pd, current_data=prod_pd)

    report_path = f"/opt/airflow/reports/{model_type}/{model_drift_date}.html"
    report_dir = os.path.dirname(report_path)
    os.makedirs(report_dir, exist_ok=True)
    my_eval.save_html("data_drift_report.html")

    with mlflow.start_run(run_id=run_id):
        mlflow.log_artifact(report_path)

    drift_json = my_eval.dict()

    drifted_features = []
    for metric in drift_json["metrics"]:
        metric_val = metric["value"]
        feature = metric["metric_id"]
        if feature.startswith("ValueDrift(column="):
            # Extract the column name
            start = feature.find("column=") + len("column=")
            end = feature.find(",", start) if "," in feature[start:] else feature.find(")", start)
            column_name = feature[start:end]
        if isinstance(metric_val, (float, int)) and metric_val > psi_threshold:
            drifted_features.append(column_name)

    if drifted_features:
        print(f" Data drift detected in: {drifted_features}")
        return True
    else:
        print("No significant data drift detected.")
        return False


# ------------------------
#  Model Performance Drift Check (labels + predictions required)
# ------------------------
def detect_model_performance_performance(pred_df, label_df,run_id):
     # Calculate accuracy
    label_pd = label_df.toPandas()
    pred_pd = pred_df.toPandas()
    # Truncate to the shorter length
    min_len = min(len(label_pd), len(pred_pd))

    # Reset index and truncate both
    label_pd_cut = label_pd.reset_index(drop=True).iloc[:min_len]
    pred_pd_cut = pred_pd.reset_index(drop=True).iloc[:min_len]

    # Merge by row index
    merged_pd = pd.concat([label_pd_cut, pred_pd_cut], axis=1)
    # Calculate accuracy
    accuracy = roc_auc_score(merged_pd['label'],merged_pd['model_predictions'])

    # Convert to datetime object
    dt = datetime.strptime(snapshotdate, "%Y-%m-%d")
    # Subtract 6 months
    dt_minus_6_months = dt - relativedelta(months=6)
    # Convert back to string
    result_str = dt_minus_6_months.strftime("%Y-%m-%d")

    
    with mlflow.start_run(run_id=run_id):
        mlflow.log_metric(f"auc_{result_str}", accuracy)

    # Convert probabilities to binary predictions (threshold=0.5)
    # pred_df['predicted_label'] = (pred_df['prediction'] >= 0.5).astype(int)
    # accuracy = (label_df['label'] == pred_df['predicted_label']).mean()
    print(" Model performance report generated.")
    # You can extend this function to return performance metric diffs if desired
    if accuracy<0.6:
        return True

def get_shadow_model_info():
    client =MlflowClient("http://mlflow:5000")

    # Resolve alias "shadow" to get the model version
    shadow_version = client.get_model_version_by_alias("job-fit-classification", "shadow")

    run_id = shadow_version.run_id
    run = client.get_run(run_id)
    run_data = client.get_run(run_id).data
    model_type = run_data.params["model_type"]  # best_thresh

    model_uri = f"models:/job-fit-classification@shadow"

    if model_type == "XGBoostClassifier":
        model = mlflow.xgboost.load_model(model_uri)
    else:
        model = mlflow.sklearn.load_model(model_uri)

    snapshot_date = run.data.params.get("snapshot_date", "Unknown")[:10]
    # # Get the training date from model creation timestamp
    # creation_time = datetime.datetime.fromtimestamp(shadow_version.creation_timestamp / 1000.0)
    # train_date = creation_time.strftime("%Y-%m-%d")
    # print("Registered on:", train_date)

    print("Shadow Model Version:", shadow_version.version)

    return model, snapshot_date, run_id


def model_monitor_shad(snapshotdate,base_path):
    mlflow.set_tracking_uri(uri="http://mlflow:5000")

    spark = SparkSession.builder.getOrCreate()
    poor = False
    
    
    dt = datetime.strptime(snapshotdate, "%Y-%m-%d") - relativedelta(month=1)
    formatted_date = f"{dt.year}-{dt.month}-{dt.day}" 

    base_path = '../datamart/gold/'
    partition_name = f"{formatted_date}.parquet"
    model_label = os.path.join(base_path, "label_store",partition_name)

    model_type = str(get_shadow_model_info()[0])

##################################################################
    if model_type.startswith("XGB"):   # to comapre with the dir what the file name is 
        model_type_file="XGBoostClassi"
    else:
        model_type_file="LogRegClassi"
########################################################################################
    run_id = get_shadow_model_info()[2]
    
    #change to group standard naming convention
    model_prediction = os.path.join(base_path,"prediction_store",model_type_file,partition_name)
    model_prediction = spark.read.parquet(model_prediction)
    model_label = spark.read.parquet(model_label)
    poor = detect_model_performance_performance(model_prediction,model_label,run_id)
    
    
    train_date = get_shadow_model_info()[1]
    train_date = datetime.strptime(train_date, '%Y-%m-%d')
    train_date = f"{train_date.year}-{train_date.month}-{train_date.day}"
    train_filename =  f"{train_date}.parquet"
    train_df_model = os.path.join(base_path,"feature_store", train_filename)

    current_filename =  f"{formatted_date}.parquet"
    current_df_model = os.path.join(base_path,"feature_store", current_filename)
    train_df = spark.read.parquet(train_df_model)
    current_df = spark.read.parquet(current_df_model)
    poor = detect_data_drift_without_labels(train_df,current_df,formatted_date, run_id,model_type)
    print(f"poor:{poor}")
    spark.stop()
    if poor is True:
        print("True")
        return poor
    else:
        print("False")
        return "False"




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="run job")
    parser.add_argument("--snapshotdate", type=str, required=True, help="YYYY-MM-DD")    
    args = parser.parse_args()
    snapshotdate = args.snapshotdate
    model_monitor_shad(snapshotdate)