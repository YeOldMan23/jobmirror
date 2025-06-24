import mlflow
from mlflow.tracking import MlflowClient
from pyspark.sql import SparkSession
from datetime import datetime

from utils.s3_utils import upload_to_s3, read_parquet_from_s3

def load_champion_model(model_name="job-fit-classifier"):
    mlflow.set_tracking_uri("http://localhost:8080")
    client = MlflowClient()
    for mv in client.search_model_versions(f"name='{model_name}'"):
        if 'champion' in dict(mv).get('aliases', []):
            return mlflow.sklearn.load_model(mv.source), model_name
    raise ValueError("Champion model not found.")

def read_gold_feature(snapshot_date : datetime, spark : SparkSession):
    '''Get data from feature store  '''
    selected_date = str(snapshot_date.year) + "-" + str(snapshot_date.month)
    table_dir     = "datamart", "gold", "online", "feature_store", f"{selected_date}.parquet"
    return read_parquet_from_s3(table_dir)

def main(snapshot_date: datetime, spark: SparkSession):

    #### Load best model ####
    model, model_name = load_champion_model()

    ##### Load Data #####
    df = read_gold_feature(snapshot_date, spark)
    features_df = df.toPandas()

    # drop the label column first
    X_inference = features_df.drop(columns="fit")

    # predict model
    y_inference = model.predict_proba(X_inference)[:, 1]

    # prepare output
    y_inference_pdf = features_df[["resume_id", "job_id","snapshot_date"]].copy()
    y_inference_pdf["model_name"] = model_name
    y_inference_pdf["model_predictions"] = y_inference

    # save gold table - IRL connect to database to write
    gold_directory = f"datamart/gold/model_predictions/{config["model_name"][:-4]}/"  
    partition_name = model_name + "_predictions_" + snapshot_date.replace('-','_') + '.parquet'
    filepath = gold_directory + partition_name
    spark.createDataFrame(y_inference_pdf).write.mode("overwrite").parquet(filepath)
    # df.toPandas().to_parquet(filepath,
    #           compression='gzip')
    print('saved to:', filepath)

if __name__ == "__main__":

    main()
