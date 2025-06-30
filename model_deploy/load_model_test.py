from load_model import load_champion_model, load_model_shadow
import mlflow 
from mlflow.models.signature import infer_signature
from mlflow.models import infer_signature
import optuna

from typing import List
from pyspark.sql import SparkSession, DataFrame
from pyspark.ml.feature import VectorAssembler
from pyspark.sql.functions import when, col, udf, monotonically_increasing_id
from pyspark.sql.types import DoubleType

import os

import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

if __name__ == "__main__":
    # Setup argparse to parse command-line arguments
    # parser = argparse.ArgumentParser(description="run job")
    # parser.add_argument("--snapshotdate", type=str, required=True, help="YYYY-MM-DD")
    
    # args = parser.parse_args()
    
    ## Call main with arguments explicitly passed
    # main(args.snapshotdate, args.modelname)
    spark = SparkSession.builder.getOrCreate()
    df_file_path = '../static/2021_6.parquet'
    input_df = spark.read.parquet(df_file_path)

    # Extract out the date and the customer ID
    date_id = input_df.select("resume_id", "job_id")

    feature_cols = [  # list your columns here exactly as used during training
    "soft_skills_mean_score", "soft_skills_max_score", "soft_skills_count",
    "soft_skills_ratio", "hard_skills_general_count", "hard_skills_general_ratio",
    "hard_skills_specific_count", "hard_skills_specific_ratio",
    "work_authorization_match", "employment_type_match", "location_preference_match",
    "relevant_yoe", "total_yoe", "avg_exp_sim", "max_exp_sim", "is_freshie"
    ]

    assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")
    input_df = assembler.transform(input_df)
    logger.info(f"input:{input_df.limit(1)}")
    input_df = input_df.limit(1)
    champ_model, champ_threshold = load_champion_model()
    shadow_model, shadow_threshold = load_model_shadow()

    # Run inference
    champ_output = champ_model.transform(input_df)
    shadow_output = shadow_model.transform(input_df)

    # Run probability threshold 
    get_prob_1 = udf(lambda v: float(v[1]), DoubleType())

    champ_output = champ_output.withColumn("prob_1", get_prob_1(col("probability")))
    champ_output = champ_output.withColumn(
        "prediction_thresh",
        when(col("prob_1") >= champ_threshold, 1).otherwise(0).cast(DoubleType())
    )

    shadow_output = shadow_output.withColumn("prob_1", get_prob_1(col("probability")))
    shadow_output = shadow_output.withColumn(
        "prediction_thresh",
        when(col("prob_1") >= shadow_threshold, 1).otherwise(0).cast(DoubleType())
    )

    # Concat the date and resume id to the side of both outputs
    shadow_output = shadow_output.withColumn("row_id", monotonically_increasing_id())
    champ_output = champ_output.withColumn("row_id", monotonically_increasing_id())
    date_id = date_id.withColumn("row_id", monotonically_increasing_id())

    # Concat here
    shadow_output = shadow_output.join(date_id, on="row_id").drop("row_id")
    champ_output = champ_output.join(date_id, on="row_id").drop("row_id")

    datamart_dir = os.path.join("opt/airflow/datamart/")
    prediction_store_dir = os.path.join(datamart_dir, "prediction_store")

    if not os.path.exists(prediction_store_dir):
        os.mkdir(prediction_store_dir)

    # Save the predictions as a parquet file
    ## Get the date from the parquet file
    champ_file_dir = os.path.join(prediction_store_dir, "champ_predictions.parquet")
    shadow_file_dir = os.path.join(prediction_store_dir, "shadow_predictions.parquet")

    champ_output.write.mode("overwrite").parquet(champ_file_dir)
    shadow_output.write.mode("overwrite").parquet(shadow_file_dir)

    logger.info(champ_output)
    logger.info(shadow_output)