from load_model import load_champion_model
import mlflow 
from mlflow.models.signature import infer_signature
from mlflow.models import infer_signature
import optuna

from typing import List
from pyspark.sql import SparkSession, DataFrame
from pyspark.ml.feature import VectorAssembler

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
    model = load_champion_model()
    output = model.transform(input_df)

    logger.info(output)