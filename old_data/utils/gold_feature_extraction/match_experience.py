# Get Spark Session
import pyspark
from pyspark.sql.dataframe import DataFrame
from pyspark.sql import SparkSession
from pyspark.sql.functions import udf, col
from pyspark.sql.types import FloatType, StructType, StructField, StringType, ArrayType, BooleanType

# Using another language model to read the values
from sentence_transformers import SentenceTransformer, util

import os

import torch
from datetime import datetime
import re

###################################################
# Gold Table Aggregations for Experience
###################################################

GLOBAL_SIM_THRESHOLD = 0.5

@udf(FloatType())
def get_relevant_yoe(sim_matrix, yoe_list):
    """
    Get the relevant YoE from the array
    """
    relevant_yoe = 0

    for cur_yoe, cur_exp_sim in zip(yoe_list, sim_matrix):
        if cur_exp_sim >= GLOBAL_SIM_THRESHOLD:
            relevant_yoe += abs(cur_yoe)

    return max(0.0, relevant_yoe) if len(yoe_list) > 0 else 0.0

@udf(FloatType())
def get_total_yoe(yoe_list):
    """
    Get the total YoE from the array
    """
    return max(0.0, sum(yoe_list)) if len(yoe_list) > 0.0 else 1.0

@udf(FloatType())
def get_avg_job_sim(sim_matrix):
    """
    Get Average Job Sim
    """
    if len(sim_matrix) > 0:
        return sum(sim_matrix) / len(sim_matrix)
    else:
        return 0
    
@udf(FloatType())
def get_max_job_sim(sim_matrix):
    """
    Get Max Job Sim
    """
    if len(sim_matrix) > 0:
        return max(sim_matrix)
    else:
        return 0
    
@udf(FloatType())
def is_freshie(sim_matrix):
    """
    Boolean to determine if the person is new to the job market
    """
    return 1.0 if len(sim_matrix) == 0 else 0.0

def process_gold_experience(df):
    """
    Process the gold experience data
    """
    df = df.withColumn("relevant_yoe", get_relevant_yoe(df["exp_sim_list"], df['YoE_list']))
    df = df.withColumn("total_yoe", get_total_yoe(df['YoE_list']))
    df = df.withColumn("avg_exp_sim", get_avg_job_sim(df["exp_sim_list"]))
    df = df.withColumn("max_exp_sim", get_max_job_sim(df["exp_sim_list"]))
    df = df.withColumn("is_freshie", is_freshie(df["exp_sim_list"]))

    # TODO Decide wether to drop the combinations with no JD
    df = df.filter(col('role_title').isNotNull())

    return df
