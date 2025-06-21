# Get Spark Session
import pyspark
from pyspark.sql.dataframe import DataFrame
from pyspark.sql import SparkSession
from pyspark.sql.functions import when, col
from pyspark.sql.types import FloatType, StructType, StructField, StringType, ArrayType, BooleanType

# Using another language model to read the values
from sentence_transformers import SentenceTransformer, util

import os

import torch
from datetime import datetime
import re

def process_labels(df):
    """
    Process the label data
    """
    df = df.withColumn(
        "fit_label",
        when(col("fit").isin("Fit"), 1.0).otherwise(0.0)
    )

    return df