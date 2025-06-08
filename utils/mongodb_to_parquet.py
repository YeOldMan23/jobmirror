"""
Converts the mongoDB to parquet file in datamart
"""
from .mongodb_utils import *
from .feature_extraction.extract_features_jd import *
from .feature_extraction.extract_features_resume import *

import pyspark
from pyspark.sql.functions import to_date, col, expr, when, struct, transform
import os
import pandas as pd

def read_silver_labels(spark : SparkSession, datamart_dir : str, snapshot_date : datetime) -> None:
    """
    Draw the labels data and fit to parquet
    """

    # Datetime is randomized in the date of 2024, so we just need the month, datetime saved as string object
    # so need to use regex

    # Get the year-month from the snapshot date
    regex_string = "^" + str(snapshot_date.year) + "-" + str(snapshot_date.month)

    # Use double curly braces to escape for formatting strings
    df = spark.read.format("mongodb") \
        .option("database", "jobmirror_db") \
        .option("collection", "bronze_labels") \
        .option("pipeline", f'''
        [
            {{
                "$match": {{
                    "snapshot_date": {{ "$regex" : "{regex_string}" }}
                }}
            }}
        ]
        ''') \
        .load()
    print("Number of label rows read : {}".format(df.count()))
    
    df2 = df.select(
        col("_id"),
        col("resume_id"),
        col("job_id"),
        col("fit"),
        to_date(col("snapshot_date"), "yyyy-MM-dd").alias("snapshot_date")
    )

    # Convert the fit to float
    df2 = df2.withColumn("fit_score",
                         when(col("fit") == "No Fit", 0.0).when(col("fit") == "Potential Fit", 0.5).when(col("fit") == "Good Fit", 1.0))

    filename    = "labels_" + str(snapshot_date.year) + "-" + str(snapshot_date.month) + ".parquet"
    output_path = os.path.join(datamart_dir, filename)
    df2.write.mode("overwrite").parquet(output_path)

def read_silver_jd(spark : SparkSession, datamart_dir : str, snapshot_date : datetime) -> None:
    """
    Draw the JD and read the data, parse to parquet
    """

    # Get the year-month from the snapshot date
    regex_string = "^" + str(snapshot_date.year) + "-" + str(snapshot_date.month)

    df = spark.read.format("mongodb") \
        .option("database", "jobmirror_db") \
        .option("collection", "bronze_job_descriptions") \
        .option("pipeline", f'''
        [
            {{
                "$match": {{
                    "snapshot_date": {{ "$regex" : "{regex_string}" }}
                }}
            }}
        ]
        ''') \
        .load()
    print("Number of JD rows read : {}".format(df.count()))
    
    df_cleaned = df.withColumn("_id", expr("string(_id)")) \
    .withColumn("snapshot_date", to_date(col("snapshot_date"), "yyyy-MM-dd"))

    df_selected = df_cleaned.select(
        "id", "company_name", "role_title", "employment_type",
        "job_location", "about_the_company",
        "job_responsibilities", "required_hard_skills",
        "required_soft_skills", "required_language_proficiencies",
        "required_education", "required_work_authorization",
        "certifications", "snapshot_date"
    )
    
    # We need to process the required education using the embeddings model and check for similarity
    filename    = "jd_" + str(snapshot_date.year) + "-" + str(snapshot_date.month) + ".parquet"
    output_path = os.path.join(datamart_dir, filename)
    df_selected.write.mode("overwrite").parquet(output_path)

def read_silver_resume(spark : SparkSession, datamart_dir : str, snapshot_date : datetime):
    """
    Draw the resume and read the data, parse to parquet
    """

    # Get the year-month from the snapshot date
    regex_string = "^" + str(snapshot_date.year) + "-" + str(snapshot_date.month)

    df = spark.read.format("mongodb") \
        .option("database", "jobmirror_db") \
        .option("collection", "bronze_resumes") \
        .option("pipeline", f'''
        [
            {{
                "$match": {{
                    "snapshot_date": {{ "$regex" : "{regex_string}" }}
                }}
            }}
        ]
        ''') \
        .load()
    print("Number of resume rows read : {}".format(df.count()))

    # Need to deal with void values
    df_fixed = df.withColumn(
        "experience",
        transform(
            col("experience"),
            lambda x: struct(
                x["role"].alias("role"),
                x["company"].alias("company"),
                x["date_start"].alias("date_start"),
                x["date_end"].alias("date_end"),
                x["role_description"].alias("role_description"),
                x["snapshot_date"].cast("string").alias("snapshot_date"),
                x["id"].cast("string").alias("id")
            )
        )
    )
    df_fixed = df_fixed.withColumn(
        "education",
        transform(
            col("education"),
            lambda x: struct(
                x["degree"].alias("degree"),
                x["institution"].alias("institution"),
                x["date_start"].alias("date_start"),
                x["date_end"].alias("date_end"),
                x["grade"].alias("grade"),
                x["description"].alias("description"),
                x["snapshot_date"].cast("string").alias("snapshot_date"),
                x["id"].cast("string").alias("id")
            )
        )
    )  
    
    df_selected = df_fixed.select(
        "id", "name", "location_preference", "work_authorizaton",
        "employment_type_preference", "hard_skills",
        "soft_skills", "languages",
        "experience", "education",
        "certifications", "snapshot_date"
    )

    filename = "resume_" + str(snapshot_date.year) + "-" + str(snapshot_date.month) + ".parquet"
    output_path = os.path.join(datamart_dir, filename)
    df_selected.write.mode("overwrite").parquet(output_path)
    