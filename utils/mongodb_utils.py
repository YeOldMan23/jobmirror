import os
import pandas as pd
import json
from datetime import datetime

from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi

from dotenv import load_dotenv

import pyspark
from pyspark.sql.functions import to_date, col, expr, when, struct, transform
from pyspark.sql import SparkSession


def get_mongo_client() -> MongoClient:
    load_dotenv()
    uri = os.environ.get("MONGO_DB_URL")
    client = MongoClient(uri, server_api=ServerApi('1'))
    return client

def get_pyspark_session() -> SparkSession:
    load_dotenv()
    mongodb_uri = os.getenv("MONGO_DB_URL")
    spark = SparkSession.builder \
        .appName("MongoDBSpark") \
        .config("spark.mongodb.read.connection.uri", mongodb_uri) \
        .config("spark.mongodb.write.connection.uri", mongodb_uri) \
        .config("spark.jars.packages", "org.mongodb.spark:mongo-spark-connector_2.12:10.5.0") \
        .config("spark.jars.repositories", "https://repo1.maven.org/maven2/") \
        .config("spark.driver.memory", "4g") \
        .config("spark.sql.debug.maxToStringFields", 100) \
        .getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")
    return spark

def test_mongo_connection(client: MongoClient):
    try:
        client.admin.command('ping')
        print("Pinged your deployment. You successfully connected to MongoDB!")
    except Exception as e:
        print(f"MongoDB connection failed: {e}")

def get_collection(db_name, collection_name):
    client = get_mongo_client()
    return client[db_name][collection_name]

def exists_in_collection(collection, doc_id):
    return collection.find_one({"id": doc_id}) is not None

#############################
# Read bronze tables
#############################
def read_bronze_table_as_pyspark(db_name, collection_name, spark: SparkSession):
    return spark.read \
        .format("mongodb") \
        .option("database", db_name) \
        .option("collection", collection_name) \
        .load()

def read_bronze_table_as_pandas(db_name, collection_name, query=None):
    client = get_mongo_client()
    collection = client[db_name][collection_name]
    data = list(collection.find(query or {}))
    if data and "_id" in data[0]:
        for d in data:
            d.pop("_id", None)
    return pd.DataFrame(data)

def read_bronze_labels_as_pyspark(snapshot_date : datetime, spark: SparkSession) -> pyspark.sql.dataframe.DataFrame:
    """
    Read labels as pyspark
    """

    # Datetime is randomized in the date of 2024, so we just need the month, datetime saved as string object
    # so need to use regex

    # Get the year-month from the snapshot date
    regex_string = "^" + str(snapshot_date.year) + "-" + str(snapshot_date.month).zfill(2)

    # Use double curly braces to escape for formatting strings
    df = read_bronze_table_as_pyspark("jobmirror_db", "bronze_labels", spark)
    df = df.filter(col("snapshot_date").rlike(regex_string))
    print("Number of label rows read : {} Snapshot Date : {}".format(df.count(), datetime.strftime(snapshot_date, "%Y-%m-%d")))
    
    df2 = df.select(
        col("_id"),
        col("resume_id"),
        col("job_id"),
        col("fit"),
        to_date(col("snapshot_date"), "yyyy-MM-dd").alias("snapshot_date")
    )

    return df2

def read_bronze_jd_as_pyspark(snapshot_date : datetime, spark: SparkSession) -> pyspark.sql.dataframe.DataFrame:
    """
    Draw the JD and read the data, parse to parquet
    """

    # Get the year-month from the snapshot date
    regex_string = "^" + str(snapshot_date.year) + "-" + str(snapshot_date.month).zfill(2)

    df = read_bronze_table_as_pyspark("jobmirror_db", "bronze_job_descriptions", spark)
    df = df.filter(col("snapshot_date").rlike(regex_string))
    print("Number of JD rows read : {} Snapshot Date : {}".format(df.count(), datetime.strftime(snapshot_date, "%Y-%m-%d")))
    
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
    
    return df_selected

def read_bronze_resume_as_pyspark(snapshot_date : datetime, spark: SparkSession) -> pyspark.sql.dataframe.DataFrame:
    """
    Draw the resume and read the data, parse to parquet
    """

    # Get the year-month from the snapshot date
    regex_string = "^" + str(snapshot_date.year) + "-" + str(snapshot_date.month).zfill(2)

    df = read_bronze_table_as_pyspark("jobmirror_db", "bronze_resumes", spark)
    df = df.filter(col("snapshot_date").rlike(regex_string))
    print("Number of Resume rows read : {} Snapshot Date : {}".format(df.count(), datetime.strftime(snapshot_date, "%Y-%m-%d")))

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
        "id", "name", "location_preference", "work_authorization",
        "employment_type_preference", "hard_skills",
        "soft_skills", "languages",
        "experience", "education",
        "certifications", "snapshot_date"
    )

    return df_selected