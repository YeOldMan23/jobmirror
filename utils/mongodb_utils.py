import os
from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi
import pandas as pd
from pyspark.sql import SparkSession
from dotenv import load_dotenv

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
        .getOrCreate()
    return spark

def test_mongo_connection(client: MongoClient):
    try:
        client.admin.command('ping')
        print("Pinged your deployment. You successfully connected to MongoDB!")
    except Exception as e:
        print(f"MongoDB connection failed: {e}")

def get_resume_collection(db_name="jobmirror_db", collection_name="bronze_resumes"):
    client = get_mongo_client()
    return client[db_name][collection_name]

def get_jd_collection(db_name="jobmirror_db", collection_name="bronze_job_descriptions"):
    client = get_mongo_client()
    return client[db_name][collection_name]

def exists_in_collection(collection, doc_id):
    return collection.find_one({"id": doc_id}) is not None

def read_bronze_table_as_pyspark(db_name, collection_name):
    spark = get_pyspark_session()
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