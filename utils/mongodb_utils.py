import os
from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi
import pandas as pd

def get_mongo_client() -> MongoClient:
    uri = os.environ.get("MONGO_DB_URL")
    client = MongoClient(uri, server_api=ServerApi('1'))
    return client

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

def read_bronze_resumes_as_dataframe(db_name, collection_name, query=None):
    client = get_mongo_client()
    collection = client[db_name][collection_name]
    data = list(collection.find(query or {}))
    if data and "_id" in data[0]:
        for d in data:
            d.pop("_id", None)
    return pd.DataFrame(data)