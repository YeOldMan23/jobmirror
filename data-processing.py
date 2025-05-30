import os
from dotenv import load_dotenv

from pyspark.sql import SparkSession

from utils.data_processing_bronze_table import process_bronze_table



if __name__ == "__main__":
    load_dotenv()
    os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")

    mongodb_uri =  os.getenv("MONGODB_URI")

    spark = SparkSession.builder \
        .appName("SaveJSONtoMongoDB") \
        .config("spark.mongodb.read.connection.uri", mongodb_uri) \
        .config("spark.mongodb.write.connection.uri", mongodb_uri) \
        .config("spark.jars.packages", "org.mongodb.spark:mongo-spark-connector_2.12:10.5.0") \
        .getOrCreate()
    
    process_bronze_table(spark)