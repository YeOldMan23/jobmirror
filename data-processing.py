import os
from dotenv import load_dotenv
import argparse

from pyspark.sql import SparkSession

from utils.data_processing_bronze_table import process_bronze_table


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Process data.")
    parser.add_argument('--start', type=int, required=True, help='Start index')
    parser.add_argument('--end', type=int, required=True, help='End index')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size for bronze table processing')
    args = parser.parse_args()

    load_dotenv()
    os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")

    mongodb_uri =  os.getenv("MONGO_DB_URL")

    spark = SparkSession.builder \
        .appName("SaveJSONtoMongoDB") \
        .config("spark.mongodb.read.connection.uri", mongodb_uri) \
        .config("spark.mongodb.write.connection.uri", mongodb_uri) \
        .config("spark.jars.packages", "org.mongodb.spark:mongo-spark-connector_2.12:10.5.0") \
        .getOrCreate()
    
    process_bronze_table(spark, args.start, args.end, args.batch_size)