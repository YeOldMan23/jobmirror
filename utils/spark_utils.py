import os
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.types import StructType

def create_spark_session(app_name: str = "JobMirrorApp") -> SparkSession:
    spark = SparkSession.builder \
        .appName(app_name) \
        .config("spark.mongodb.read.connection.uri", os.environ.get("MONGO_DB_URL")) \
        .config("spark.mongodb.write.connection.uri", os.environ.get("MONGO_DB_URL")) \
        .config("spark.jars.packages", "org.mongodb.spark:mongo-spark-connector_2.12:10.2.0") \
        .getOrCreate()
    return spark

def create_dataframe(spark: SparkSession, data: list, schema: StructType) -> DataFrame:
    return spark.createDataFrame(data, schema)

def write_to_mongodb(df: DataFrame, database: str, collection: str, mode: str = "overwrite"):
    df.write.format("mongodb") \
        .mode(mode) \
        .option("database", database) \
        .option("collection", collection) \
        .save()