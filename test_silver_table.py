"""
Test the silver table stuff 
"""
from utils.data_processing_silver_table import *
from utils.date_utils import *

from pyspark.sql import SparkSession

if __name__ == "__main__":
    # Get the pyspark session
    spark = pyspark.sql.SparkSession.builder \
            .appName("SilverParquet") \
            .config("spark.ui.enabled", "true") \
            .config("spark.hadoop.hadoop.native.lib", "false") \
            .config("spark.driver.memory", "4g") \
            .getOrCreate()
    
    # Some people have many experiences, we need to increase the string fields amount
    spark.conf.set("spark.sql.debug.maxToStringFields", 100)
    
    # Datamart dir
    datamart_dir = os.path.join(os.getcwd(), "datamart")

    # Get the range of dates
    date_range = get_snapshot_dates(datetime(2021, 1, 1), datetime(2021, 12, 31))

    # For each range, read the silver table and parse
    for cur_date in date_range:
        snapshot_date = f"{cur_date.year}-{cur_date.month}"

        print("Processing silver {}".format(snapshot_date))

        # Process the date
        data_processing_silver_table(datamart_dir, snapshot_date, spark)

