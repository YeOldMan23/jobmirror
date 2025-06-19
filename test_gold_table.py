from utils.data_processing_gold_table import data_processing_gold_features

from utils.date_utils import *
from utils.mongodb_utils import get_pyspark_session
import os

if __name__ == "__main__":
    # Get the pyspark session
    spark = get_pyspark_session()
    
    # Datamart dir
    datamart_dir = os.path.join(os.getcwd(), "datamart")

    # Get the range of dates
    date_range = get_snapshot_dates(datetime(2021, 6, 1), datetime(2021, 7, 31))

    # For each range, read the silver table and parse
    for cur_date in date_range:
        snapshot_date = f"{cur_date.year}-{cur_date.month}"

        print("Processing gold {}".format(snapshot_date))

        data_processing_gold_features(cur_date, spark)