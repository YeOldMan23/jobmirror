from utils.data_processing_gold_table import data_processing_gold_features, data_processing_gold_labels

from utils.date_utils import *
from utils.mongodb_utils import get_pyspark_session
import os

if __name__ == "__main__":
    # Get the pyspark session
    spark = get_pyspark_session()
    
    # Datamart dir
    datamart_dir = os.path.join(os.getcwd(), "datamart")

    # Check if the gold feature and label store exsits
    gold_dir = os.path.join(datamart_dir, "gold")
    feature_dir = os.path.join(gold_dir, 'feature_store')
    label_dir = os.path.join(gold_dir, "label_store")

    for cur_dir in [gold_dir, feature_dir, label_dir]:
        if not os.path.exists(cur_dir):
            os.mkdir(cur_dir)

    # Get the range of dates
    date_range = get_snapshot_dates(datetime(2022, 6, 1), datetime(2022, 12, 31))

    # For each range, read the silver table and parse
    for cur_date in date_range:
        snapshot_date = f"{cur_date.year}-{cur_date.month}"
        print("Processing gold {}".format(snapshot_date))

        data_processing_gold_features(cur_date, spark)
        data_processing_gold_labels(cur_date, spark)
