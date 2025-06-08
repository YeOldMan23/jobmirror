"""
Test the MongoDB features to extract
"""

from utils.mongodb_to_parquet import *

from datetime import datetime

def get_snapshot_dates(startdate : datetime, enddate : datetime):
    """
    Return the first day of all year-month combos
    """
    current_date = datetime(startdate.year, startdate.month, 1)

    all_dates = []
    while current_date <= enddate:
        all_dates.append(current_date)

        if current_date.month == 12:
            current_date = datetime(current_date.year + 1, 1, 1)

        else:
            current_date = datetime(current_date.year, current_date.month + 1, 1)

    return all_dates

if __name__ == "__main__":
    # Initialize spark session
    spark = get_pyspark_session()
    datamart_dir = os.path.join(os.getcwd(), "datamart")

    # Get all dates from the start of 2024 to the end
    snapshot_dates = get_snapshot_dates(datetime(2021, 1, 1), datetime(2021, 12, 1))
    print("Number of Dates : ", len(snapshot_dates))

    # For each of the snapshot dates, get the corresponding data
    for snapshot_date in snapshot_dates:
        read_silver_resume(spark, datamart_dir, snapshot_date)
        read_silver_jd(spark, datamart_dir, snapshot_date)
        read_silver_labels(spark, datamart_dir, snapshot_date)
    
    pass