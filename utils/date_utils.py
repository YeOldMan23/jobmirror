"""
Datetime utils
"""

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