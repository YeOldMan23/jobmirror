from pyspark.sql import DataFrame
from pyspark.sql.functions import col, lower, when, regexp_replace, split, trim, udf
from pyspark.sql.types import StringType
import pandas as pd 
import re

#######################################
# Resume and JD cleaning
#######################################

def clean_employment_type_column(df: DataFrame, employment_type: str) -> DataFrame:
    # Lowercase and clean employment type
    cleaned_col = f"{employment_type}_cleaned"
    return df.withColumn(
        cleaned_col,
        when(col(employment_type).isNull(), 'full-time')
        .otherwise(
            when(
                (lower(col(employment_type)).rlike('full|remote|flexible|fte')),
                'full-time'
            ).when(
                lower(col(employment_type)).contains('part'),
                'part-time'
            ).when(
                lower(col(employment_type)).contains('intern'),
                'internship'
            ).when(
                lower(col(employment_type)).contains('contract'),
                'contract'
            ).otherwise('full-time')
        )
    )

def location_lookup(target_country_code='US'):
    city_df = pd.read_csv("datamart/bronze/location/cities.csv")
    # This function returns a Python dict, so you can use it in a UDF
    location_lookup = {}
    for idx, row in city_df.iterrows():
        if row['country_code'] != target_country_code:
            continue
        city_name = str(row['name']).strip().lower() if pd.notna(row['name']) else None
        state_code = str(row['state_code']).strip().lower() if pd.notna(row['state_code']) else None
        state_name = str(row['state_name']).strip().lower() if pd.notna(row['state_name']) else None
        if state_code:
            location_lookup[state_code] = row['state_code']
        if state_name:
            location_lookup[state_name] = row['state_code']
        if city_name:
            location_lookup[city_name] = row['state_code']
    return location_lookup

def standardize_location_column(df: DataFrame, location_col: str, location_lookup: dict, location_cleaned=None) -> DataFrame:
    if location_cleaned is None:
        location_cleaned = f"{location_col}_cleaned"

    def clean_location(x):
        if x is None:
            return 'Not specified'
        loc = x.strip().lower()
        parts = re.split(r',|\bor\b|\/|\band\b', loc)
        for part in parts:
            part = part.strip()
            if part in location_lookup:
                return location_lookup[part]
        return 'Not specified'

    clean_location_udf = udf(clean_location, StringType())
    return df.withColumn(location_cleaned, clean_location_udf(col(location_col)))

def clean_work_authorization_column(df: DataFrame, work_auth: str) -> DataFrame:
    clean_col = work_auth + '_cleaned'
    def standardize_work_authorization(x):
        if x is None or str(x).strip().lower() == 'none':
            return 'not needed'
        elif re.search(r'do\s*not\s*require|no\s*sponsorship|none', str(x), re.IGNORECASE):
            return 'not needed'
        else:
            return 'needed'
    standardize_work_auth_udf = udf(standardize_work_authorization, StringType())
    return df.withColumn(clean_col, standardize_work_auth_udf(col(work_auth)))

#######################################
# Label cleaning 
#######################################

def standardize_label(df: DataFrame, label: str, label_clean=None) -> DataFrame:
    if label_clean is None:
        label_clean = label + '_cleaned'
    def regroup_label(fit_value):
        if fit_value in ['Good Fit', 'Potential Fit']:
            return 'Fit'
        else:
            return fit_value
    regroup_label_udf = udf(regroup_label, StringType())
    return df.withColumn(label_clean, regroup_label_udf(col(label)))