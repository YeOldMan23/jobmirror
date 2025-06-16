import pandas as pd
import re


#######################################
# Resume and JD cleaning
#######################################

def clean_employment_type_column(df, emplyment_type):
    def clean_type(val):
        if pd.isna(val):
            return 'full-time'
        val = str(val).lower()
        if any(keyword in val for keyword in ['full', 'remote', 'flexible', 'fte']):
            return 'full-time'
        elif 'part' in val:
            return 'part-time'
        elif 'intern' in val:
            return 'internship'
        elif 'contract' in val:
            return 'contract'
        else:
            return 'full-time'
    
    cleaned_emplyment_type = f"{emplyment_type}_cleaned"
    df[cleaned_emplyment_type] = df[emplyment_type].apply(clean_type)
    return df

def location_lookup(city_df_list, target_country_code='US'):
    
    location_lookup = {}

    for idx, row in city_df_list.iterrows():
        if row['country_code'] != target_country_code:
            continue

        city_name = str(row['name']).strip().lower() if pd.notna(row['name']) else None
        state_code = str(row['state_code']).strip().lower() if pd.notna(row['state_code']) else None
        state_name = str(row['state_name']).strip().lower() if pd.notna(row['state_name']) else None

        if state_code:
            location_lookup[state_code] = row['state_code']  # Map state code to itself
        if state_name:
            location_lookup[state_name] = row['state_code']  # Map state name to state code
        if city_name:
            location_lookup[city_name] = row['state_code']  # Map city name to state code

    return location_lookup



def standardize_location_column(df, location_str, location_lookup, location_cleaned=None):
    def clean_location(x):
        if pd.isna(x):
            return 'Not specified'

        loc = x.strip().lower()
        parts = re.split(r',|\bor\b|\/|\band\b', loc)

        for part in parts:
            part = part.strip()
            if part in location_lookup:
                return location_lookup[part]

        return 'Not specified'

    if location_cleaned is None:
        location_cleaned = f"{location_str}_cleaned"

    df[location_cleaned] = df[location_str].apply(clean_location)
    return df

def clean_work_authorization_column(df, work_auth):
    def standardize_work_authorization(x):
        if pd.isna(x) or str(x).strip().lower() == 'none':
            return 'not needed'
        elif re.search(r'do\s*not\s*require|no\s*sponsorship|none', str(x), re.IGNORECASE):
            return 'not needed'
        else:
            return 'needed'
    
    clean_work_auth = work_auth + '_cleaned'
    df[clean_work_auth] = df[work_auth].apply(standardize_work_authorization)
    return df


#######################################
# Label cleaning
#######################################

def standardize_label(df, label, label_clean=None):
    def regroup_label(fit_value):
        if fit_value in ['Good Fit', 'Potential Fit']:
            return 'Fit'
        else:
            return fit_value

    if label_clean is None:
        label_clean = label + '_cleaned'

    df[label_clean] = df[label].apply(regroup_label)
    return df
