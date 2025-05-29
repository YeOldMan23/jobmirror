"""
Extract features from the Job Description
"""
import re

def extract_authorization(jd_json : dict):
    """
    Get the authorization from the JD
    """

    if not jd_json['required_work_authorization']:
        return []
    
    authorization_information = jd_json['required_work_authorization'].lower()

    match = re.search(r'(\b[\w\s]+?)\s+citizen\b', authorization_information, re.IGNORECASE)

    if match:
        return match.group(0)

    return []

def extract_required_education(jd_json : dict):
    """
    Extract the required education from the JD
    """