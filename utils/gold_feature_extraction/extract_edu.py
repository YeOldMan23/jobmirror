"""
File includes functions to extract educational features from the combined silver dataset & engineers gold level features

Input features from silver dataset:
- resume features:
    - edu_highest_level: highest level of education achieved (e.g., Bachelor's, Master's)
    - edu_field: field of study (e.g., Computer Science, Engineering)
    - edu_gpa: GPA (e.g., 3.5)
    - edu_institution: name of the educational institution (e.g., Stanford University)
    - cert_categories: list of certification categories (e.g., AWS, Azure)
- jd features:
    - edu_required_level: required level of education (e.g., Bachelor's, Master's)
    - edu_required_field: required field of study (e.g., Computer Science, Engineering)
    - required_cert_categories: list of required certification categories (e.g., AWS, Azure)

Output features for gold dataset:
- edu_level_match: boolean indicating if the resume's highest education level matches the JD's required level
- edu_level_score: float between 0 and 1 {1: exact match, 0.75: overqualified, 0: underqualified} indicating the match score based on education level
- edu_field_match: float indicating the match score between the resume's field of study and the JD's required field
- edu_gpa: float between 0 and 1 indicating performance based on GPA, larger is better
- edu_institution: float between 0 and 1 indicating the prestige of the institution, larger is better
- cert_match: boolean indicating if the resume's certifications match the JD's required certifications
"""

