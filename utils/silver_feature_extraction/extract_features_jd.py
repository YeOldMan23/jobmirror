# """
# Extract features from the Job Description
# """
# import re

# def extract_authorization(jd_json : dict):
#     """
#     Get the authorization from the JD
#     """

#     if not jd_json['required_work_authorization']:
#         return []
    
#     authorization_information = jd_json['required_work_authorization'].lower()

#     match = re.search(r'(\b[\w\s]+?)\s+citizen\b', authorization_information, re.IGNORECASE)

#     if match:
#         return match.group(0)

#     return []

# def extract_hard_skills_jd(jd_json : dict) -> list:
#     """
#     Extract the required job descriptions from the job descripton
#     """
#     hard_skills = jd_json['required_hard_skills']

#     for idx in range(len(hard_skills)):
#         hard_skills[idx] = hard_skills[idx].lower()

#     return hard_skills

# def extract_job_title_jd(jd_json : dict) -> list:
#     """
#     Get the job title from the description
#     """
#     job_title = jd_json["role_title"].lower()

#     return [job_title]

# def extract_required_education(jd_json : dict):
#     """
#     Extract the required education from the from the JD
#     """
#     pass