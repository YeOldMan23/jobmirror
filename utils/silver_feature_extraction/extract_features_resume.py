# """
# Extract the number of years of experience from the json file
# """

# from datetime import datetime

# def extract_yoe(resume_json : dict) -> float:
#     """
#     Get Years of experience from resume
#     """
#     experience_list = resume_json['experience']

#     total_yoe = 0

#     for experience in experience_list:
#         start_time = datetime.fromisoformat(experience['date_start'])
#         end_time   = datetime.fromisoformat(experience['date_end'])

#         day_difference = (end_time - start_time).days

#         year_difference = day_difference / 365.25

#         total_yoe += year_difference

#     return total_yoe

# def extract_language(resume_json : dict) -> list:
#     """
#     Get the languages from resume, if no language listed,
#     assume English
#     """
#     languages = resume_json['languages']

#     # Inferred english language
#     if len(languages) == 0:
#         return ['english']

#     for idx in range(len(languages)):
#         languages[idx] = languages[idx].lower()

#     # Append english if not inside, inferred
#     if 'english' not in languages:
#         languages.append('english')

#     return languages

# def extract_hard_skills_resume(resume_json : dict) -> list:
#     """
#     Extract the required job descriptions from the resume
#     """
#     hard_skills = resume_json['hard_skills']

#     for idx in range(len(hard_skills)):
#         hard_skills[idx] = hard_skills[idx].lower()

#     return hard_skills

# def extract_job_titles_resume(resume_json : dict) -> list:
#     """
#     Get all the job titles from the resume
#     """
#     job_experiences = resume_json["experience"]

#     prev_jobs = []
#     for job_exp in job_experiences:
#         prev_jobs.append(job_exp["role"].lower())

#     return prev_jobs