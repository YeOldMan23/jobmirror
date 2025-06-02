import os
import json
from data_processing_utils.match_features import *

examples_dir = os.path.join(os.getcwd(), "examples")
JD_dir = os.path.join(examples_dir, "jd")
resume_dir = os.path.join(examples_dir, "resume")

# Read the JD's resume
all_JDs = os.listdir(JD_dir)

for JD in all_JDs:
    # Get the correspondng resume
    resume_full_dir = os.path.join(examples_dir, "resume", JD)
    jd_full_dir = os.path.join(JD_dir, JD)

    # Read the data
    with open(resume_full_dir, "r") as resume_file:
        resume_data = json.load(resume_file)
    with open(jd_full_dir, "r") as jd_file:
        jd_data = json.load(jd_file)

    # parse the data, pass through resume
    hs_match_score = get_hard_skill_match_score(jd_data, resume_data)
    jd_match_score = get_job_title_match_score(jd_data, resume_data)
    print(f"JD Name : {JD} Hard Skill Match Score : {hs_match_score} Job Title Match Score : {jd_match_score}")
