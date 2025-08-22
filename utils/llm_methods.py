import os
import numpy as np

from google import genai
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage
from langchain.prompts import PromptTemplate

load_dotenv()
google_apikey = os.getenv("GEMINI_APIKEY")

date_str = "31-12-2021"
gemini_date_prompt = "Take current reference date as {}".format(date_str)

gemini_resume_prompt = """

You are a resume information extractor, extracting out relevant information from 
a resume. You need to extract out the following information: Job Experience, Education, Skill Certifications.
Return only the JSON Format, as shown below.

**JSON Format**
{
    "work_experience" : [
        "JOB_1" : { // Reference as the job title that was placed in the resume
            "years_of_experience": float,
            "work_summary" : string
        }
        ... // There can be more than one job option in the list
    ],
    "education" : { // Indicate the fields for each education
        "phd" : str | None,
        "masters" : str | None,
        "bachelors" : str | None,
        "high_school" : str | None
    },
    "skill_certifications" : [], // Only consider well known ones, do online search if necessary 
}

**Additional Information**
For Work Experience, if years_of_experience reference date is in month-year, assume day is the last day of the month. If
years_of_experience is just year format, assume month is last month in the year. If no year is provided, provide default 
year of experience is 1 year. For the work summary, make a summary of all the points inidicating the impact and performance
during their time at the job.

For education, if the field is not indicated, default is "General".

** Resume Details **

"""

gemini_jd_prompt = """

You are a job description information extractor, extracting out relevant information from 
a job description. You need to extract out the following information: Job Requirements, Education Requirement, 
Skill Certifications, Work Certifications. Return only the JSON Format, as shown below.

**JSON Format**
{   
    "company_name" : string | None,
    "job_requirements" : {
        "required_years_of_experience" : float,
        "job_summary" : string
    },
    "education_requirements" : {
        "highest_level" : string, // Choose from the list here ["phd", "masters", "bachelors", "high_school"]
        "field" : string // Choose the best name that represents 
    },
    "skill_requirements" : [] // Only consider well known ones, do online search if necessary 
}

** Additional Information **

For the work summary, make a summary of all the points inidicating the impact and performance
during their time at the job. For the required years of experience, if no number is provided, get an estimate of
the number of years based off the job description.

"""

llm_match_prompt = f"""

You are a recruiter for the company {COMPANY_NAME} and your job is to determine the match of the anonymous JD with
the anonymous resume. The sensitive details for the applicant have been removed, so there should be no age, race or gender
bias in the matching. Below are the string based inputs for the JD and the resume stuff, so do your best as a recruiter 
to do a proper comparison between the 2 values.

** JD metrics **

** Resume Metrics ** 

Return the value for the question below

{QUESTION}

"""

def get_response(pre_prompt : str = ""):
    """
    Get the prompt from gemini
    """
    pass

def get_resume_metrics(pre_prompt : str, resume_details : str):
    """
    Get the resume details from the resume string
    """
    pass

def get_jd_metrics(pre_prompt : str, jd_details : str):
    """
    Get the JD details from the JD string
    """
    pass

def get_matching_details(jd_json : dict, resume_json : dict):
    """
    Use the JD and Resume details, do a comparison between 2 details within the resume and JD
    to give a score
    """
    pass
    

    
