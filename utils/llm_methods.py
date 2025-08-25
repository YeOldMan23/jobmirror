import os
import numpy as np
import json
import requests

llama_url = "http://llm-api:5000/"

# LLM Stuff
INST_BEGIN = "[INST] \n"
INST_END   = "\n[INST]"

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

# TODO Apply post prompt protection


def get_response(main_information : str) -> str:
    """
    Get the prompt from Llama Dockerfile, and return the response 
    """
    generate_url = llama_url + "generate"
    headers      = {"Content-Type": "application/json"}

    # Make the prompt
    total_prompt = gemini_date_prompt + INST_BEGIN + main_information + INST_END

    payload = {"prompt" : total_prompt}

    try:
        response = requests.post(generate_url, data=json.dumps(payload), headers=headers)
        response.raise_for_status()  # Raise an exception for bad status codes
        result = response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error calling LLM API: {e}")
        raise

    return result

def parse_json_string(json_string : str):
    try:
        # Attempt to load the string as a JSON object
        json_object = json.loads(json_string)
        print("The string is a valid JSON string.")
        return json_object
    except json.JSONDecodeError as e:
        # Catch the specific error for invalid JSON format
        print(f"The string is not a valid JSON string. Error: {e}")
        return None

def get_resume_metrics(pre_prompt : str, resume_details : str, verbose : bool = False):
    """
    Get the resume details from the resume string
    """
    resume_prompt = pre_prompt + resume_details
    json_string = get_response(resume_prompt)
    
    # Print if necessary
    if verbose:
        print("Resume Metrics Result")
        print(json_string)

    # Check if the results is of json format
    json_data = parse_json_string(json_string)
    if json_data is None:
        print("Resume JSON Data is unparsable")

    return json_data

def get_jd_metrics(pre_prompt : str, jd_details : str, verbose : bool = False):
    """
    Get the JD details from the JD string
    """
    jd_prompt = pre_prompt + jd_details
    json_string = get_response(jd_prompt)
    
    # Print if necessary
    if verbose:
        print("Job Description Metrics Result")
        print(json_string)

    # Check if the results is of json format
    json_data = parse_json_string(json_string)
    if json_data is None:
        print("Job Description JSON Data is unparsable")

    return json_data

def get_matching_details(jd_json : dict, resume_json : dict, question : str, expected_type = float):
    """
    Use the JD and Resume details, do a comparison between 2 details within the resume and JD
    to give a score
    """
    company_name = jd_json['company_name']

    # Make the JSONs into string
    jd_json_string = json.dumps(jd_json, indent=4)
    resume_json_string = json.dumps(resume_json, indent=4)

    llm_match_prompt = f"""

    You are a recruiter for the company {company_name} and your job is to determine the match of the anonymous JD with
    the anonymous resume. The sensitive details for the applicant have been removed, so there should be no age, race or gender
    bias in the matching. Below are the string based inputs for the JD and the resume stuff, so do your best as a recruiter 
    to do a proper comparison between the 2 values.

    ** JD metrics **

    {jd_json_string}

    ** Resume Metrics ** 

    {resume_json_string}

    Return the value for the question below, consider its interpretation or just the actual word if it is available in the dictionary

    {question}

    """

    # Parse it throught 
    answer_string = get_response(llm_match_prompt)

    try:
        parsed_output = expected_type(answer_string)
    except:
        print(f"{answer_string} cannot be converted to type {expected_type}")

    return parsed_output