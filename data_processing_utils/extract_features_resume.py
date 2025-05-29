"""
Extract the number of years of experience from the json file
"""

from datetime import datetime

def extract_yoe(resume_json : dict) -> float:
    """
    Get Years of experience from resume
    """
    experience_list = resume_json['experience']

    total_yoe = 0

    for experience in experience_list:
        start_time = datetime.fromisoformat(experience['date_start'])
        end_time   = datetime.fromisoformat(experience['date_end'])

        day_difference = (end_time - start_time).days

        year_difference = day_difference / 365.25

        total_yoe += year_difference

    return total_yoe

def extract_language(resume_json : dict) -> list:
    """
    Get the languages from resume, if no language listed,
    assume English
    """
    languages = resume_json['languages']

    # Inferred english language
    if len(languages) == 0:
        return ['english']

    for idx in range(len(languages)):
        languages[idx] = languages[idx].lower()

    # Append english if not inside, inferred
    if 'english' not in languages:
        languages.append('english')

    return languages