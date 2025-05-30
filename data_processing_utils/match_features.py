"""
Match the features between the JD and Resume to get more features
"""

from .extract_features_jd import *
from .extract_features_resume import *

import pandas as pd
import torch

# BERT based comparison
from sentence_transformers import SentenceTransformer, util

embedding_model = SentenceTransformer("BAAI/bge-large-en-v1.5")

def get_hard_skill_match_score(jd_json : dict, resume_json : dict) -> float:
    """
    Get the hard skills from both the JD and resume, and return a
    matching score on the match of the skills from the match
    """
    
    jd_skills_list     = extract_hard_skills_jd(jd_json)
    resume_skills_list = extract_hard_skills_resume(resume_json)

    # Prompt engineering
    jd_skills_list     = ["Represent this sentence for retrieval: " + i for i in jd_skills_list]
    resume_skills_list = ["Represent this sentence for retrieval: " + i for i in resume_skills_list]

    # Get embeddings
    resume_embeddings = embedding_model.encode(resume_skills_list, 
                                               convert_to_tensor=True,
                                               normalize_embeddings=True)
    job_embeddings    = embedding_model.encode(jd_skills_list, 
                                               convert_to_tensor=True,
                                               normalize_embeddings=True)

    # Get similarity Matrix
    similarity_matrix = util.cos_sim(job_embeddings, resume_embeddings)
    top_scores_jd     = torch.max(similarity_matrix, dim=1).values
    # top_scores_resume  = torch.max(similarity_matrix, dim=0).values

    # Get the average score to do matching
    average_score = torch.mean(top_scores_jd).item()

    # TODO We can also do fuzzy-matching between

    return average_score

def get_date_of_applcation(jd_json : dict, resume_json : dict) -> datetime:
    """
    Get the date of applicaton from the either the JD or the resume
    if the date is not available from the JD, take it from the resume
    """

    # If the applicaton date is in the JD use that as the reference date
    if "application_date" in jd_json.keys():
        application_date = datetime.fromisoformat(jd_json["application_date"])
        return application_date
    
    # Try to extract the resume json
    end_dates = []
    for experience in resume_json["experience"]:
        # Get the end date from the experience
        try:
            end_date = datetime.fromisoformat(experience["date_end"])
        except:
            # Assume the current date 
            end_date = datetime.now()

        end_dates.append(end_date)

    latest_date = max(end_dates)

    return latest_date
