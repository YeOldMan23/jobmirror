from pydantic import BaseModel, Field
from typing import List, Optional, get_origin, get_args, Union
import datetime

class JD(BaseModel):
    company_name: Optional[str] = Field(None, description="Name of the company posting the job")
    role_title: Optional[str] = Field(None, description="The title or name of the job role being offered")
    employment_type: Optional[str] = Field(None, description="Type of employment, such as Full-time, Part-time, Contract, Freelance, or Internship")
    about_the_company: Optional[str] = Field(None, description="A brief overview or description of the company")
    job_responsibilities: List[str] = Field(..., description="A list of key duties, tasks, or responsibilities associated with the job")
    required_hard_skills: List[str] = Field(..., description="A list of technical or hard skills required or preferred for the job")
    required_soft_skills: List[str] = Field(..., description="A list of soft skills or character required or preferred for the job")
    required_language_proficiencies: List[str] = Field(..., description="A list of language proficiencies required for the job, excluding programming languages")
    required_work_authorization: Optional[str] = Field(None, description="Work authorization required for the job")
    required_education: Optional[str] = Field(None, description="The minimum educational qualification required for the job, such as a degree or certification")
    job_location: Optional[str] = Field(None, description="Location where the job is based, such as a city or remote")
    date_posted: Optional[datetime.datetime] = Field(None, description="The date that the job is posted")