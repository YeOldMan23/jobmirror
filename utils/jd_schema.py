from pydantic import BaseModel, Field
from typing import List, Optional, get_origin, get_args, Union
import datetime

class JD(BaseModel):
    company_name: Optional[str] = Field(None, description="Name of the company posting the job")
    role_title: Optional[str] = Field(None, description="Job title or position being offered")
    application_deadline: Optional[str] = Field(None, description="The deadline for submitting applications. Dates must be in ISO 8601 format (YYYY-MM-DDTHH:MM:SS)")
    date_posted: Optional[str] = Field(None, description="The date when the job was posted. Dates must be in ISO 8601 format (YYYY-MM-DDTHH:MM:SS)")
    employment_type: Optional[str] = Field(None, description="Type of employment, such as Full-time, Part-time, Contract, Freelance, or Internship. If not stated, it is assumed to be Full-time")
    about_the_company: Optional[str] = Field(None, description="A brief overview or description of the company")
    job_responsibilities: List[str] = Field(default_factory=list, description="A list of key duties, tasks, or responsibilities associated with the job")
    required_hard_skills: List[str] = Field(default_factory=list, description="A list of technical or hard skills required or preferred for the job. Keep it as keywords. This includes programming languages, software tools, or frameworks like Python, Java, SQL")
    required_soft_skills: List[str] = Field(default_factory=list, description="A list of soft skills or character required or preferred for the job. Keep it as keywords. This includes communication, teamwork, or leadership skills")   
    required_language_proficiencies: List[str] = Field(default_factory=list, description="A list of language proficiencies required for the job if stated. If the job description does not mention any languages, then fill this with the language that the job description is written in")
    required_education: Optional[str] = Field(None, description="The minimum educational qualification required for the job, such as a degree or certification")
    required_work_authorization: Optional[str] = Field(None, description="Work authorization required for the job")
    job_location: Optional[str] = Field(None, description="Location where the job is based, such as a city or remote")
    certifications: List[str] = Field(default_factory=list, description="A list of certifications or licenses related with hard skills, medical skills, and software tools required by the job. certifications should relate only to verifiable credentials (e.g., AWS, CISSP, PMP). Do not include work roles or job titles as certifications")