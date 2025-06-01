from pydantic import BaseModel, Field, field_validator
from typing import List, Optional, get_origin, get_args, Union
import datetime

class Experience(BaseModel):
    role: Optional[str] = Field(None, description="The job title or position held")
    company: Optional[str] = Field(None, description="The name of the company")
    date_start: Optional[str] = Field(None, description="The start date of the job in ISO 8601 format (YYYY-MM-DD or YYYY-MM-DDTHH:MM:SS)")
    date_end: Optional[str] = Field(None, description="The end date of the job in ISO 8601 format (YYYY-MM-DD or YYYY-MM-DDTHH:MM:SS), or 'current'/'present'/'ongoing' if specified")
    role_description: Optional[str] = Field(None, description="A description of the responsibilities and achievements in the role")

class Education(BaseModel):
    degree: Optional[str] = Field(None, description="The academic degree obtained")
    institution: Optional[str] = Field(None, description="The name of the educational institution")
    date_start: Optional[str] = Field(None, description="The start date of the education program in ISO 8601 format (YYYY-MM-DD or YYYY-MM-DDTHH:MM:SS)")
    date_end: Optional[str] = Field(None, description="The end date of the education program in ISO 8601 format (YYYY-MM-DD or YYYY-MM-DDTHH:MM:SS), or 'current'/'present'/'ongoing' if specified")
    grade: Optional[float] = Field(None, description="The GPA or final grade, if available")
    description: Optional[str] = Field(None, description="Additional details about the education")

class Resume(BaseModel):
    name: Optional[str] = Field(None, description="Full name of the person")
    location_preference: Optional[str] = Field(None, description="Preference for their work location or remote, if stated")
    work_authorization: Optional[str] = Field(None, description="Work authorization that the person holds, such as citizenship, if stated")
    employment_type_preference: Optional[str] = Field(
        None,
        description="Type of employment the resume is looking for such as Full-time, Part-time, Contract, Freelance, or Internship, if stated"
    )
    hard_skills: List[str] = Field(..., 
                                   description="A list of proficiencies in tools, technologies, frameworks, programming languages, platforms, methodologies, and key professional terms mentioned in the resume. " \
                                   "Avoid duplicates and use concise wording." \
                                   "Clean up tool names and merge variations.")
    soft_skills: List[str] = Field(..., description="A list of soft skills mentioned in the resume, such as communication, teamwork, and leadership. Avoid duplication.")
    languages: List[str]= Field(..., description="A list of language proficiencies mentioned in the resume, excluding programming languages")
    experience: List[Experience] = Field(..., description="A list of past work experiences")
    education: List[Education] = Field(..., description="A list of educational qualifications")
    certifications: List[str] = Field(..., description="A list of certifications or licenses mentioned in the resume, such as AWS Certified Solutions Architect, PMP, etc.")