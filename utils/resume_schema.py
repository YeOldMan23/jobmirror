from pydantic import BaseModel, Field, field_validator
from typing import List, Optional, get_origin, get_args, Union
import datetime

class Experience(BaseModel):
    role: Optional[str] = Field(None, description="The job title or position held")
    company: Optional[str] = Field(None, description="The name of the company. Exclude other description or location")
    date_start: Optional[str] = Field(None, description="The start date of the job. Dates must be in ISO 8601 format (YYYY-MM-DDTHH:MM:SS) or use the keywords 'present', 'current', or 'ongoing'")
    date_end: Optional[str] = Field(None, description="The end date of the job. Dates must be in ISO 8601 format (YYYY-MM-DDTHH:MM:SS) or use the keywords 'present', 'current', or 'ongoing'")
    role_description: Optional[str] = Field(None, description="A description of the responsibilities and achievements in the role")

class Education(BaseModel):
    degree: Optional[str] = Field(None, description="The academic degree obtained")
    institution: Optional[str] = Field(None, description="The name of the educational institution")
    date_start: Optional[str] = Field(None, description="The start date of the education program. Dates must be in ISO 8601 format (YYYY-MM-DDTHH:MM:SS) or use the keywords 'present', 'current', or 'ongoing'")
    date_end: Optional[str] = Field(None, description="The end date of the education program. Dates must be in ISO 8601 format (YYYY-MM-DDTHH:MM:SS) or use the keywords 'present', 'current', or 'ongoing'")
    grade: Optional[float] = Field(None, description="The GPA or final grade, if available")
    description: Optional[str] = Field(None, description="Additional details about the education")

class Resume(BaseModel):
    name: Optional[str] = Field(None, description="Full name of the person")
    location_preference: Optional[str] = Field(None, description="Preference for their work location / remote, if stated")
    work_authorizaton: Optional[str] = Field(None, description="Work authorization that the person holds, such as citizenship, if stated")
    employment_type_preference: Optional[str] = Field(
        None,
        description="Type of employment the resume is looking for such as Full-time, Part-time, Contract, Freelance, or Internship, if stated. It can also be a preference for remote work or on-site work"
    )
    hard_skills: List[str] = Field(default_factory=list, description="A list of hard or technical skills mentioned in the resume. All hard skills are tools, frameworks, or programming languages (e.g., Python, TensorFlow, Docker). Keep it as keywwords. Exclude certification or license")
    soft_skills: List[str] = Field(default_factory=list, description="A list of soft skills mentioned in the resume. Soft skills are qualities like communication, teamwork, leadership. Keep it as keywwords. Exclude required languages")
    languages: List[str]= Field(default_factory=list, description="A list of language proficiencies mentioned in the resume. If the resume does not mention any languages, then fill this with the language that the resume is written in")
    experience: List[Experience] = Field(default_factory=list, description="A list of past work experiences in reverse chronological order (most recent first).")
    education: List[Education] = Field(default_factory=list, description="A list of educational qualifications")
    certifications: List[str] = Field(default_factory=list, description="A list of certifications or licenses related with hard skills, medical skills, and software tools mentioned in the resume. For example, AWS Certified Solutions Architect, PMP, etc. Certifications must exclude any work role IDs, only include valid licenses or certifications.")