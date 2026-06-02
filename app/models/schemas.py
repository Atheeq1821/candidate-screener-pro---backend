from datetime import datetime
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field


class JobSpecBase(BaseModel):
    role_name: str
    experience_needed: str
    immediate_joiner: int = Field(ge=0)
    skills_required: List[str]
    job_description: str


class JobSpecCreate(JobSpecBase):
    pass


class JobSpecUpdate(JobSpecBase):
    pass


class JobSpec(JobSpecBase):
    id: str
    created_at: datetime
    updated_at: datetime


class CandidateProcessResponse(BaseModel):
    run_id: str
    status: Literal["queued", "processing", "completed", "failed"]


class FitResult(BaseModel):
    role_name: str
    overall_score: float
    skills_fit_score: float
    experience_fit_score: float
    joiner_fit_score: float
    top_alignments: List[str]
    top_gaps: List[str]
    notes: str
    matched_skills: List[str] = Field(default_factory=list)
    missing_skills: List[str] = Field(default_factory=list)


class CandidateRun(BaseModel):
    run_id: str
    status: Literal["queued", "processing", "completed", "failed"]
    job_id: str
    resume_file: str
    linkedin_url: Optional[str] = None
    error: Optional[str] = None
    fit_result: Optional[FitResult] = None
    resume_parsed: Optional[Dict[str, Any]] = None
    linkedin_parsed: Optional[Dict[str, Any]] = None
    created_at: datetime
    updated_at: datetime


class ChatRequest(BaseModel):
    run_id: str
    question: str


class ChatResponse(BaseModel):
    answer: str
    evidence: List[Dict[str, str]]
