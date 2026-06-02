from fastapi import APIRouter, HTTPException

from app.core.deps import job_repo
from app.models.schemas import JobSpec, JobSpecCreate, JobSpecUpdate

router = APIRouter(prefix="/jobs", tags=["jobs"])


@router.post("", response_model=JobSpec)
def create_job(payload: JobSpecCreate) -> JobSpec:
    return job_repo.create(payload)


@router.get("", response_model=list[JobSpec])
def list_jobs() -> list[JobSpec]:
    return job_repo.list()


@router.get("/{job_id}", response_model=JobSpec)
def get_job(job_id: str) -> JobSpec:
    job = job_repo.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return job


@router.put("/{job_id}", response_model=JobSpec)
def update_job(job_id: str, payload: JobSpecUpdate) -> JobSpec:
    updated = job_repo.update(job_id, payload)
    if not updated:
        raise HTTPException(status_code=404, detail="Job not found")
    return updated
