from datetime import datetime
from pathlib import Path
from uuid import uuid4

from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from app.core.deps import job_repo, run_repo
from app.models.schemas import CandidateProcessResponse, CandidateRun
from app.services.pipeline import process_candidate, save_uploaded_resume

router = APIRouter(prefix="/candidates", tags=["candidates"])


@router.post("/process", response_model=CandidateProcessResponse)
async def process_candidate_endpoint(
    resume: UploadFile = File(...),
    job_id: str = Form(...),
    linkedin_url: str | None = Form(default=None),
) -> CandidateProcessResponse:
    job = job_repo.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    temp_path = Path("backend") / "data" / "uploads" / f"tmp_{uuid4().hex}_{resume.filename}"
    temp_path.parent.mkdir(parents=True, exist_ok=True)
    content = await resume.read()
    temp_path.write_bytes(content)

    run_id = f"run_{uuid4().hex[:10]}"
    now = datetime.utcnow()
    run = CandidateRun(
        run_id=run_id,
        status="processing",
        job_id=job_id,
        resume_file=resume.filename,
        linkedin_url=linkedin_url,
        created_at=now,
        updated_at=now,
    )
    run_repo.create(run)

    try:
        project_root = Path(__file__).resolve().parents[4]
        saved_resume = save_uploaded_resume(temp_path, resume.filename, project_root)
        resume_parsed, linkedin_parsed, analytics = process_candidate(saved_resume, linkedin_url, job, project_root)
        updated_payload = run.model_dump()
        updated_payload["status"] = "completed"
        updated_payload["updated_at"] = datetime.utcnow()
        updated_payload["fit_result"] = analytics["fit_result"]
        updated_payload["resume_parsed"] = resume_parsed
        updated_payload["linkedin_parsed"] = linkedin_parsed
        updated = CandidateRun(**updated_payload)
        run_repo.update(run_id, updated)
    except Exception as exc:
        failed_payload = run.model_dump()
        failed_payload["status"] = "failed"
        failed_payload["error"] = str(exc)
        failed_payload["updated_at"] = datetime.utcnow()
        failed = CandidateRun(**failed_payload)
        run_repo.update(run_id, failed)
    finally:
        if temp_path.exists():
            temp_path.unlink(missing_ok=True)

    current = run_repo.get(run_id)
    return CandidateProcessResponse(run_id=run_id, status=current.status if current else "failed")


@router.get("/runs/{run_id}", response_model=CandidateRun)
def get_run(run_id: str) -> CandidateRun:
    run = run_repo.get(run_id)
    if not run:
        raise HTTPException(status_code=404, detail="Run not found")
    return run
