from datetime import datetime
from typing import Dict
from uuid import uuid4

from app.models.schemas import CandidateRun, JobSpec, JobSpecCreate, JobSpecUpdate


class InMemoryJobRepository:
    def __init__(self) -> None:
        self._jobs: Dict[str, JobSpec] = {}

    def create(self, payload: JobSpecCreate) -> JobSpec:
        now = datetime.utcnow()
        job = JobSpec(id=f"job_{uuid4().hex[:10]}", created_at=now, updated_at=now, **payload.model_dump())
        self._jobs[job.id] = job
        return job

    def list(self) -> list[JobSpec]:
        return sorted(self._jobs.values(), key=lambda item: item.updated_at, reverse=True)

    def get(self, job_id: str) -> JobSpec | None:
        return self._jobs.get(job_id)

    def update(self, job_id: str, payload: JobSpecUpdate) -> JobSpec | None:
        existing = self._jobs.get(job_id)
        if not existing:
            return None
        updated = JobSpec(
            id=existing.id,
            created_at=existing.created_at,
            updated_at=datetime.utcnow(),
            **payload.model_dump(),
        )
        self._jobs[job_id] = updated
        return updated


class InMemoryRunRepository:
    def __init__(self) -> None:
        self._runs: Dict[str, CandidateRun] = {}

    def create(self, run: CandidateRun) -> CandidateRun:
        self._runs[run.run_id] = run
        return run

    def get(self, run_id: str) -> CandidateRun | None:
        return self._runs.get(run_id)

    def update(self, run_id: str, run: CandidateRun) -> CandidateRun | None:
        if run_id not in self._runs:
            return None
        self._runs[run_id] = run
        return run
