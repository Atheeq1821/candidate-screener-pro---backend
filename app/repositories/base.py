from typing import Protocol

from app.models.schemas import CandidateRun, JobSpec, JobSpecCreate, JobSpecUpdate


class JobRepository(Protocol):
    def create(self, payload: JobSpecCreate) -> JobSpec:
        ...

    def list(self) -> list[JobSpec]:
        ...

    def get(self, job_id: str) -> JobSpec | None:
        ...

    def update(self, job_id: str, payload: JobSpecUpdate) -> JobSpec | None:
        ...


class RunRepository(Protocol):
    def create(self, run: CandidateRun) -> CandidateRun:
        ...

    def get(self, run_id: str) -> CandidateRun | None:
        ...

    def update(self, run_id: str, run: CandidateRun) -> CandidateRun | None:
        ...
