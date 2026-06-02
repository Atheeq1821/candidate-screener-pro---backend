from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from pathlib import Path

from app.api.routes.candidates import router as candidates_router
from app.api.routes.chat import router as chat_router
from app.api.routes.jobs import router as jobs_router

project_root = Path(__file__).resolve().parents[2]
load_dotenv(project_root / "pdf_extraction" / ".env")

app = FastAPI(title="Candidate Filter API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


app.include_router(jobs_router)
app.include_router(candidates_router)
app.include_router(chat_router)
