from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from pathlib import Path
import logging
import shutil

from app.api.routes.candidates import router as candidates_router
from app.api.routes.chat import router as chat_router
from app.api.routes.jobs import router as jobs_router
from app.core.paths import LINKEDIN_PARSING_DIR, PDF_EXTRACTION_DIR, ROOT_LINKEDIN_PARSING_DIR, ROOT_PDF_EXTRACTION_DIR
from app.services.vectorless_profile import clear_vectorless_runtime

# Load env from backend root .env (contains GROQ_API, LINKDAPI_KEY, etc.)
_BACKEND_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(_BACKEND_ROOT / ".env", override=False)

# Reset ephemeral vectorless RAG logs on startup so each server run starts clean.
clear_vectorless_runtime()

# Enable INFO-level logging so pipeline/parser logs are visible in uvicorn output
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

app = FastAPI(title="Candidate Filter API", version="0.1.0")


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/clear-cache")
def clear_cache() -> dict[str, str]:
    # Clear PDF extraction cache
    for d in [PDF_EXTRACTION_DIR, ROOT_PDF_EXTRACTION_DIR]:
        out = d / "output"
        if out.exists():
            shutil.rmtree(out, ignore_errors=True)
            out.mkdir(parents=True, exist_ok=True)
            
    # Clear LinkedIn parsing cache
    for d in [LINKEDIN_PARSING_DIR, ROOT_LINKEDIN_PARSING_DIR]:
        out = d / "output"
        if out.exists():
            shutil.rmtree(out, ignore_errors=True)
            out.mkdir(parents=True, exist_ok=True)

    clear_vectorless_runtime()
        
    logging.info("All caches (backend and root) cleared successfully via /clear-cache endpoint.")
    return {"status": "caches_cleared"}


app.include_router(jobs_router)
app.include_router(candidates_router)
app.include_router(chat_router)
