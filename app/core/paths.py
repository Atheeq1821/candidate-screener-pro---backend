from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]  # backend/
DATA_DIR = PROJECT_ROOT / "data"
UPLOADS_DIR = DATA_DIR / "uploads"
VECTORLESS_RAG_DIR = DATA_DIR / "vectorless_rag"
PDF_EXTRACTION_DIR = PROJECT_ROOT / "pdf_extraction"  # backend/pdf_extraction/
LINKEDIN_PARSING_DIR = PROJECT_ROOT / "linkedin_parsing"  # backend/linkedin_parsing/

# Root-level (standalone script) output dirs — already-scraped data lives here
_REPO_ROOT = PROJECT_ROOT.parent  # candidate_filter/
ROOT_LINKEDIN_PARSING_DIR = _REPO_ROOT / "linkedin_parsing"  # candidate_filter/linkedin_parsing/
ROOT_PDF_EXTRACTION_DIR = _REPO_ROOT / "pdf_extraction"  # candidate_filter/pdf_extraction/
