from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
UPLOADS_DIR = DATA_DIR / "uploads"
PDF_EXTRACTION_DIR = PROJECT_ROOT / "pdf_extraction"
LINKEDIN_PARSING_DIR = PROJECT_ROOT / "linkedin_parsing"
