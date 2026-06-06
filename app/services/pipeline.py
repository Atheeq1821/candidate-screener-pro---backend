import json
import logging
import os
import re
import shutil
from pathlib import Path
from typing import Any, Dict, Optional

from app.core.paths import LINKEDIN_PARSING_DIR, PDF_EXTRACTION_DIR, ROOT_LINKEDIN_PARSING_DIR, UPLOADS_DIR
from app.models.schemas import JobSpec
from app.parsers.extract_resume_pdf import extract_pdf, _parse_with_groq
from app.parsers.parse_linkedin_linkdapi import parse_linkedin_profile
from app.services.vectorless_engine import run_role_fit
from app.services.vectorless_profile import build_vectorless_profile, persist_vectorless_profile


def _slug_from_linkedin_url(url: str) -> str:
    return url.strip().rstrip("/").split("/")[-1]


def _simple_resume_parse(extracted: Dict[str, Any]) -> Dict[str, Any]:
    text = extracted.get("combined_text", "")
    skills_text = extracted.get("section_slices", {}).get("skills", "")
    skill_candidates = re.split(r"[,:\n]", skills_text)
    skills = sorted({token.strip() for token in skill_candidates if len(token.strip()) > 1 and len(token.strip()) < 40})
    return {
        "name": "",
        "headline": "",
        "about": extracted.get("section_slices", {}).get("summary", ""),
        "skills": skills,
        "experience": [],
        "education": [],
        "certifications": [],
        "projects": [],
        "raw_text": text,
    }


import logging

logger = logging.getLogger(__name__)


def _ensure_resume_parsed(
    resume_path: Path,
    project_root: Path,
    parsed_dir: Path,
    extraction_dir: Path,
) -> Dict[str, Any]:
    resume_stem = resume_path.stem
    parsed_resume_path = parsed_dir / f"{resume_stem}_parsed.json"
    if parsed_resume_path.exists():
        return json.loads(parsed_resume_path.read_text(encoding="utf-8"))

    extracted = extract_pdf(resume_path)

    extraction_json = extraction_dir / f"{resume_stem}.json"
    extraction_json.write_text(json.dumps(extracted.__dict__, indent=2, ensure_ascii=True), encoding="utf-8")

    groq_api = os.getenv("GROQ_API")
    if groq_api:
        try:
            parsed = _parse_with_groq(extracted.combined_text, "llama-3.3-70b-versatile")
            # Only cache if Groq returned meaningful data (has name or experience)
            if parsed.get("name") or parsed.get("experience") or parsed.get("education"):
                parsed_resume_path.write_text(json.dumps(parsed, indent=2, ensure_ascii=True), encoding="utf-8")
            return parsed
        except Exception as exc:
            logger.warning("Groq parsing failed for %s: %s — falling back to simple parser", resume_path.name, exc)
    else:
        logger.warning("GROQ_API not set — using simple regex parser. Set GROQ_API in backend/.env for full parsing.")

    # Simple fallback — NOT cached so the next upload attempt retries Groq
    return _simple_resume_parse(extracted.__dict__)



def _ensure_linkedin_parsed(
    linkedin_url: str,
    project_root: Path,
) -> Dict[str, Any]:
    slug = _slug_from_linkedin_url(linkedin_url)
    backend_output_dir = LINKEDIN_PARSING_DIR / "output"
    root_output_dir = ROOT_LINKEDIN_PARSING_DIR / "output"

    # 1. Backend cache (written by this service on previous successful runs)
    backend_path = backend_output_dir / f"{slug}_linkedin_parsed.json"
    if backend_path.exists():
        logger.info("[LinkedIn] Cache hit (backend) — %s", backend_path)
        return json.loads(backend_path.read_text(encoding="utf-8"))

    # 2. Root standalone-script cache (scraped by linkedin_parsing/parse_linkedin_apify.py CLI)
    root_path = root_output_dir / f"{slug}_linkedin_parsed.json"
    if root_path.exists():
        logger.info("[LinkedIn] Cache hit (root/standalone) — %s", root_path)
        data = json.loads(root_path.read_text(encoding="utf-8"))
        # Promote to backend cache so future lookups stay local
        backend_output_dir.mkdir(parents=True, exist_ok=True)
        backend_path.write_text(json.dumps(data, indent=2, ensure_ascii=True), encoding="utf-8")
        logger.info("[LinkedIn] Promoted root cache → backend cache: %s", backend_path)
        return data

    # 3. Live Apify call
    logger.info("[LinkedIn] No cache found for slug=%r — calling Apify", slug)

    linkdapi_key = (
        os.getenv("LINKDAPI_KEY")
        or os.getenv("LinkdAPI_KEY")
        or os.getenv("LINKDAPI_API_KEY")
    )
    if not linkdapi_key:
        raise ValueError("LINKDAPI_KEY not set — cannot parse LinkedIn profile")

    try:
        output_paths = parse_linkedin_profile(
            linkedin_url=linkedin_url,
            output_dir=backend_output_dir,
            token=linkdapi_key,
            timeout_seconds=30,
        )
        formatted = output_paths["formatted"]
        if Path(formatted).exists():
            logger.info("[LinkedIn] LinkdAPI parse successful — reading %s", formatted)
            return json.loads(Path(formatted).read_text(encoding="utf-8"))
        else:
            logger.warning("[LinkedIn] LinkdAPI returned no cacheable content for %s", linkedin_url)
            return {"identity": {"linkedin_url": linkedin_url}, "experience": [], "education": [], "skills": [], "certifications": [], "projects": []}
    except Exception as exc:
        logger.error("[LinkedIn] LinkdAPI parse failed for %s: %s", linkedin_url, exc)
        return {"identity": {"linkedin_url": linkedin_url}, "experience": [], "education": [], "skills": [], "certifications": [], "projects": []}




def process_candidate(
    resume_path: Path,
    linkedin_url: Optional[str],
    role: JobSpec,
    project_root: Path,
    run_id: Optional[str] = None,
) -> tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    parsed_dir = PDF_EXTRACTION_DIR / "output" / "parsed"
    extraction_dir = PDF_EXTRACTION_DIR / "output"
    parsed_dir.mkdir(parents=True, exist_ok=True)
    extraction_dir.mkdir(parents=True, exist_ok=True)

    resume_parsed = _ensure_resume_parsed(resume_path, project_root, parsed_dir, extraction_dir)

    linkedin_parsed: Dict[str, Any] = {"identity": {}, "experience": [], "education": [], "skills": [], "certifications": [], "projects": []}
    if linkedin_url:
        linkedin_parsed = _ensure_linkedin_parsed(linkedin_url, project_root)

    vectorless_rag = build_vectorless_profile(resume_parsed, linkedin_parsed)
    vectorless_rag_path = persist_vectorless_profile(run_id or resume_path.stem, vectorless_rag)
    fit_result, chunks = run_role_fit(role, resume_parsed, linkedin_parsed)
    analytics = {
        "fit_result": fit_result.model_dump(),
        "vectorless_rag": vectorless_rag,
        "vectorless_rag_path": str(vectorless_rag_path),
        "charts": {
            "scores": [
                {"name": "Overall", "value": fit_result.overall_score},
                {"name": "Skills", "value": fit_result.skills_fit_score},
                {"name": "Experience", "value": fit_result.experience_fit_score},
                {"name": "Joiner", "value": fit_result.joiner_fit_score},
            ],
            "skills": {
                "matched": fit_result.skills_fit_score,
                "missing": round(100 - fit_result.skills_fit_score, 2),
            },
        },
        "evidence": [{"source": c.source, "path": c.path, "text": c.text[:220]} for c in chunks[:20]],
    }
    return resume_parsed, linkedin_parsed, analytics


def save_uploaded_resume(src_temp_path: Path, filename: str, project_root: Path) -> Path:
    UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
    target = UPLOADS_DIR / filename
    shutil.copy2(src_temp_path, target)
    return target
