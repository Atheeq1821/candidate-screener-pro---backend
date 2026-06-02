import json
import importlib.util
import os
import re
import shutil
from pathlib import Path
from typing import Any, Dict, Optional

from app.models.schemas import JobSpec
from app.services.vectorless_engine import run_role_fit


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


def _load_module_from_path(module_path: Path, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load module: {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


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

    extraction_module_path = project_root / "pdf_extraction" / "extract_resume_pdf.py"
    module = _load_module_from_path(extraction_module_path, "extract_resume_pdf_module")
    extract_pdf = module.extract_pdf
    extracted = extract_pdf(resume_path)

    extraction_json = extraction_dir / f"{resume_stem}.json"
    extraction_json.write_text(json.dumps(extracted.__dict__, indent=2, ensure_ascii=True), encoding="utf-8")

    groq_api = os.getenv("GROQ_API")
    if groq_api and hasattr(module, "_parse_with_groq"):
        try:
            parsed = module._parse_with_groq(extracted.combined_text, "llama-3.3-70b-versatile")
            parsed_resume_path.write_text(json.dumps(parsed, indent=2, ensure_ascii=True), encoding="utf-8")
            return parsed
        except Exception:
            # fall back to deterministic parser below
            pass

    parsed = _simple_resume_parse(extracted.__dict__)
    parsed_resume_path.write_text(json.dumps(parsed, indent=2, ensure_ascii=True), encoding="utf-8")
    return parsed


def _ensure_linkedin_parsed(
    linkedin_url: str,
    project_root: Path,
) -> Dict[str, Any]:
    slug = _slug_from_linkedin_url(linkedin_url)
    output_dir = project_root / "linkedin_parsing" / "output"
    formatted_path = output_dir / f"{slug}_linkedin_parsed.json"
    if formatted_path.exists():
        return json.loads(formatted_path.read_text(encoding="utf-8"))

    module_path = project_root / "linkedin_parsing" / "parse_linkedin_apify.py"
    module = _load_module_from_path(module_path, "parse_linkedin_apify_module")
    apify_token = os.getenv("APIFY_API_TOKEN")
    if not apify_token:
        return {"identity": {"linkedin_url": linkedin_url}, "experience": [], "education": [], "skills": [], "certifications": [], "projects": []}

    actor_id = os.getenv("APIFY_LINKEDIN_ACTOR_ID", "supreme_coder/linkedin-profile-scraper")
    proxy_country = os.getenv("APIFY_PROXY_COUNTRY", "US")
    linkedin_cookie = os.getenv("LINKEDIN_COOKIE")
    try:
        output_paths = module.parse_linkedin_profile(
            linkedin_url=linkedin_url,
            output_dir=output_dir,
            actor_id=actor_id,
            token=apify_token,
            poll_interval_seconds=5,
            timeout_seconds=240,
            linkedin_cookie=linkedin_cookie,
            proxy_country=proxy_country,
        )
        formatted = output_paths["formatted"]
        return json.loads(Path(formatted).read_text(encoding="utf-8"))
    except Exception:
        return {"identity": {"linkedin_url": linkedin_url}, "experience": [], "education": [], "skills": [], "certifications": [], "projects": []}


def process_candidate(
    resume_path: Path,
    linkedin_url: Optional[str],
    role: JobSpec,
    project_root: Path,
) -> tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    parsed_dir = project_root / "pdf_extraction" / "output" / "parsed"
    extraction_dir = project_root / "pdf_extraction" / "output"
    parsed_dir.mkdir(parents=True, exist_ok=True)
    extraction_dir.mkdir(parents=True, exist_ok=True)

    resume_parsed = _ensure_resume_parsed(resume_path, project_root, parsed_dir, extraction_dir)

    linkedin_parsed: Dict[str, Any] = {"identity": {}, "experience": [], "education": [], "skills": [], "certifications": [], "projects": []}
    if linkedin_url:
        linkedin_parsed = _ensure_linkedin_parsed(linkedin_url, project_root)

    fit_result, chunks = run_role_fit(role, resume_parsed, linkedin_parsed)
    analytics = {
        "fit_result": fit_result.model_dump(),
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
    target_dir = project_root / "backend" / "data" / "uploads"
    target_dir.mkdir(parents=True, exist_ok=True)
    target = target_dir / filename
    shutil.copy2(src_temp_path, target)
    return target
