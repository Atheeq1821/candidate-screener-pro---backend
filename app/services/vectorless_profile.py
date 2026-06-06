import json
import re
import shutil
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

from app.core.paths import VECTORLESS_RAG_DIR

VECTORLESS_PROFILE_FILENAME = "latest_vectorless_profile.json"
VECTORLESS_TRACE_FILENAME = "retrieval_trace.jsonl"


def _normalize_text(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    return " ".join(text.split())


def _normalize_key(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "", _normalize_text(value).lower())


def _normalize_skill(skill: Any) -> str:
    return _normalize_text(skill).lower()


_MONTH_ABBR = ["", "Jan", "Feb", "Mar", "Apr", "May", "Jun",
               "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


def _normalize_date_value(value: Any) -> Any:
    if not isinstance(value, dict):
        return value
    year = value.get("year")
    month = value.get("month")
    if year and month:
        try:
            m = int(month)
            name = _MONTH_ABBR[m] if 1 <= m <= 12 else str(m)
            return f"{name} {year}"
        except (ValueError, IndexError):
            pass
    if year:
        return str(year)
    return value


def _pick_first(*values: Any) -> Any:
    for value in values:
        if value not in (None, "", [], {}):
            return value
    return ""


def _merge_scalar(primary: Dict[str, Any], secondary: Dict[str, Any], key: str) -> None:
    if not primary.get(key):
        value = secondary.get(key)
        if value not in (None, "", [], {}):
            primary[key] = value


def _merge_contact(primary: Dict[str, Any], secondary: Dict[str, Any]) -> Dict[str, Any]:
    merged = deepcopy(primary) if primary else {}
    secondary = secondary or {}
    for key in ("email", "phone", "linkedin", "location"):
        _merge_scalar(merged, secondary, key)
    return merged


def _merge_identity(resume: Dict[str, Any], linkedin: Dict[str, Any]) -> Dict[str, Any]:
    resume_contact = resume.get("contact") if isinstance(resume.get("contact"), dict) else {}
    linkedin_identity = linkedin.get("identity") if isinstance(linkedin.get("identity"), dict) else {}

    identity = {
        "name": _pick_first(resume.get("name"), linkedin_identity.get("name")),
        "headline": _pick_first(resume.get("headline"), linkedin_identity.get("headline")),
        "about": _pick_first(resume.get("about"), linkedin_identity.get("about")),
        "contact": _merge_contact(resume_contact, linkedin_identity),
    }
    return identity


def _merge_list_of_dicts(
    primary: Iterable[Dict[str, Any]],
    secondary: Iterable[Dict[str, Any]],
    key_fields: Tuple[str, ...],
    field_order: Tuple[str, ...],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    merged: List[Dict[str, Any]] = []
    provenance: List[Dict[str, Any]] = []
    index: Dict[str, int] = {}

    def signature(item: Dict[str, Any]) -> str:
        parts = [_normalize_key(item.get(field, "")) for field in key_fields]
        return "|".join(parts)

    def append_item(item: Dict[str, Any], source: str) -> None:
        normalized = {field: _normalize_date_value(item.get(field, "")) for field in field_order}
        for field in field_order:
            if normalized[field] in (None, ""):
                normalized[field] = ""
        merged.append(normalized)
        provenance.append({"source": source, "signature": signature(item)})
        index[signature(item)] = len(merged) - 1

    for item in primary:
        if not isinstance(item, dict):
            continue
        append_item(item, "resume")

    for item in secondary:
        if not isinstance(item, dict):
            continue
        sig = signature(item)
        if not sig.strip("|"):
            continue
        if sig in index:
            existing = merged[index[sig]]
            for field in field_order:
                li_val = _normalize_date_value(item.get(field, ""))
                if li_val in (None, ""):
                    continue
                resume_val = existing.get(field, "")
                if not resume_val:
                    existing[field] = li_val
                elif _normalize_key(str(resume_val)) != _normalize_key(str(li_val)):
                    # Both sources have different values — surface the conflict so the LLM can flag it.
                    existing[field] = f"{resume_val} [resume] / {li_val} [linkedin]"
        else:
            append_item(item, "linkedin")

    return merged, provenance


def _merge_skills(resume_skills: Iterable[Any], linkedin_skills: Iterable[Any]) -> Tuple[List[str], Dict[str, List[str]]]:
    merged: List[str] = []
    provenance = {"resume": [], "linkedin": []}
    seen: Dict[str, str] = {}

    for source, skills in (("resume", resume_skills), ("linkedin", linkedin_skills)):
        for skill in skills:
            normalized = _normalize_skill(skill)
            if not normalized:
                continue
            if normalized not in seen:
                seen[normalized] = source
                merged.append(str(skill).strip())
                provenance[source].append(str(skill).strip())

    return merged, provenance


def _normalize_linkedin_experience(item: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize LinkedIn experience field names and date formats to match the merged schema."""
    out = dict(item)
    out["start_date"] = _normalize_date_value(out.get("start_date", ""))
    out["end_date"] = _normalize_date_value(out.get("end_date", ""))
    return out


def _normalize_linkedin_education(item: Dict[str, Any]) -> Dict[str, Any]:
    """
    LinkedIn education uses 'school' / 'grade' / 'field_of_study'.
    Map them to 'institution' / 'score' so deduplication and field merging work.
    """
    out = dict(item)
    if "school" in out and not out.get("institution"):
        out["institution"] = out.pop("school")
    else:
        out.pop("school", None)
    if "grade" in out and not out.get("score"):
        out["score"] = out.pop("grade")
    else:
        out.pop("grade", None)
    out.pop("school_url", None)
    out.pop("description", None)
    return out


def _normalize_linkedin_certification(item: Dict[str, Any]) -> Dict[str, Any]:
    """LinkedIn certifications use 'issue_date' instead of 'date'."""
    out = dict(item)
    if "issue_date" in out and not out.get("date"):
        out["date"] = out.pop("issue_date")
    else:
        out.pop("issue_date", None)
    out.pop("expiry_date", None)
    return out


def _merge_experience(resume: Dict[str, Any], linkedin: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    resume_items = resume.get("experience") or []
    raw_linkedin = linkedin.get("experience") or []
    linkedin_items = [
        _normalize_linkedin_experience(i) for i in raw_linkedin if isinstance(i, dict)
    ]
    field_order = ("title", "company", "location", "employment_type", "start_date", "end_date", "description")
    merged, provenance = _merge_list_of_dicts(
        primary=resume_items if isinstance(resume_items, list) else [],
        secondary=linkedin_items,
        key_fields=("company", "title", "start_date", "end_date"),
        field_order=field_order,
    )
    return merged, provenance


def _merge_education(resume: Dict[str, Any], linkedin: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    resume_items = resume.get("education") or []
    raw_linkedin = linkedin.get("education") or []
    linkedin_items = [
        _normalize_linkedin_education(i) for i in raw_linkedin if isinstance(i, dict)
    ]
    field_order = ("degree", "institution", "field_of_study", "start_date", "end_date", "score")
    merged, provenance = _merge_list_of_dicts(
        primary=resume_items if isinstance(resume_items, list) else [],
        secondary=linkedin_items,
        key_fields=("degree", "institution"),
        field_order=field_order,
    )
    return merged, provenance


def _merge_certifications(resume: Dict[str, Any], linkedin: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    resume_items = resume.get("certifications") or []
    raw_linkedin = linkedin.get("certifications") or []
    linkedin_items = [
        _normalize_linkedin_certification(i) for i in raw_linkedin if isinstance(i, dict)
    ]
    field_order = ("name", "issuer", "date", "url")
    merged, provenance = _merge_list_of_dicts(
        primary=resume_items if isinstance(resume_items, list) else [],
        secondary=linkedin_items,
        key_fields=("name", "issuer"),
        field_order=field_order,
    )
    return merged, provenance


def _merge_projects(resume: Dict[str, Any], linkedin: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    resume_items = resume.get("projects", [])
    linkedin_items = linkedin.get("projects", [])
    field_order = ("name", "description", "tools")
    merged, provenance = _merge_list_of_dicts(
        primary=resume_items if isinstance(resume_items, list) else [],
        secondary=linkedin_items if isinstance(linkedin_items, list) else [],
        key_fields=("name", "description", "tools"),
        field_order=field_order,
    )
    return merged, provenance


def build_vectorless_profile(resume_parsed: Dict[str, Any], linkedin_parsed: Dict[str, Any]) -> Dict[str, Any]:
    """
    Build a resume-first merged profile for vectorless retrieval.

    Resume is the primary source. LinkedIn only fills gaps or adds missing
    unique items. The returned structure is intentionally retrieval-friendly
    and separate from provenance so chunking stays clean.
    """
    resume_parsed = resume_parsed or {}
    linkedin_parsed = linkedin_parsed or {}

    linkedin_identity = linkedin_parsed.get("identity") if isinstance(linkedin_parsed.get("identity"), dict) else {}

    identity = _merge_identity(resume_parsed, linkedin_parsed)
    experience, experience_sources = _merge_experience(resume_parsed, linkedin_parsed)
    education, education_sources = _merge_education(resume_parsed, linkedin_parsed)
    certifications, certification_sources = _merge_certifications(resume_parsed, linkedin_parsed)
    projects, project_sources = _merge_projects(resume_parsed, linkedin_parsed)
    skills, skill_sources = _merge_skills(
        resume_parsed.get("skills", []) if isinstance(resume_parsed.get("skills", []), list) else [],
        linkedin_parsed.get("skills", []) if isinstance(linkedin_parsed.get("skills", []), list) else [],
    )

    li_meta = linkedin_parsed.get("meta") if isinstance(linkedin_parsed.get("meta"), dict) else {}

    profile = {
        "name": identity.get("name", ""),
        "headline": identity.get("headline", ""),
        # LinkedIn headline often carries signals the resume omits (e.g. "Immediate Joiner", notice period).
        "linkedin_headline": linkedin_identity.get("headline", ""),
        # LinkedIn about may contain richer context than the resume summary.
        "linkedin_about": linkedin_identity.get("about", ""),
        "current_company": linkedin_identity.get("current_company", ""),
        "location": linkedin_identity.get("location", "") or (
            identity.get("contact", {}).get("location", "") if isinstance(identity.get("contact"), dict) else ""
        ),
        "about": identity.get("about", ""),
        "contact": identity.get("contact", {}),
        "linkedin_url": _pick_first(
            resume_parsed.get("contact", {}).get("linkedin") if isinstance(resume_parsed.get("contact"), dict) else "",
            linkedin_identity.get("linkedin_url"),
        ),
        "connections": li_meta.get("connections", ""),
        "followers": li_meta.get("followers", ""),
        "skills": skills,
        "experience": experience,
        "education": education,
        "certifications": certifications,
        "projects": projects,
    }
    # Drop empty top-level scalar fields so they don't produce blank chunks.
    profile = {k: v for k, v in profile.items() if v not in ("", None, [], {})}

    provenance = {
        "resume_keys": sorted([key for key, value in resume_parsed.items() if value not in (None, "", [], {})]),
        "linkedin_keys": sorted([key for key, value in linkedin_parsed.items() if value not in (None, "", [], {})]),
        "skills": skill_sources,
        "experience": experience_sources,
        "education": education_sources,
        "certifications": certification_sources,
        "projects": project_sources,
    }

    source_summary = {
        "resume_experience_count": len(resume_parsed.get("experience", []) or []),
        "linkedin_experience_count": len(linkedin_parsed.get("experience", []) or []),
        "merged_experience_count": len(experience),
        "resume_skill_count": len(resume_parsed.get("skills", []) or []),
        "linkedin_skill_count": len(linkedin_parsed.get("skills", []) or []),
        "merged_skill_count": len(skills),
    }

    return {
        "candidate_profile": profile,
        "provenance": provenance,
        "source_summary": source_summary,
    }


def persist_vectorless_profile(run_id: str, vectorless_profile: Dict[str, Any]) -> Path:
    VECTORLESS_RAG_DIR.mkdir(parents=True, exist_ok=True)
    path = VECTORLESS_RAG_DIR / f"{run_id}_{VECTORLESS_PROFILE_FILENAME}"
    payload = {
        "run_id": run_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "vectorless_profile": vectorless_profile,
    }
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")
    return path


def append_retrieval_trace(
    run_id: str,
    question: str,
    retrieved_chunks: List[Dict[str, Any]],
    vectorless_profile: Dict[str, Any] | None = None,
) -> Path:
    VECTORLESS_RAG_DIR.mkdir(parents=True, exist_ok=True)
    path = VECTORLESS_RAG_DIR / VECTORLESS_TRACE_FILENAME
    payload = {
        "run_id": run_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "question": question,
        "retrieved_chunks": retrieved_chunks,
    }
    if vectorless_profile is not None:
        payload["vectorless_profile"] = vectorless_profile
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=True) + "\n")
    return path


def clear_vectorless_runtime() -> None:
    if VECTORLESS_RAG_DIR.exists():
        shutil.rmtree(VECTORLESS_RAG_DIR, ignore_errors=True)
    VECTORLESS_RAG_DIR.mkdir(parents=True, exist_ok=True)
