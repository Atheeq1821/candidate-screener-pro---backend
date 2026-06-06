import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

import httpx

logger = logging.getLogger(__name__)

LINKDAPI_BASE_URL = "https://linkdapi.com"


def _validate_linkedin_url(url: str) -> str:
    value = url.strip()
    if "linkedin.com/in/" not in value and "linkedin.com/pub/" not in value:
        raise ValueError(f"Not a valid LinkedIn profile URL: {url}")
    return value


def _extract_username(linkedin_url: str) -> str:
    parsed = urlparse(linkedin_url)
    parts = [part for part in parsed.path.split("/") if part]
    if not parts:
        raise ValueError(f"Could not extract username from LinkedIn URL: {linkedin_url}")
    return parts[-1]


def _pick_first(*values: Any) -> Any:
    for value in values:
        if value not in (None, "", [], {}):
            return value
    return ""


def _to_list(value: Any) -> List[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return [item for item in value if item not in (None, "", {})]
    return [] if value in (None, "", {}) else [value]


def _build_name(profile: Dict[str, Any]) -> str:
    first = str(_pick_first(profile.get("firstName"), profile.get("multiLocaleFirstName", {}).get("en_US") if isinstance(profile.get("multiLocaleFirstName"), dict) else "")).strip()
    last = str(_pick_first(profile.get("lastName"), profile.get("multiLocaleLastName", {}).get("en_US") if isinstance(profile.get("multiLocaleLastName"), dict) else "")).strip()
    joined = f"{first} {last}".strip()
    if joined:
        return " ".join(joined.split())
    return str(_pick_first(profile.get("fullName"), profile.get("name"), profile.get("publicIdentifier"))).strip()


def _normalize_positions(items: List[Any]) -> List[Dict[str, Any]]:
    result: List[Dict[str, Any]] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        company = item.get("company") if isinstance(item.get("company"), dict) else {}
        start = item.get("start") if isinstance(item.get("start"), dict) else {}
        end = item.get("end") if isinstance(item.get("end"), dict) else {}
        result.append(
            {
                "company": _pick_first(
                    item.get("companyName"),
                    company.get("name"),
                    item.get("multiLocaleCompanyName", {}).get("en_US") if isinstance(item.get("multiLocaleCompanyName"), dict) else "",
                ),
                "company_url": _pick_first(item.get("companyURL"), company.get("url")),
                "title": _pick_first(item.get("title"), item.get("multiLocaleTitle", {}).get("en_US") if isinstance(item.get("multiLocaleTitle"), dict) else ""),
                "location": _pick_first(item.get("location"), item.get("locationName")),
                "start_date": {
                    "year": start.get("year"),
                    "month": start.get("month"),
                    "day": start.get("day"),
                } if start else item.get("startDate"),
                "end_date": {
                    "year": end.get("year"),
                    "month": end.get("month"),
                    "day": end.get("day"),
                } if end else item.get("endDate"),
                "description": _pick_first(item.get("description"), ""),
                "employment_type": _pick_first(item.get("employmentType"), ""),
                "is_current": not bool(end),
            }
        )
    return result


def _normalize_education(items: List[Any]) -> List[Dict[str, Any]]:
    result: List[Dict[str, Any]] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        time_period = item.get("timePeriod") if isinstance(item.get("timePeriod"), dict) else {}
        result.append(
            {
                "school": _pick_first(item.get("schoolName"), item.get("school")),
                "school_url": _pick_first(item.get("schoolUrl"), item.get("schoolURL")),
                "degree": _pick_first(item.get("degreeName"), item.get("degree")),
                "field_of_study": _pick_first(item.get("fieldOfStudy"), item.get("field_of_study")),
                "start_date": _pick_first(
                    time_period.get("startDate"),
                    item.get("startDate"),
                ),
                "end_date": _pick_first(
                    time_period.get("endDate"),
                    item.get("endDate"),
                ),
                "grade": _pick_first(item.get("grade"), ""),
                "description": _pick_first(item.get("description"), ""),
            }
        )
    return result


def _normalize_skills(items: List[Any]) -> List[str]:
    seen: set[str] = set()
    result: List[str] = []
    for item in items:
        if isinstance(item, dict):
            name = str(item.get("name", "")).strip()
        else:
            name = str(item).strip()
        key = name.lower()
        if name and key not in seen:
            seen.add(key)
            result.append(name)
    return result


def _normalize_certifications(items: List[Any]) -> List[Dict[str, Any]]:
    result: List[Dict[str, Any]] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        time_period = item.get("timePeriod") if isinstance(item.get("timePeriod"), dict) else {}
        result.append(
            {
                "name": _pick_first(item.get("name"), item.get("title")),
                "issuer": _pick_first(item.get("issuer"), item.get("authority")),
                "issue_date": _pick_first(item.get("issueDate"), time_period.get("startDate")),
                "expiry_date": _pick_first(time_period.get("endDate"), item.get("expiryDate")),
                "url": _pick_first(item.get("url"), ""),
            }
        )
    return result


def _normalize_projects(items: List[Any]) -> List[Dict[str, Any]]:
    result: List[Dict[str, Any]] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        time_period = item.get("timePeriod") if isinstance(item.get("timePeriod"), dict) else {}
        result.append(
            {
                "name": _pick_first(item.get("title"), item.get("name")),
                "description": _pick_first(item.get("description"), ""),
                "tools": item.get("tools", []) if isinstance(item.get("tools"), list) else [],
                "url": _pick_first(item.get("url"), ""),
                "start_date": _pick_first(time_period.get("startDate"), item.get("startDate")),
                "end_date": _pick_first(time_period.get("endDate"), item.get("endDate")),
            }
        )
    return result


def _normalize_profile(payload: Dict[str, Any], source_url: str) -> Dict[str, Any]:
    profile = payload.get("data", payload)
    if not isinstance(profile, dict):
        raise RuntimeError("LinkdAPI returned an unexpected payload shape")

    experience = _normalize_positions(
        _to_list(_pick_first(profile.get("fullPositions"), profile.get("currentPositions"), profile.get("experiences"), profile.get("positions"), profile.get("experience")))
    )
    education = _normalize_education(_to_list(_pick_first(profile.get("education"), profile.get("educations"))))
    skills = _normalize_skills(_to_list(profile.get("skills")))
    certifications = _normalize_certifications(_to_list(profile.get("certifications")))
    projects = _normalize_projects(_to_list(profile.get("projects")))

    current_company = _pick_first(
        profile.get("companyName"),
        profile.get("currentCompany", {}).get("name") if isinstance(profile.get("currentCompany"), dict) else "",
        experience[0]["company"] if experience else "",
    )

    return {
        "source": {
            "provider": "linkdapi",
            "requested_url": source_url,
        },
        "status": _pick_first(payload.get("status"), profile.get("status"), "ok"),
        "identity": {
            "name": _build_name(profile),
            "headline": _pick_first(profile.get("headline"), profile.get("multiLocaleHeadline", {}).get("en_US") if isinstance(profile.get("multiLocaleHeadline"), dict) else ""),
            "about": _pick_first(profile.get("summary"), profile.get("about")),
            "linkedin_url": _pick_first(profile.get("url"), source_url),
            "location": _pick_first(
                profile.get("geoLocationName"),
                profile.get("location"),
                profile.get("geo", {}).get("full") if isinstance(profile.get("geo"), dict) else "",
            ),
            "current_company": current_company,
        },
        "experience": experience,
        "education": education,
        "skills": skills,
        "certifications": certifications,
        "projects": projects,
        "meta": {
            "urn": _pick_first(profile.get("urn"), ""),
            "connections": _pick_first(profile.get("connectionsCount"), ""),
            "followers": _pick_first(profile.get("followerCount"), ""),
        },
    }


def parse_linkedin_profile(
    linkedin_url: str,
    output_dir: Path,
    actor_id: str = "",
    token: str = "",
    poll_interval_seconds: int = 0,
    timeout_seconds: int = 30,
    linkedin_cookie: Optional[str] = None,
    proxy_country: str = "US",
) -> Dict[str, Path]:
    del actor_id, poll_interval_seconds, linkedin_cookie, proxy_country

    valid_url = _validate_linkedin_url(linkedin_url)
    username = _extract_username(valid_url)
    api_key = token or os.getenv("LINKDAPI_KEY") or os.getenv("LinkdAPI_KEY") or os.getenv("LINKDAPI_API_KEY")
    if not api_key:
        raise ValueError("LINKDAPI_KEY not found in backend/.env")

    url = f"{LINKDAPI_BASE_URL}/api/v1/profile/full"
    headers = {"X-linkdapi-apikey": api_key}
    params = {"username": username}

    logger.info("[LinkdAPI] Fetching full profile for username=%s", username)
    with httpx.Client(timeout=timeout_seconds) as client:
        response = client.get(url, headers=headers, params=params)

    if response.status_code == 401:
        raise RuntimeError("LinkdAPI authentication failed: missing or invalid API key")
    if response.status_code == 429:
        raise RuntimeError(f"LinkdAPI rate limit reached: {response.text}")
    response.raise_for_status()

    payload = response.json()
    if isinstance(payload, dict) and payload.get("success") is False:
        message = payload.get("message") or "LinkdAPI returned success=false"
        raise RuntimeError(message)

    normalized = _normalize_profile(payload, valid_url)
    logger.info(
        "[LinkdAPI] Normalized name=%r | experience=%d | education=%d | skills=%d",
        normalized["identity"].get("name"),
        len(normalized["experience"]),
        len(normalized["education"]),
        len(normalized["skills"]),
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    slug = username or valid_url.rstrip("/").split("/")[-1] or "linkedin_profile"
    formatted_path = output_dir / f"{slug}_linkedin_parsed.json"
    raw_path = output_dir / f"{slug}_linkedin_raw.json"

    formatted_path.write_text(json.dumps(normalized, indent=2, ensure_ascii=True), encoding="utf-8")
    raw_path.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")
    logger.info("[LinkdAPI] Saved parsed result -> %s", formatted_path)
    logger.info("[LinkdAPI] Saved raw result -> %s", raw_path)

    return {"formatted": formatted_path, "raw": raw_path}
