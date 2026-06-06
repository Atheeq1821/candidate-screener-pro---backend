import argparse
import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import quote

import httpx
from dotenv import load_dotenv


APIFY_BASE_URL = "https://api.apify.com/v2"

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# URL validation
# ---------------------------------------------------------------------------

def _validate_linkedin_url(url: str) -> str:
    value = url.strip()
    if "linkedin.com/in/" not in value and "linkedin.com/pub/" not in value:
        raise ValueError(f"Not a valid LinkedIn profile URL: {url}")
    return value


# ---------------------------------------------------------------------------
# Actor input builders — each actor has a different expected input shape
# ---------------------------------------------------------------------------

def _build_actor_input_candidates(
    linkedin_url: str,
    actor_id: str,
    linkedin_cookie: Optional[str],
    proxy_country: str,
) -> List[Dict[str, Any]]:
    actor_key = actor_id.strip().lower()

    # supreme_coder / dev_fusion / data-slayer — no cookie needed
    if actor_key in (
        "supreme_coder/linkedin-profile-scraper",
        "dev_fusion/linkedin-profile-scraper",
        "data-slayer/linkedin-profile-scraper",
    ):
        return [
            {"profileUrls": [linkedin_url]},
            {"urls": [{"url": linkedin_url}]},
            {"startUrls": [{"url": linkedin_url}]},
        ]

    # curious_coder — requires LinkedIn session cookie
    if actor_key == "curious_coder/linkedin-profile-scraper":
        if not linkedin_cookie:
            raise ValueError(
                "curious_coder/linkedin-profile-scraper requires a LinkedIn session cookie. "
                "Set LINKEDIN_COOKIE in pdf_extraction/.env"
            )
        return [
            {
                "profileUrls": [linkedin_url],
                "startUrls": [{"url": linkedin_url}],
                "minDelay": 15,
                "maxDelay": 60,
                "cookie": linkedin_cookie,
                "proxy": {"useApifyProxy": True, "apifyProxyCountry": proxy_country},
            },
            {
                "profileUrls": [linkedin_url],
                "minDelay": 15,
                "maxDelay": 60,
                "cookie": linkedin_cookie,
                "proxy": {"useApifyProxy": True, "apifyProxyCountry": proxy_country},
            },
            {
                "startUrls": [{"url": linkedin_url}],
                "minDelay": 15,
                "maxDelay": 60,
                "cookie": linkedin_cookie,
                "proxy": {"useApifyProxy": True, "apifyProxyCountry": proxy_country},
            },
        ]

    # Generic fallback — tries all common input shapes
    return [
        {
            "profileUrls": [linkedin_url],
            "startUrls": [{"url": linkedin_url}],
            "urls": [linkedin_url],
            "maxItems": 1,
        }
    ]


# ---------------------------------------------------------------------------
# Apify run + polling
# ---------------------------------------------------------------------------

def _run_actor_and_get_items(
    client: httpx.Client,
    token: str,
    actor_id: str,
    actor_input: Dict[str, Any],
    poll_interval_seconds: int,
    timeout_seconds: int,
) -> List[Dict[str, Any]]:
    actor_path = actor_id.strip().replace("/", "~")
    actor_path = quote(actor_path, safe="~")
    run_url = f"{APIFY_BASE_URL}/acts/{actor_path}/runs"

    logger.info("[Apify] POST %s | input keys: %s", run_url, list(actor_input.keys()))
    run_resp = client.post(run_url, params={"token": token}, json=actor_input, timeout=60)
    logger.info("[Apify] Run start response: HTTP %s", run_resp.status_code)

    if run_resp.status_code == 404:
        raise RuntimeError(
            f"Apify actor not found: '{actor_id}'. "
            "Check the exact actor id on Apify Store (owner/actor-name)."
        )
    if run_resp.status_code == 403:
        try:
            detail = f" Response: {run_resp.json()}"
        except Exception:
            detail = f" Response text: {run_resp.text}"
        raise RuntimeError(
            f"Access denied for actor '{actor_id}' (HTTP 403). "
            "Approve actor permissions at: "
            "https://console.apify.com and re-run."
            f"{detail}"
        )
    if run_resp.status_code == 400:
        try:
            detail = json.dumps(run_resp.json(), indent=2, ensure_ascii=True)
        except Exception:
            detail = run_resp.text
        raise RuntimeError(
            f"Bad input for actor '{actor_id}' (HTTP 400). "
            f"Apify validation response:\n{detail}"
        )

    run_resp.raise_for_status()
    run_data = run_resp.json()["data"]
    run_id = run_data["id"]
    logger.info("[Apify] Run started: run_id=%s", run_id)

    deadline = time.time() + timeout_seconds
    status = run_data.get("status", "READY")
    dataset_id: Optional[str] = run_data.get("defaultDatasetId")

    while status not in {"SUCCEEDED", "FAILED", "ABORTED", "TIMED-OUT"}:
        if time.time() > deadline:
            raise TimeoutError(f"Apify run timed out after {timeout_seconds}s (run_id={run_id})")
        time.sleep(poll_interval_seconds)
        status_resp = client.get(
            f"{APIFY_BASE_URL}/actor-runs/{run_id}",
            params={"token": token},
            timeout=30,
        )
        status_resp.raise_for_status()
        status_data = status_resp.json()["data"]
        status = status_data.get("status", status)
        dataset_id = status_data.get("defaultDatasetId", dataset_id)
        logger.debug("[Apify] Poll run_id=%s → status=%s", run_id, status)

    logger.info("[Apify] Run finished: run_id=%s status=%s dataset_id=%s", run_id, status, dataset_id)

    if status != "SUCCEEDED":
        raise RuntimeError(f"Apify run ended with status={status}, run_id={run_id}")
    if not dataset_id:
        return []

    items_resp = client.get(
        f"{APIFY_BASE_URL}/datasets/{dataset_id}/items",
        params={"token": token, "clean": "true"},
        timeout=60,
    )
    items_resp.raise_for_status()
    items = items_resp.json()
    logger.info("[Apify] Dataset fetched: %d item(s) returned", len(items))
    return items


# ---------------------------------------------------------------------------
# Normalization helpers
# ---------------------------------------------------------------------------

def _pick_first(*values: Any) -> Any:
    for value in values:
        if value not in (None, "", [], {}):
            return value
    return ""


def _to_list(value: Any) -> List[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return [v for v in value if v not in (None, "", {})]
    return [] if value in (None, "", {}) else [value]


def _build_name(profile: Dict[str, Any], item: Dict[str, Any]) -> str:
    first = str(_pick_first(profile.get("firstName"), item.get("firstName"))).strip()
    last = str(_pick_first(profile.get("lastName"), item.get("lastName"))).strip()
    joined = f"{first} {last}".strip()
    if joined:
        return " ".join(joined.split())
    fallback = _pick_first(
        profile.get("fullName"), profile.get("name"),
        item.get("fullName"), item.get("name"),
    )
    return str(fallback).strip()


def _flatten_positions(positions: List[Any]) -> List[Dict[str, Any]]:
    """
    Handle supreme_coder's nested structure:
      positions[i].company   → company info
      positions[i].positions → list of roles at that company

    Also handles flat structures where each item is a single role.
    """
    flat: List[Dict[str, Any]] = []
    for block in positions:
        if not isinstance(block, dict):
            continue

        company_info = block.get("company", {})
        company_name = (
            company_info.get("name", "") if isinstance(company_info, dict)
            else str(company_info)
        )
        company_url = company_info.get("url", "") if isinstance(company_info, dict) else ""

        sub_positions = block.get("positions", [])

        if sub_positions and isinstance(sub_positions, list):
            # Nested structure (supreme_coder style)
            for pos in sub_positions:
                if not isinstance(pos, dict):
                    continue
                time_period = pos.get("timePeriod", {}) or {}
                flat.append({
                    "company": company_name,
                    "company_url": company_url,
                    "title": pos.get("title", ""),
                    "location": pos.get("locationName", ""),
                    "start_date": time_period.get("startDate"),
                    "end_date": time_period.get("endDate"),
                    "duration": pos.get("totalDuration", ""),
                    "description": pos.get("description", ""),
                    "skills_insight": pos.get("insights", ""),
                    "is_current": time_period.get("endDate") is None,
                })
        else:
            # Flat structure — block itself is a single role
            time_period = block.get("timePeriod", {}) or {}
            # company name may be directly on block if no nested company obj
            flat_company = company_name or block.get("companyName", "")
            flat.append({
                "company": flat_company,
                "company_url": company_url,
                "title": block.get("title", ""),
                "location": block.get("locationName", ""),
                "start_date": time_period.get("startDate"),
                "end_date": time_period.get("endDate"),
                "duration": block.get("totalDuration", ""),
                "description": block.get("description", ""),
                "skills_insight": block.get("insights", ""),
                "is_current": time_period.get("endDate") is None,
            })
    return flat


def _normalize_education(educations: List[Any]) -> List[Dict[str, Any]]:
    result: List[Dict[str, Any]] = []
    for edu in educations:
        if not isinstance(edu, dict):
            continue
        time_period = edu.get("timePeriod", {}) or {}
        result.append({
            "school": edu.get("schoolName", ""),
            "school_url": edu.get("schoolUrl", ""),
            "degree": edu.get("degreeName", ""),
            "field_of_study": edu.get("fieldOfStudy", ""),
            "start_date": time_period.get("startDate"),
            "end_date": time_period.get("endDate"),
            "grade": edu.get("grade", ""),
            "description": edu.get("description", ""),
        })
    return result


def _normalize_skills(skills: List[Any]) -> List[str]:
    """Deduplicate skill names, case-insensitively."""
    seen: set = set()
    result: List[str] = []
    for skill in skills:
        if isinstance(skill, dict):
            name = skill.get("name", "").strip()
        elif isinstance(skill, str):
            name = skill.strip()
        else:
            continue
        key = name.lower()
        if name and key not in seen:
            seen.add(key)
            result.append(name)
    return result


def _normalize_certifications(certs: List[Any]) -> List[Dict[str, Any]]:
    result: List[Dict[str, Any]] = []
    for cert in certs:
        if not isinstance(cert, dict):
            continue
        time_period = cert.get("timePeriod", {}) or {}
        result.append({
            "name": cert.get("name", ""),
            "issuer": _pick_first(cert.get("issuer"), cert.get("authority"), ""),
            "issue_date": _pick_first(cert.get("issueDate"), time_period.get("startDate")),
            "expiry_date": time_period.get("endDate"),
            "url": cert.get("url", ""),
        })
    return result


def _normalize_projects(projects: List[Any]) -> List[Dict[str, Any]]:
    result: List[Dict[str, Any]] = []
    for proj in projects:
        if not isinstance(proj, dict):
            continue
        time_period = proj.get("timePeriod", {}) or {}
        result.append({
            "title": proj.get("title", ""),
            "description": proj.get("description", ""),
            "url": proj.get("url", ""),
            "start_date": time_period.get("startDate"),
            "end_date": time_period.get("endDate"),
        })
    return result


# ---------------------------------------------------------------------------
# Main normalization entry point
# ---------------------------------------------------------------------------

def _normalize_profile(item: Dict[str, Any], source_url: str, actor_id: str) -> Dict[str, Any]:
    """
    Normalize a raw Apify LinkedIn item into a consistent schema.

    Handles multiple actor output shapes:
      - supreme_coder: positions / educations (nested)
      - dev_fusion / data-slayer: experiences / education (flat)
      - generic fallback
    """
    profile = item.get("profile", item)

    # Experience — supreme_coder uses 'positions', others use 'experiences'/'experience'
    raw_positions = _to_list(_pick_first(
        profile.get("positions"),
        item.get("positions"),
        profile.get("experiences"),
        profile.get("experience"),
        item.get("experiences"),
        item.get("experience"),
    ))

    # Education — supreme_coder uses 'educations', others use 'education'
    raw_educations = _to_list(_pick_first(
        profile.get("educations"),
        item.get("educations"),
        profile.get("education"),
        item.get("education"),
    ))

    raw_skills = _to_list(_pick_first(
        profile.get("skills"),
        item.get("skills"),
    ))

    raw_certs = _to_list(_pick_first(
        profile.get("certifications"),
        item.get("certifications"),
    ))

    raw_projects = _to_list(_pick_first(
        profile.get("projects"),
        item.get("projects"),
    ))

    current_company_name = _pick_first(
        profile.get("companyName"),
        item.get("companyName"),
        (
            profile.get("currentCompany", {}).get("name")
            if isinstance(profile.get("currentCompany"), dict)
            else None
        ),
        (
            item.get("currentCompany", {}).get("name")
            if isinstance(item.get("currentCompany"), dict)
            else None
        ),
    )

    return {
        "source": {
            "provider": "apify",
            "actor_id": actor_id,
            "requested_url": source_url,
        },
        "status": _pick_first(item.get("status"), "ok"),
        "identity": {
            "name": _build_name(profile, item),
            "headline": _pick_first(profile.get("headline"), item.get("headline")),
            "about": _pick_first(
                profile.get("about"), profile.get("summary"),
                item.get("about"), item.get("summary"),
            ),
            "linkedin_url": _pick_first(
                profile.get("linkedinUrl"), profile.get("url"), source_url,
            ),
            "location": _pick_first(
                profile.get("geoLocationName"), item.get("geoLocationName"),
                profile.get("location"), item.get("location"),
            ),
            "current_company": current_company_name,
        },
        "experience": _flatten_positions(raw_positions),
        "education": _normalize_education(raw_educations),
        "skills": _normalize_skills(raw_skills),
        "certifications": _normalize_certifications(raw_certs),
        "projects": _normalize_projects(raw_projects),
        "meta": {
            "connections": _pick_first(
                item.get("connectionsCount"), profile.get("connectionsCount"),
            ),
            "followers": _pick_first(
                item.get("followerCount"), profile.get("followerCount"),
            ),
        }
    }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def parse_linkedin_profile(
    linkedin_url: str,
    output_dir: Path,
    actor_id: str,
    token: str,
    poll_interval_seconds: int,
    timeout_seconds: int,
    linkedin_cookie: Optional[str] = None,
    proxy_country: str = "US",
) -> Dict[str, Path]:
    logger.info("[LinkedIn] Starting parse for URL: %s | actor: %s", linkedin_url, actor_id)

    valid_url = _validate_linkedin_url(linkedin_url)
    actor_inputs = _build_actor_input_candidates(
        valid_url, actor_id, linkedin_cookie, proxy_country
    )
    logger.info("[LinkedIn] Generated %d actor input candidate(s) to try", len(actor_inputs))

    with httpx.Client() as client:
        last_error: Optional[Exception] = None
        items: List[Dict[str, Any]] = []

        for idx, actor_input in enumerate(actor_inputs):
            logger.info("[LinkedIn] Trying actor input #%d: %s", idx + 1, actor_input)
            try:
                items = _run_actor_and_get_items(
                    client=client,
                    token=token,
                    actor_id=actor_id,
                    actor_input=actor_input,
                    poll_interval_seconds=poll_interval_seconds,
                    timeout_seconds=timeout_seconds,
                )
                logger.info("[LinkedIn] Actor input #%d succeeded — %d item(s) returned", idx + 1, len(items))
                break
            except RuntimeError as exc:
                logger.warning("[LinkedIn] Actor input #%d failed: %s", idx + 1, exc)
                if "HTTP 400" in str(exc):
                    last_error = exc
                    continue
                raise

        if not items and last_error:
            raise last_error

    if not items:
        raise RuntimeError(f"No profile data returned for URL: {valid_url}")

    first_item = items[0]
    logger.info("[LinkedIn] Raw item top-level keys: %s", list(first_item.keys()))

    # Detect Apify error responses (e.g. free-tier limit hit)
    if "error" in first_item and len(first_item) <= 2:
        error_msg = str(first_item["error"]).strip()
        logger.error("[LinkedIn] Apify returned an error item: %s", error_msg)
        raise RuntimeError(f"Apify actor returned an error: {error_msg}")

    normalized = _normalize_profile(first_item, source_url=valid_url, actor_id=actor_id)
    logger.info(
        "[LinkedIn] Normalized — name=%r | experience=%d | education=%d | skills=%d",
        normalized["identity"].get("name"),
        len(normalized["experience"]),
        len(normalized["education"]),
        len(normalized["skills"]),
    )

    if first_item.get("status") == "not_found":
        normalized["error"] = (
            "Profile not found by selected Apify actor. "
            "Try another actor or verify the profile is public."
        )
        logger.warning("[LinkedIn] Profile not_found for URL: %s", valid_url)

    output_dir.mkdir(parents=True, exist_ok=True)
    slug = valid_url.rstrip("/").split("/")[-1] or "linkedin_profile"

    formatted_path = output_dir / f"{slug}_linkedin_parsed.json"
    raw_path = output_dir / f"{slug}_linkedin_raw.json"

    # Only cache if the result has meaningful content
    has_content = bool(
        normalized["identity"].get("name")
        or normalized["experience"]
        or normalized["education"]
    )
    if has_content:
        formatted_path.write_text(
            json.dumps(normalized, indent=2, ensure_ascii=True), encoding="utf-8"
        )
        logger.info("[LinkedIn] Saved parsed result → %s", formatted_path)
    else:
        logger.warning(
            "[LinkedIn] Normalized result has no meaningful content — NOT caching to disk. "
            "Raw item will still be saved for debugging."
        )
        # Remove stale empty cache if it exists from a previous bad run
        if formatted_path.exists():
            formatted_path.unlink()
            logger.warning("[LinkedIn] Deleted stale empty cache: %s", formatted_path)

    raw_path.write_text(
        json.dumps(first_item, indent=2, ensure_ascii=True), encoding="utf-8"
    )
    logger.info("[LinkedIn] Saved raw item → %s", raw_path)

    return {"formatted": formatted_path, "raw": raw_path}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Parse a LinkedIn profile via an Apify actor and normalize the output."
    )
    parser.add_argument(
        "--linkedin-url", required=True,
        help="LinkedIn profile URL (linkedin.com/in/...).",
    )
    parser.add_argument(
        "--actor-id",
        default="supreme_coder/linkedin-profile-scraper",
        help=(
            "Apify actor id. Supported: "
            "supreme_coder/linkedin-profile-scraper (default, no cookie), "
            "dev_fusion/linkedin-profile-scraper, "
            "data-slayer/linkedin-profile-scraper, "
            "curious_coder/linkedin-profile-scraper (requires LINKEDIN_COOKIE)."
        ),
    )
    parser.add_argument(
        "--output-dir",
        default=str(Path("linkedin_parsing") / "output"),
        help="Directory where normalized and raw LinkedIn JSON files will be written.",
    )
    parser.add_argument(
        "--poll-interval", type=int, default=5,
        help="Seconds between Apify run status polls.",
    )
    parser.add_argument(
        "--timeout-seconds", type=int, default=180,
        help="Max seconds to wait for Apify run completion.",
    )
    parser.add_argument(
        "--proxy-country", default="US",
        help="Apify proxy country code (e.g. US, IN). Used only by cookie-based actors.",
    )
    return parser.parse_args()


def main() -> None:
    load_dotenv(Path("pdf_extraction") / ".env")
    args = parse_args()

    token = os.getenv("APIFY_API_TOKEN")
    if not token:
        raise ValueError("APIFY_API_TOKEN not found in pdf_extraction/.env")

    output_paths = parse_linkedin_profile(
        linkedin_url=args.linkedin_url,
        output_dir=Path(args.output_dir).resolve(),
        actor_id=args.actor_id,
        token=token,
        poll_interval_seconds=args.poll_interval,
        timeout_seconds=args.timeout_seconds,
        linkedin_cookie=os.getenv("LINKEDIN_COOKIE"),
        proxy_country=args.proxy_country,
    )

    print(f"LinkedIn formatted profile -> {output_paths['formatted']}")
    print(f"LinkedIn raw profile       -> {output_paths['raw']}")


if __name__ == "__main__":
    main()
