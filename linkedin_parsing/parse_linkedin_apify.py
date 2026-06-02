import argparse
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import quote

import httpx
from dotenv import load_dotenv


APIFY_BASE_URL = "https://api.apify.com/v2"


def _validate_linkedin_url(url: str) -> str:
    value = url.strip()
    if "linkedin.com/in/" not in value and "linkedin.com/pub/" not in value:
        raise ValueError(f"Not a valid LinkedIn profile URL: {url}")
    return value


def _build_actor_input_candidates(
    linkedin_url: str,
    actor_id: str,
    linkedin_cookie: Optional[str],
    proxy_country: str,
) -> List[Dict[str, Any]]:
    actor_key = actor_id.strip().lower()
    if actor_key in ("supreme_coder/linkedin-profile-scraper", "dev_fusion/linkedin-profile-scraper", "data-slayer/linkedin-profile-scraper"):
        return [{"profileUrls": [linkedin_url]}, {"urls": [{"url": linkedin_url}]}, {"startUrls": [{"url": linkedin_url}]}]
    if actor_key == "curious_coder/linkedin-profile-scraper":
        if not linkedin_cookie:
            raise ValueError("curious_coder/linkedin-profile-scraper requires a LinkedIn session cookie.")
        return [
            {"profileUrls": [linkedin_url], "startUrls": [{"url": linkedin_url}], "minDelay": 15, "maxDelay": 60, "cookie": linkedin_cookie, "proxy": {"useApifyProxy": True, "apifyProxyCountry": proxy_country}},
            {"profileUrls": [linkedin_url], "minDelay": 15, "maxDelay": 60, "cookie": linkedin_cookie, "proxy": {"useApifyProxy": True, "apifyProxyCountry": proxy_country}},
            {"startUrls": [{"url": linkedin_url}], "minDelay": 15, "maxDelay": 60, "cookie": linkedin_cookie, "proxy": {"useApifyProxy": True, "apifyProxyCountry": proxy_country}},
        ]
    return [{"profileUrls": [linkedin_url], "startUrls": [{"url": linkedin_url}], "urls": [linkedin_url], "maxItems": 1}]


def _run_actor_and_get_items(
    client: httpx.Client,
    token: str,
    actor_id: str,
    actor_input: Dict[str, Any],
    poll_interval_seconds: int,
    timeout_seconds: int,
) -> List[Dict[str, Any]]:
    actor_path = quote(actor_id.strip().replace("/", "~"), safe="~")
    run_url = f"{APIFY_BASE_URL}/acts/{actor_path}/runs"
    run_resp = client.post(run_url, params={"token": token}, json=actor_input, timeout=60)
    run_resp.raise_for_status()
    run_data = run_resp.json()["data"]
    run_id = run_data["id"]
    deadline = time.time() + timeout_seconds
    status = run_data.get("status", "READY")
    dataset_id: Optional[str] = run_data.get("defaultDatasetId")

    while status not in {"SUCCEEDED", "FAILED", "ABORTED", "TIMED-OUT"}:
        if time.time() > deadline:
            raise TimeoutError(f"Apify run timed out after {timeout_seconds}s (run_id={run_id})")
        time.sleep(poll_interval_seconds)
        status_resp = client.get(f"{APIFY_BASE_URL}/actor-runs/{run_id}", params={"token": token}, timeout=30)
        status_resp.raise_for_status()
        status_data = status_resp.json()["data"]
        status = status_data.get("status", status)
        dataset_id = status_data.get("defaultDatasetId", dataset_id)

    if status != "SUCCEEDED" or not dataset_id:
        return []

    items_resp = client.get(f"{APIFY_BASE_URL}/datasets/{dataset_id}/items", params={"token": token, "clean": "true"}, timeout=60)
    items_resp.raise_for_status()
    return items_resp.json()


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
    fallback = _pick_first(profile.get("fullName"), profile.get("name"), item.get("fullName"), item.get("name"))
    return str(fallback).strip()


def _normalize_profile(item: Dict[str, Any], source_url: str, actor_id: str) -> Dict[str, Any]:
    profile = item.get("profile", item)
    raw_positions = _to_list(_pick_first(profile.get("positions"), item.get("positions"), profile.get("experiences"), profile.get("experience"), item.get("experiences"), item.get("experience")))
    raw_educations = _to_list(_pick_first(profile.get("educations"), item.get("educations"), profile.get("education"), item.get("education")))
    raw_skills = _to_list(_pick_first(profile.get("skills"), item.get("skills")))
    raw_certs = _to_list(_pick_first(profile.get("certifications"), item.get("certifications")))
    raw_projects = _to_list(_pick_first(profile.get("projects"), item.get("projects")))

    return {
        "source": {"provider": "apify", "actor_id": actor_id, "requested_url": source_url},
        "status": _pick_first(item.get("status"), "ok"),
        "identity": {
            "name": _build_name(profile, item),
            "headline": _pick_first(profile.get("headline"), item.get("headline")),
            "about": _pick_first(profile.get("about"), profile.get("summary"), item.get("about"), item.get("summary")),
            "linkedin_url": _pick_first(profile.get("linkedinUrl"), profile.get("url"), source_url),
            "location": _pick_first(profile.get("geoLocationName"), item.get("geoLocationName"), profile.get("location"), item.get("location")),
            "current_company": _pick_first(profile.get("companyName"), item.get("companyName"), (profile.get("currentCompany", {}) or {}).get("name") if isinstance(profile.get("currentCompany"), dict) else None, (item.get("currentCompany", {}) or {}).get("name") if isinstance(item.get("currentCompany"), dict) else None),
        },
        "experience": raw_positions,
        "education": raw_educations,
        "skills": raw_skills,
        "certifications": raw_certs,
        "projects": raw_projects,
        "meta": {"connections": _pick_first(item.get("connectionsCount"), profile.get("connectionsCount")), "followers": _pick_first(item.get("followerCount"), profile.get("followerCount"))},
    }


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
    valid_url = _validate_linkedin_url(linkedin_url)
    actor_inputs = _build_actor_input_candidates(valid_url, actor_id, linkedin_cookie, proxy_country)
    with httpx.Client() as client:
        items: List[Dict[str, Any]] = []
        last_error: Optional[Exception] = None
        for actor_input in actor_inputs:
            try:
                items = _run_actor_and_get_items(client, token, actor_id, actor_input, poll_interval_seconds, timeout_seconds)
                break
            except RuntimeError as exc:
                last_error = exc
                continue
        if not items and last_error:
            raise last_error
    if not items:
        raise RuntimeError(f"No profile data returned for URL: {valid_url}")
    first_item = items[0]
    normalized = _normalize_profile(first_item, source_url=valid_url, actor_id=actor_id)
    output_dir.mkdir(parents=True, exist_ok=True)
    slug = valid_url.rstrip("/").split("/")[-1] or "linkedin_profile"
    formatted_path = output_dir / f"{slug}_linkedin_parsed.json"
    raw_path = output_dir / f"{slug}_linkedin_raw.json"
    formatted_path.write_text(json.dumps(normalized, indent=2, ensure_ascii=True), encoding="utf-8")
    raw_path.write_text(json.dumps(first_item, indent=2, ensure_ascii=True), encoding="utf-8")
    return {"formatted": formatted_path, "raw": raw_path}


def main() -> None:
    load_dotenv(Path("pdf_extraction") / ".env")
    parser = argparse.ArgumentParser(description="Parse a LinkedIn profile via an Apify actor.")
    parser.add_argument("--linkedin-url", required=True)
    parser.add_argument("--actor-id", default="supreme_coder/linkedin-profile-scraper")
    parser.add_argument("--output-dir", default=str(Path("linkedin_parsing") / "output"))
    parser.add_argument("--poll-interval", type=int, default=5)
    parser.add_argument("--timeout-seconds", type=int, default=180)
    parser.add_argument("--proxy-country", default="US")
    args = parser.parse_args()

    token = os.getenv("APIFY_API_TOKEN")
    if not token:
        raise ValueError("APIFY_API_TOKEN not found in pdf_extraction/.env")

    parse_linkedin_profile(
        linkedin_url=args.linkedin_url,
        output_dir=Path(args.output_dir).resolve(),
        actor_id=args.actor_id,
        token=token,
        poll_interval_seconds=args.poll_interval,
        timeout_seconds=args.timeout_seconds,
        linkedin_cookie=os.getenv("LINKEDIN_COOKIE"),
        proxy_country=args.proxy_country,
    )


if __name__ == "__main__":
    main()
