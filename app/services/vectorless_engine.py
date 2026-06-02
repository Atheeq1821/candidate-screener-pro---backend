import math
import os
import json
import re
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Any, Dict, List, Tuple

from app.models.schemas import FitResult, JobSpec


@dataclass
class Chunk:
    source: str
    path: str
    text: str


@dataclass
class ScoredChunk:
    chunk: Chunk
    score: float


def flatten_to_chunks(obj: Any, source: str, base_path: str = "") -> List[Chunk]:
    chunks: List[Chunk] = []
    if isinstance(obj, dict):
        for key, value in obj.items():
            child_path = f"{base_path}.{key}" if base_path else key
            chunks.extend(flatten_to_chunks(value, source, child_path))
    elif isinstance(obj, list):
        for idx, value in enumerate(obj):
            child_path = f"{base_path}[{idx}]"
            chunks.extend(flatten_to_chunks(value, source, child_path))
    else:
        text = str(obj).strip()
        if text:
            chunks.append(Chunk(source=source, path=base_path, text=text))
    return chunks


def _tokenize(text: str) -> List[str]:
    return re.findall(r"[a-zA-Z0-9_]+", text.lower())


def _build_index(chunks: List[Chunk]) -> Tuple[Dict[str, float], List[Dict[str, int]], List[int]]:
    doc_freq: Dict[str, int] = {}
    term_freqs: List[Dict[str, int]] = []
    doc_lens: List[int] = []
    for chunk in chunks:
        tokens = _tokenize(chunk.text)
        tf: Dict[str, int] = {}
        for token in tokens:
            tf[token] = tf.get(token, 0) + 1
        for token in tf:
            doc_freq[token] = doc_freq.get(token, 0) + 1
        term_freqs.append(tf)
        doc_lens.append(len(tokens))
    total_docs = max(len(chunks), 1)
    idf = {token: math.log((total_docs + 1) / (df + 0.5)) for token, df in doc_freq.items()}
    return idf, term_freqs, doc_lens


def _fuzzy_score(query_tokens: List[str], doc_tokens: List[str]) -> float:
    score = 0.0
    doc_set = set(doc_tokens)
    for qt in query_tokens:
        if qt in doc_set:
            score += 1.0
            continue
        best = 0.0
        for dt in doc_set:
            if abs(len(qt) - len(dt)) > 3:
                continue
            best = max(best, SequenceMatcher(None, qt, dt).ratio())
        if best >= 0.84:
            score += 0.7
    return score


def retrieve(query: str, chunks: List[Chunk], top_k: int = 8) -> List[ScoredChunk]:
    idf, tfs, lens = _build_index(chunks)
    avg_len = sum(lens) / max(len(lens), 1)
    q_tokens = _tokenize(query)
    results: List[ScoredChunk] = []
    for i, chunk in enumerate(chunks):
        tf = tfs[i]
        doc_len = lens[i]
        bm25 = 0.0
        k1, b = 1.5, 0.75
        for token in q_tokens:
            if token in tf:
                num = tf[token] * (k1 + 1)
                den = tf[token] + k1 * (1 - b + b * (doc_len / max(avg_len, 1.0)))
                bm25 += idf.get(token, 0.0) * (num / den)
        exact = 2.0 if query.lower().strip() in chunk.text.lower() else 0.0
        fuzzy = _fuzzy_score(q_tokens, _tokenize(chunk.text)) * 0.8
        score = bm25 + exact + fuzzy
        if score > 0:
            results.append(ScoredChunk(chunk=chunk, score=score))
    results.sort(key=lambda item: item.score, reverse=True)
    return results[:top_k]


def _normalize_skills(raw: List[Any]) -> set[str]:
    normalized = set()
    for entry in raw:
        if isinstance(entry, str):
            value = entry.strip().lower()
        elif isinstance(entry, dict):
            value = str(entry.get("name", "")).strip().lower()
        else:
            value = ""
        if value:
            normalized.add(value)
    return normalized


def _extract_years(linkedin: Dict[str, Any]) -> float:
    months = 0
    for item in linkedin.get("experience", []):
        if not isinstance(item, dict):
            continue
        start = item.get("start_date") or {}
        end = item.get("end_date")
        if isinstance(start, dict) and start.get("year"):
            sy = int(start.get("year", 0))
            sm = int(start.get("month", 1))
            if isinstance(end, dict) and end.get("year"):
                ey = int(end.get("year", sy))
                em = int(end.get("month", sm))
            else:
                ey, em = 2026, 5
            diff = (ey - sy) * 12 + (em - sm)
            if diff > 0:
                months += diff
    return round(months / 12.0, 2)


def _parse_needed_years(text: str) -> float:
    nums = re.findall(r"\d+(?:\.\d+)?", text or "")
    return float(nums[0]) if nums else 0.0


def _assess_joiner_with_rag(role: JobSpec, chunks: List[Chunk]) -> tuple[float, str]:
    availability_queries = [
        "notice period",
        "immediate joiner",
        "available to join",
        "serving notice",
        "joining from",
        "last working day",
    ]
    evidence_items: List[ScoredChunk] = []
    for query in availability_queries:
        evidence_items.extend(retrieve(query, chunks, top_k=2))

    if not evidence_items:
        return 50.0, "No explicit joiner evidence found; treated as unknown (neutral score)."

    # dedupe near-duplicate evidence by path+text
    unique: Dict[str, ScoredChunk] = {}
    for item in evidence_items:
        key = f"{item.chunk.source}|{item.chunk.path}|{item.chunk.text[:80]}"
        if key not in unique or unique[key].score < item.score:
            unique[key] = item
    top_evidence = sorted(unique.values(), key=lambda x: x.score, reverse=True)[:10]

    api_key = os.getenv("GROQ_API")
    if not api_key:
        return 50.0, "GROQ_API missing; joiner treated as unknown (neutral score)."

    try:
        from groq import Groq

        evidence_text = "\n".join([f"- [{item.chunk.source}::{item.chunk.path}] {item.chunk.text}" for item in top_evidence])
        client = Groq(api_key=api_key)
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            temperature=0.1,
            response_format={"type": "json_object"},
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are an availability assessor. Use only supplied evidence. "
                        "Return strict JSON: "
                        "{\"is_immediate_joiner\":\"yes|no|unknown\","
                        "\"notice_period_months\":number|null,"
                        "\"confidence\":0_to_1,"
                        "\"reason\":\"short reason\"}."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"Role immediate_joiner requirement in months: {role.immediate_joiner}\n\n"
                        f"Evidence:\n{evidence_text}"
                    ),
                },
            ],
        )
        payload = response.choices[0].message.content
        data = json.loads(payload) if payload and payload.strip().startswith("{") else {}
        status = str(data.get("is_immediate_joiner", "unknown")).lower()
        notice = data.get("notice_period_months", None)
        confidence = float(data.get("confidence", 0.0) or 0.0)
        reason = str(data.get("reason", "No reliable explicit availability signal found."))

        if status == "yes":
            score = 100.0
        elif status == "no":
            if isinstance(notice, (int, float)):
                allowed = role.immediate_joiner
                if notice <= allowed:
                    score = 90.0
                elif notice <= allowed + 1:
                    score = 70.0
                elif notice <= allowed + 2:
                    score = 40.0
                else:
                    score = 10.0
            else:
                score = 30.0
        else:
            score = 50.0

        summary = (
            f"Joiner assessment: {status}, notice={notice}, confidence={round(confidence, 2)}. "
            f"Reason: {reason}"
        )
        return score, summary
    except Exception as exc:
        return 50.0, f"Joiner assessment fallback to unknown due to parser error: {exc}"


def run_role_fit(role: JobSpec, resume: Dict[str, Any], linkedin: Dict[str, Any]) -> tuple[FitResult, List[Chunk]]:
    chunks = flatten_to_chunks(resume, "resume") + flatten_to_chunks(linkedin, "linkedin")
    candidate_skills = _normalize_skills(resume.get("skills", [])) | _normalize_skills(linkedin.get("skills", []))
    required = [skill.strip().lower() for skill in role.skills_required if skill.strip()]
    hits = [skill for skill in required if skill in candidate_skills]
    missing = [skill for skill in required if skill not in hits]
    skills_score = round((len(hits) / max(len(required), 1)) * 100, 2)

    exp = _extract_years(linkedin)
    needed = _parse_needed_years(role.experience_needed)
    exp_score = 50.0 if needed <= 0 else min(100.0, round((exp / needed) * 100, 2))
    joiner_score, joiner_summary = _assess_joiner_with_rag(role, chunks)
    overall = round((skills_score * 0.5) + (exp_score * 0.35) + (joiner_score * 0.15), 2)

    if overall >= 80:
        fit_band = "strong fit"
    elif overall >= 60:
        fit_band = "moderate fit"
    else:
        fit_band = "limited fit"

    missing_preview = ", ".join(missing[:4]) if missing else "no major missing required skills"
    notes = (
        f"Candidate is a {fit_band} for '{role.role_name}'. "
        f"Skill coverage is {len(hits)}/{len(required)} and experience shows {exp} years against {needed} required. "
        f"Current gap focus: {missing_preview}. {joiner_summary}"
    )

    fit = FitResult(
        role_name=role.role_name,
        overall_score=overall,
        skills_fit_score=skills_score,
        experience_fit_score=exp_score,
        joiner_fit_score=joiner_score,
        top_alignments=[
            f"Skill match: {len(hits)}/{len(required)} required skills found.",
            f"Experience fit: {exp} years vs required {needed} years.",
        ],
        top_gaps=[f"Missing or weakly evidenced skills: {', '.join(missing[:8])}" if missing else "No major skill gaps detected."],
        notes=notes,
        matched_skills=hits,
        missing_skills=missing,
    )
    return fit, chunks
