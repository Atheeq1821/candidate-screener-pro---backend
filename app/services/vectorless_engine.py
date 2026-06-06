import json
import logging
import math
import os
import re
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Any, Dict, List, Tuple, Set

from app.models.schemas import FitResult, JobSpec
from app.services.groq_client import get_groq_client
from app.services.vectorless_profile import build_vectorless_profile


logger = logging.getLogger(__name__)


@dataclass
class Chunk:
    source: str
    path: str
    text: str


@dataclass
class ScoredChunk:
    chunk: Chunk
    score_total: float
    score_bm25: float
    score_exact: float
    score_fuzzy: float
    score_path: float


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


_STOP_WORDS: frozenset = frozenset({
    "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "by", "from", "is", "are", "was", "were", "be", "been",
    "being", "have", "has", "had", "do", "does", "did", "will", "would",
    "could", "should", "may", "might", "shall", "can", "what", "which",
    "who", "whom", "this", "that", "these", "those", "i", "you", "he",
    "she", "it", "we", "they", "me", "him", "her", "us", "them",
    "candidate", "candidates",
})


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


def _path_boost(path: str, query_tokens: List[str]) -> float:
    lowered_path = path.lower()
    boost = 0.0
    if any(token in {"work", "company", "companies", "experience", "role", "current", "worked"} for token in query_tokens):
        if any(marker in lowered_path for marker in ["experience", "current_company", "company", "title"]):
            boost += 1.6
    if any(token in {"skill", "skills"} for token in query_tokens) and "skills" in lowered_path:
        boost += 1.4
    if any(token in {"education", "degree", "college", "school"} for token in query_tokens) and "education" in lowered_path:
        boost += 1.4
    if any(token in {"project", "projects"} for token in query_tokens) and "projects" in lowered_path:
        boost += 1.4
    if any(token in {"joiner", "join", "available", "availability", "notice", "immediate"} for token in query_tokens):
        if any(marker in lowered_path for marker in ["linkedin_headline", "headline", "about", "notice"]):
            boost += 2.0
    return boost


def _score_chunk(
    query: str,
    chunk: Chunk,
    idf: Dict[str, float],
    tf: Dict[str, int],
    doc_len: int,
    avg_doc_len: float,
) -> ScoredChunk:
    query_tokens = _tokenize(query)
    bm25_query_tokens = [t for t in query_tokens if t not in _STOP_WORDS]
    k1 = 1.5
    b = 0.75
    score_bm25 = 0.0
    for token in bm25_query_tokens:
        if token not in tf:
            continue
        token_idf = idf.get(token, 0.0)
        numerator = tf[token] * (k1 + 1)
        denominator = tf[token] + k1 * (1 - b + b * (doc_len / max(avg_doc_len, 1.0)))
        score_bm25 += token_idf * (numerator / denominator)

    lowered_chunk = chunk.text.lower()
    lowered_query = query.lower().strip()
    score_exact = 2.0 if lowered_query and lowered_query in lowered_chunk else 0.0
    doc_tokens = _tokenize(chunk.text)
    score_fuzzy = _fuzzy_score(query_tokens, doc_tokens) * 0.8
    score_path = _path_boost(chunk.path, query_tokens)
    score_total = score_bm25 + score_exact + score_fuzzy + score_path
    return ScoredChunk(
        chunk=chunk,
        score_total=score_total,
        score_bm25=score_bm25,
        score_exact=score_exact,
        score_fuzzy=score_fuzzy,
        score_path=score_path,
    )


def retrieve(query: str, chunks: List[Chunk], top_k: int = 8) -> List[ScoredChunk]:
    idf, tfs, lens = _build_index(chunks)
    avg_len = sum(lens) / max(len(lens), 1)
    results: List[ScoredChunk] = []
    for i, chunk in enumerate(chunks):
        scored_chunk = _score_chunk(query, chunk, idf, tfs[i], lens[i], avg_len)
        if scored_chunk.score_total > 0:
            results.append(scored_chunk)
    results.sort(key=lambda item: item.score_total, reverse=True)
    return results[:top_k]


def _normalize_skills(raw: List[Any]) -> Set[str]:
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


def _is_skill_match(req_skill: str, candidate_skills: Set[str]) -> bool:
    # Aggressive normalization: remove all non-alphanumeric chars
    req_clean = re.sub(r'[^a-z0-9]', '', req_skill.lower())
    if not req_clean:
        return False
        
    for cand in candidate_skills:
        cand_clean = re.sub(r'[^a-z0-9]', '', cand.lower())
        if not cand_clean:
            continue
            
        # 1. Exact match after aggressive normalization (e.g. "spring_boot" == "Spring Boot")
        if req_clean == cand_clean:
            return True
            
        # 2. Fuzzy match for typos, only if lengths are close to avoid weird substring artifacts
        if abs(len(req_clean) - len(cand_clean)) <= 2:
            if SequenceMatcher(None, req_clean, cand_clean).ratio() >= 0.85:
                return True
                
    return False


def _comprehensive_llm_assessment(role: JobSpec, chunks: List[Chunk], missing_skills: List[str]) -> Tuple[float, float, str, List[str]]:
    client = get_groq_client()
    if client is None:
        return 0.0, 50.0, "Groq client unavailable; fallback to neutral scores.", []

    # To avoid exceeding context window, we'll use all chunks if they are reasonably small.
    # We will deduplicate chunks by text.
    unique_texts = list(set([c.text.strip() for c in chunks if c.text.strip()]))
    evidence_text = "\n".join([f"- {text}" for text in unique_texts])
    
    # Simple truncate to avoid blowing up the context window (rough estimate of 6000 words ~ 8k tokens)
    evidence_text = " ".join(evidence_text.split()[:6000])

    try:
        prompt = (
            f"You are an expert technical recruiter assessing a candidate's fit based on their resume and LinkedIn data.\n"
            f"Role immediate_joiner requirement in months: {role.immediate_joiner}\n"
            f"These skills were not found via exact keyword match: {json.dumps(missing_skills)}\n\n"
            f"Evidence:\n{evidence_text}"
        )
        
        system_prompt = (
            "Return strict JSON matching this schema exactly:\n"
            "{\n"
            "  \"total_experience_years\": <float, calculate total years of experience across all roles>,\n"
            "  \"is_immediate_joiner\": <\"yes\"|\"no\"|\"unknown\">,\n"
            "  \"notice_period_months\": <number|null>,\n"
            "  \"missing_skills_found_in_text\": [<list of strings from the provided missing skills list that the candidate actually possesses based on semantics or aliases>],\n"
            "  \"reasoning\": \"<short explanation for experience and joiner status>\"\n"
            "}"
        )

        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            temperature=0.1,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
        )
        
        payload = response.choices[0].message.content
        data = json.loads(payload) if payload and payload.strip().startswith("{") else {}
        
        # 1. Experience
        exp_years = float(data.get("total_experience_years", 0.0) or 0.0)
        
        # 2. Joiner status
        status = str(data.get("is_immediate_joiner", "unknown")).lower()
        notice = data.get("notice_period_months", None)
        reason = str(data.get("reasoning", "No explicit reasoning provided."))
        
        if status == "yes":
            joiner_score = 100.0
        elif status == "no":
            if isinstance(notice, (int, float)):
                allowed = role.immediate_joiner
                if notice <= allowed:
                    joiner_score = 90.0
                elif notice <= allowed + 1:
                    joiner_score = 70.0
                elif notice <= allowed + 2:
                    joiner_score = 40.0
                else:
                    joiner_score = 10.0
            else:
                joiner_score = 30.0
        else:
            joiner_score = 50.0
            
        joiner_summary = f"Assessment: {status}, notice={notice}. Reason: {reason}"
        
        # 3. Found missing skills
        found_skills = data.get("missing_skills_found_in_text", [])
        if not isinstance(found_skills, list):
            found_skills = []
            
        return exp_years, joiner_score, joiner_summary, [str(s) for s in found_skills]
        
    except Exception as exc:
        return 0.0, 50.0, f"LLM assessment failed: {exc}", []

def _parse_needed_years(text: str) -> float:
    nums = re.findall(r"\d+(?:\.\d+)?", text or "")
    return float(nums[0]) if nums else 0.0


def _llm_generate_retrieval_queries(role: JobSpec, fit_result: FitResult) -> List[str]:
    client = get_groq_client()
    if client is None:
        return []

    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            temperature=0.1,
            response_format={"type": "json_object"},
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Generate focused retrieval queries for vectorless RAG. "
                        "Return JSON only: {\"queries\": [..]} with max 8 short queries."
                    ),
                },
                {
                    "role": "user",
                    "content": json.dumps(
                        {
                            "role_name": role.role_name,
                            "skills_required": role.skills_required,
                            "job_description": role.job_description,
                            "current_top_gaps": fit_result.top_gaps,
                        }
                    ),
                },
            ],
        )
        payload = json.loads(response.choices[0].message.content)
        queries = payload.get("queries", [])
        return [str(query).strip() for query in queries if str(query).strip()][:8]
    except Exception as exc:
        logger.warning("Groq retrieval query generation failed; using fallback queries: %s", exc)
        return []


def _llm_refine_fit_result(role: JobSpec, fit_result: FitResult, evidence_chunks: List[ScoredChunk]) -> FitResult:
    client = get_groq_client()
    if client is None:
        return fit_result

    evidence_text = "\n".join(
        [f"- [{item.chunk.source}::{item.chunk.path}] {item.chunk.text}" for item in evidence_chunks[:12]]
    )
    prompt_payload = {
        "role_name": role.role_name,
        "experience_needed": role.experience_needed,
        "skills_required": role.skills_required,
        "current_fit_result": fit_result.model_dump(mode="json"),
    }

    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            temperature=0.2,
            response_format={"type": "json_object"},
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are an ATS analyst. Return JSON only with keys: top_alignments, top_gaps, notes. "
                        "Use only evidence provided. Keep each gap concise."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"Input:\n{json.dumps(prompt_payload)}\n\nEvidence:\n{evidence_text}\n\n"
                        "Return strict JSON."
                    ),
                },
            ],
        )
        llm_json = json.loads(response.choices[0].message.content)
        alignments = llm_json.get("top_alignments") or fit_result.top_alignments
        gaps = llm_json.get("top_gaps") or fit_result.top_gaps
        notes = llm_json.get("notes") or fit_result.notes
        return FitResult(
            role_name=fit_result.role_name,
            overall_score=fit_result.overall_score,
            skills_fit_score=fit_result.skills_fit_score,
            experience_fit_score=fit_result.experience_fit_score,
            joiner_fit_score=fit_result.joiner_fit_score,
            top_alignments=[str(x) for x in alignments][:6],
            top_gaps=[str(x) for x in gaps][:6],
            notes=str(notes),
            matched_skills=fit_result.matched_skills,
            missing_skills=fit_result.missing_skills,
        )
    except Exception as exc:
        logger.warning("Groq fit refinement failed; using deterministic fit result: %s", exc)
        return fit_result


def _llm_audit_with_evidence(role: JobSpec, fit_result: FitResult, evidence_by_query: Dict[str, List[ScoredChunk]]) -> FitResult:
    client = get_groq_client()
    if client is None:
        return fit_result

    flattened_evidence: List[str] = []
    for query, items in evidence_by_query.items():
        flattened_evidence.append(f"Query: {query}")
        for item in items[:4]:
            flattened_evidence.append(f"- [{item.chunk.source}::{item.chunk.path}] {item.chunk.text}")
    evidence_text = "\n".join(flattened_evidence[:80])

    prompt = {
        "role": role.model_dump(mode="json"),
        "deterministic_result": fit_result.model_dump(mode="json"),
        "required_output": {
            "top_alignments": ["..."],
            "top_gaps": ["..."],
            "notes": "...",
            "possible_false_negatives": ["..."],
            "possible_false_positives": ["..."],
        },
    }

    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            temperature=0.2,
            response_format={"type": "json_object"},
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are an ATS auditor. Use only supplied evidence. "
                        "If evidence is insufficient, explicitly say cannot confirm. Return strict JSON."
                    ),
                },
                {
                    "role": "user",
                    "content": f"Input:\n{json.dumps(prompt)}\n\nEvidence:\n{evidence_text}",
                },
            ],
        )
        output = json.loads(response.choices[0].message.content)
        alignments = output.get("top_alignments") or fit_result.top_alignments
        gaps = output.get("top_gaps") or fit_result.top_gaps
        notes = str(output.get("notes") or fit_result.notes)
        fn = output.get("possible_false_negatives") or []
        fp = output.get("possible_false_positives") or []
        if fn or fp:
            notes = (
                f"{notes} | Audit: FN={'; '.join([str(x) for x in fn][:3]) or 'none'}, "
                f"FP={'; '.join([str(x) for x in fp][:3]) or 'none'}"
            )
        return FitResult(
            role_name=fit_result.role_name,
            overall_score=fit_result.overall_score,
            skills_fit_score=fit_result.skills_fit_score,
            experience_fit_score=fit_result.experience_fit_score,
            joiner_fit_score=fit_result.joiner_fit_score,
            top_alignments=[str(x) for x in alignments][:6],
            top_gaps=[str(x) for x in gaps][:6],
            notes=notes,
            matched_skills=fit_result.matched_skills,
            missing_skills=fit_result.missing_skills,
        )
    except Exception as exc:
        logger.warning("Groq audit failed; using deterministic fit result: %s", exc)
        return fit_result


def run_role_fit(role: JobSpec, resume: Dict[str, Any], linkedin: Dict[str, Any]) -> Tuple[FitResult, List[Chunk]]:
    vectorless_rag = build_vectorless_profile(resume, linkedin)
    candidate_profile = vectorless_rag["candidate_profile"]
    chunks = flatten_to_chunks(candidate_profile, "vectorless")
    candidate_skills = _normalize_skills(candidate_profile.get("skills", []))
    required = [skill.strip().lower() for skill in role.skills_required if skill.strip()]
    
    # 1. Fast fuzzy match
    hits = [skill for skill in required if _is_skill_match(skill, candidate_skills)]
    missing = [skill for skill in required if skill not in hits]
    
    # 2. Comprehensive LLM Assessment (for experience, joiner status, and semantic skill checking)
    exp, joiner_score, joiner_summary, llm_found_skills = _comprehensive_llm_assessment(role, chunks, missing)
    
    # Promote semantically matched skills
    for fs in llm_found_skills:
        fs_lower = fs.strip().lower()
        if fs_lower in missing:
            missing.remove(fs_lower)
            if fs_lower not in hits:
                hits.append(fs_lower)
                
    skills_score = round((len(hits) / max(len(required), 1)) * 100, 2)

    needed = _parse_needed_years(role.experience_needed)
    exp_score = 50.0 if needed <= 0 else min(100.0, round((exp / needed) * 100, 2))
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

    base_fit = FitResult(
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

    # 3. CLI RAG Flow: Refine and Audit
    planner_queries = _llm_generate_retrieval_queries(role, base_fit)
    evidence_by_query: Dict[str, List[ScoredChunk]] = {}
    
    if planner_queries:
        for query in planner_queries:
            evidence_by_query[query] = retrieve(query, chunks, top_k=8)
    else:
        fallback_query = role.job_description
        evidence_by_query[fallback_query] = retrieve(fallback_query, chunks, top_k=10)

    pooled = [item for items in evidence_by_query.values() for item in items]
    
    # Sort pooled by score and deduplicate
    pooled.sort(key=lambda item: item.score_total, reverse=True)
    unique_pooled = []
    seen_texts = set()
    for item in pooled:
        if item.chunk.text not in seen_texts:
            seen_texts.add(item.chunk.text)
            unique_pooled.append(item)
            
    refined_fit = _llm_refine_fit_result(role, base_fit, unique_pooled)
    final_fit = _llm_audit_with_evidence(role, refined_fit, evidence_by_query)

    return final_fit, chunks
