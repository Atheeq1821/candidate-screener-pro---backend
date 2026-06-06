import logging
import os
from typing import Any, Dict, List

from app.services.groq_client import get_groq_client
from app.services.vectorless_engine import Chunk, retrieve
from app.services.vectorless_profile import append_retrieval_trace, build_vectorless_profile


logger = logging.getLogger(__name__)


def _llm_answer(question: str, chunks: List[Chunk]) -> str:
    client = get_groq_client()
    if client is None:
        return "Groq client is unavailable. Unable to generate LLM answer."

    context = "\n".join([f"- [{c.source}::{c.path}] {c.text}" for c in chunks])
    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            temperature=0.2,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a conversational recruiting assistant. "
                        "Answer questions about a candidate directly and naturally in 1–3 sentences — "
                        "no bullet lists, no numbered steps, no showing your working. "
                        "Use only the evidence provided. "
                        "If information is missing, say so briefly in one sentence."
                    ),
                },
                {"role": "user", "content": f"Question: {question}\n\nEvidence:\n{context}"},
            ],
        )
        return response.choices[0].message.content.strip()
    except Exception as exc:
        logger.warning("Groq answer generation failed; returning fallback text: %s", exc)
        return "I cannot confirm from available candidate evidence."


def answer_question(
    question: str,
    resume_parsed: Dict[str, Any],
    linkedin_parsed: Dict[str, Any],
    run_id: str | None = None,
    vectorless_rag: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    from app.services.vectorless_engine import flatten_to_chunks

    vectorless_artifact = vectorless_rag or build_vectorless_profile(resume_parsed, linkedin_parsed)
    chunks = flatten_to_chunks(vectorless_artifact["candidate_profile"], "vectorless")
    scored = retrieve(question, chunks, top_k=15)
    selected = [item.chunk for item in scored]
    if not selected:
        append_retrieval_trace(
            run_id or "anonymous",
            question,
            [],
        )
        return {"answer": "I cannot confirm from available candidate evidence.", "evidence": []}
    answer = _llm_answer(question, selected)
    evidence = [{"source": item.source, "path": item.path, "text": item.text[:220]} for item in selected[:5]]
    append_retrieval_trace(
        run_id or "anonymous",
        question,
        [
            {
                "source": item.chunk.source,
                "path": item.chunk.path,
                "text": item.chunk.text[:220],
                "score_total": item.score_total,
                "score_bm25": item.score_bm25,
                "score_exact": item.score_exact,
                "score_fuzzy": item.score_fuzzy,
                "score_path": item.score_path,
            }
            for item in scored
        ],
    )
    return {"answer": answer, "evidence": evidence}
