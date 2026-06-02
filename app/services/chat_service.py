import os
from typing import Any, Dict, List

from app.services.vectorless_engine import Chunk, retrieve


def _llm_answer(question: str, chunks: List[Chunk]) -> str:
    from groq import Groq

    api_key = os.getenv("GROQ_API")
    if not api_key:
        return "GROQ_API is not configured. Unable to generate LLM answer."

    context = "\n".join([f"- [{c.source}::{c.path}] {c.text}" for c in chunks])
    client = Groq(api_key=api_key)
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        temperature=0.2,
        messages=[
            {
                "role": "system",
                "content": "Answer using only provided candidate evidence and role context. If missing, say cannot confirm.",
            },
            {"role": "user", "content": f"Question: {question}\n\nEvidence:\n{context}"},
        ],
    )
    return response.choices[0].message.content.strip()


def answer_question(question: str, resume_parsed: Dict[str, Any], linkedin_parsed: Dict[str, Any]) -> Dict[str, Any]:
    from app.services.vectorless_engine import flatten_to_chunks

    chunks = flatten_to_chunks(resume_parsed, "resume") + flatten_to_chunks(linkedin_parsed, "linkedin")
    scored = retrieve(question, chunks, top_k=8)
    selected = [item.chunk for item in scored]
    if not selected:
        return {"answer": "I cannot confirm from available candidate evidence.", "evidence": []}
    answer = _llm_answer(question, selected)
    evidence = [{"source": item.source, "path": item.path, "text": item.text[:220]} for item in selected[:5]]
    return {"answer": answer, "evidence": evidence}
