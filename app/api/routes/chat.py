from fastapi import APIRouter, HTTPException

from app.core.deps import run_repo
from app.models.schemas import ChatRequest, ChatResponse
from app.services.chat_service import answer_question

router = APIRouter(prefix="/chat", tags=["chat"])


@router.post("", response_model=ChatResponse)
def chat(payload: ChatRequest) -> ChatResponse:
    run = run_repo.get(payload.run_id)
    if not run:
        raise HTTPException(status_code=404, detail="Run not found")
    if run.status != "completed" or not run.resume_parsed:
        raise HTTPException(status_code=400, detail="Run not ready for chat")

    output = answer_question(
        payload.question,
        run.resume_parsed or {},
        run.linkedin_parsed or {},
        run_id=payload.run_id,
        vectorless_rag=run.vectorless_rag,
    )
    return ChatResponse(answer=output["answer"], evidence=output["evidence"])
