from fastapi import APIRouter
from pydantic import BaseModel
from app.supabase_client import supabase

router = APIRouter()


class FeedbackRequest(BaseModel):
    question: str
    answer: str
    rating: str
    comment: str = ""
    insurer: str = ""
    document_title: str = ""


@router.get("/")
def get_feedback():
    result = (
        supabase
        .table("feedback")
        .select("*")
        .order("created_at", desc=True)
        .limit(100)
        .execute()
    )

    return {"items": result.data}


@router.post("/")
def save_feedback(data: FeedbackRequest):
    supabase.table("feedback").insert({
        "question": data.question,
        "answer": data.answer,
        "rating": data.rating,
        "comment": data.comment,
        "insurer": data.insurer,
        "document_title": data.document_title
    }).execute()

    return {"success": True}
