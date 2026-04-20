import sqlite3
from pathlib import Path

from fastapi import APIRouter
from pydantic import BaseModel

router = APIRouter()

BASE_DIR = Path(__file__).resolve().parent.parent.parent
DB_PATH = BASE_DIR / "feedback.db"


class FeedbackRequest(BaseModel):
    question: str
    answer: str
    rating: str
    comment: str = ""
    insurer: str = ""
    document_title: str = ""


def init_db():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS feedback (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            question TEXT,
            answer TEXT,
            rating TEXT,
            comment TEXT,
            insurer TEXT,
            document_title TEXT
        )
        """
    )

    conn.commit()
    conn.close()


@router.on_event("startup")
def startup():
    init_db()


@router.get("/")
def get_feedback():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    cursor.execute(
        """
        SELECT *
        FROM feedback
        ORDER BY id DESC
        LIMIT 100
        """
    )

    rows = cursor.fetchall()
    conn.close()

    return {
        "items": [
            dict(row)
            for row in rows
        ]
    }


@router.post("/")
def save_feedback(data: FeedbackRequest):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute(
        """
        INSERT INTO feedback (
            question,
            answer,
            rating,
            comment,
            insurer,
            document_title
        )
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (
            data.question,
            data.answer,
            data.rating,
            data.comment,
            data.insurer,
            data.document_title,
        ),
    )

    conn.commit()
    conn.close()

    return {
        "success": True
    }