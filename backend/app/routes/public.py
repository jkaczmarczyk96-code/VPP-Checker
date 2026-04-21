from fastapi import APIRouter

from app.services.storage import (
    get_insurers,
    get_documents,
)

router = APIRouter(
    prefix="/api/v1/public",
    tags=["Public"]
)


@router.get("/insurers")
def public_insurers():
    return {
        "insurers": get_insurers()
    }


@router.get("/documents")
def public_documents(
    insurer: str = ""
):
    docs = get_documents(
        insurer=insurer
    )

    return {
        "documents": [
            {
                "document_title":
                    d.get(
                        "document_title",
                        ""
                    )
            }
            for d in docs
        ]
    }