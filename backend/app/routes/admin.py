from fastapi import (
    APIRouter,
    HTTPException,
    Response,
    Cookie,
)

from pydantic import BaseModel
import os
from typing import Optional

from app.services.storage import (
    get_insurers,
    get_documents,
    add_insurer as storage_add_insurer,
    delete_insurer as storage_delete_insurer,
)

router = APIRouter(
    prefix="/api/v1/admin",
    tags=["Admin"]
)

COOKIE_NAME = "vpp_admin_session"


class LoginRequest(BaseModel):
    password: str


class InsurerRequest(BaseModel):
    name: str


def require_admin(session: str):
    if session != "logged_in":
        raise HTTPException(
            status_code=401,
            detail="Unauthorized"
        )


@router.post("/login")
def admin_login(
    data: LoginRequest,
    response: Response
):
    admin_password = os.getenv(
        "ADMIN_PASSWORD"
    )

    if not admin_password:
        raise HTTPException(
            status_code=500,
            detail="ADMIN_PASSWORD není nastaven."
        )

    if data.password != admin_password:
        raise HTTPException(
            status_code=401,
            detail="Neplatné heslo."
        )

    response.set_cookie(
        key=COOKIE_NAME,
        value="logged_in",
        httponly=True,
        secure=True,
        samesite="none",
        max_age=60 * 60 * 24 * 7
    )   

    return {"success": True}


@router.get("/me")
def admin_me(
    session: str = Cookie(
        default=None,
        alias=COOKIE_NAME
    )
):
    require_admin(session)
    return {"authenticated": True}


@router.post("/logout")
def admin_logout(response: Response):
    response.delete_cookie(COOKIE_NAME)
    return {"success": True}


@router.get("/insurers")
def list_insurers(
    session: str = Cookie(
        default=None,
        alias=COOKIE_NAME
    )
):
    require_admin(session)

    return {
        "insurers": get_insurers()
    }


@router.post("/insurers")
def add_insurer(
    
    data: InsurerRequest,
    session: str = Cookie(
        default=None,
        alias=COOKIE_NAME
    )
):
    print("ADMIN ADD INSURER ROUTE HIT")

    require_admin(session)

    name = data.name.strip()

    if not name:
        raise HTTPException(
            status_code=400,
            detail="Název je povinný."
        )

    storage_add_insurer(name)

    return {
        "success": True,
        "name": name
    }


@router.delete("/insurers/{name}")
def delete_insurer(
    name: str,
    session: str = Cookie(
        default=None,
        alias=COOKIE_NAME
    )
):
    require_admin(session)

    storage_delete_insurer(name)

    return {
        "success": True,
        "deleted": name
    }


@router.get("/documents")
def list_documents(
    insurer: Optional[str] = None,
    session: str = Cookie(
        default=None,
        alias=COOKIE_NAME
    )
):
    require_admin(session)

    docs = get_documents(insurer=insurer)

    return {
        "documents": [
            {
                "insurer": d.get("insurer"),
                "document_title": d.get("document_title"),
                "file_name": d.get("file_name")
            }
            for d in docs
        ]
    }
