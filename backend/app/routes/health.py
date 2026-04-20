from fastapi import APIRouter

router = APIRouter(prefix="/api/v1", tags=["Health"])


@router.get("/")
def root():
    return {
        "message": "VPP Checker API running"
    }


@router.get("/health")
def health():
    return {
        "status": "ok",
        "server": "running"
    }