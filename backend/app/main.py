from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.routes.health import router as health_router
from app.routes.chat import router as chat_router
from app.routes.upload import router as upload_router
from app.routes.admin import router as admin_router
from app.routes.feedback import router as feedback_router

app = FastAPI(
    title="VPP Checker API",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "https://vpp-checker.vercel.app",
        "https://www.vpp-checker.vercel.app",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health_router)
app.include_router(chat_router)
app.include_router(upload_router)
app.include_router(admin_router)
app.include_router(
    feedback_router,
    prefix="/api/v1/feedback",
    tags=["feedback"]
)
