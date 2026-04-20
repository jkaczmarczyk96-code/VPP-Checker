import os
from dotenv import load_dotenv

load_dotenv()


class Settings:
    APP_NAME = os.getenv(
        "APP_NAME",
        "VPP Checker"
    )

    DEBUG = (
        os.getenv(
            "DEBUG",
            "False"
        ) == "True"
    )

    # =====================
    # AI KEYS
    # =====================
    GOOGLE_API_KEY = os.getenv(
        "GOOGLE_API_KEY",
        ""
    )

    GROQ_API_KEY = os.getenv(
        "GROQ_API_KEY",
        ""
    )

    OPENAI_API_KEY = os.getenv(
        "OPENAI_API_KEY",
        ""
    )

    OPENROUTER_API_KEY = os.getenv(
        "OPENROUTER_API_KEY",
        ""
    )

    # =====================
    # VECTOR DB
    # =====================
    QDRANT_URL = os.getenv(
        "QDRANT_URL",
        ""
    )

    QDRANT_API_KEY = os.getenv(
        "QDRANT_API_KEY",
        ""
    )

    # =====================
    # GOOGLE SHEETS
    # =====================
    GOOGLE_SHEETS_CREDENTIALS = os.getenv(
        "GOOGLE_SHEETS_CREDENTIALS",
        "credentials.json"
    )

    GOOGLE_SHEET_NAME = os.getenv(
        "GOOGLE_SHEET_NAME",
        ""
    )

    # =====================
    # STORAGE
    # =====================
    UPLOAD_DIR = os.getenv(
        "UPLOAD_DIR",
        "uploads"
    )


settings = Settings()