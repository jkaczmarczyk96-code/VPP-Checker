import os
import json
import re

from google import genai
from openai import OpenAI

from app.config import settings


# =========================
# CLIENTS
# =========================
gemini_client = None
groq_client = None
openrouter_client = None

if settings.GOOGLE_API_KEY:
    gemini_client = genai.Client(
        api_key=settings.GOOGLE_API_KEY
    )

if settings.GROQ_API_KEY:
    groq_client = OpenAI(
        api_key=settings.GROQ_API_KEY,
        base_url="https://api.groq.com/openai/v1"
    )

if settings.OPENROUTER_API_KEY:
    openrouter_client = OpenAI(
        api_key=settings.OPENROUTER_API_KEY,
        base_url="https://openrouter.ai/api/v1"
    )


DOCUMENT_TEXT = ""

BASE_DIR = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        "..",
        ".."
    )
)

CACHE_FILE = os.path.join(
    BASE_DIR,
    "uploads",
    "ai_cache.json"
)


# =========================
# CACHE
# =========================
def load_cache():
    if not os.path.exists(CACHE_FILE):
        return {}

    try:
        with open(
            CACHE_FILE,
            "r",
            encoding="utf-8"
        ) as f:
            return json.load(f)
    except Exception:
        return {}


def save_cache(data):
    try:
        with open(
            CACHE_FILE,
            "w",
            encoding="utf-8"
        ) as f:
            json.dump(
                data,
                f,
                ensure_ascii=False,
                indent=2
            )
    except Exception:
        pass


def make_cache_key(text: str):
    return text.strip().lower()


# =========================
# DOCUMENT CONTEXT
# =========================
def set_document_text(text: str):
    global DOCUMENT_TEXT
    DOCUMENT_TEXT = text


# =========================
# LOCAL FALLBACK
# =========================
def local_answer(question: str):
    if not DOCUMENT_TEXT:
        return None

    words = re.findall(
        r"\w+",
        question.lower()
    )

    words = [
        w for w in words
        if len(w) > 2
    ]

    found = []

    for line in DOCUMENT_TEXT.split("\n"):
        row = line.strip()

        if len(row) < 20:
            continue

        score = 0
        lower = row.lower()

        for word in words:
            if word in lower:
                score += 1

        if score > 0:
            found.append((score, row))

    found.sort(
        key=lambda x: x[0],
        reverse=True
    )

    best = found[:4]

    if not best:
        return (
            "V poskytnutém dokumentu jsem tuto informaci nenašel."
        )

    bullets = "\n".join(
        [f"- {x[1]}" for x in best]
    )

    return f"""## Shrnutí

Ve zdrojovém textu byly nalezeny relevantní informace.

## Klíčové body

{bullets}

## Prakticky pro vás

Doporučuji ověřit přesné znění citace."""
    

# =========================
# PROVIDERS
# =========================
def ask_openrouter(prompt: str):
    if not openrouter_client:
        return None

    models = [
        "openai/gpt-4o-mini",
        "mistralai/mistral-small-3.1-24b-instruct",
        "meta-llama/llama-3.3-70b-instruct",
    ]

    for model_name in models:
        try:
            response = openrouter_client.chat.completions.create(
                model=model_name,
                messages=[
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=0.2,
                timeout=25,
            )

            text = response.choices[0].message.content

            if text:
                return text

        except Exception as e:
            print("OpenRouter error:", e)

    return None


def ask_gemini(prompt: str):
    if not gemini_client:
        return None

    try:
        response = gemini_client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
        )

        if response.text:
            return response.text

    except Exception as e:
        print("Gemini error:", e)

    return None


def ask_groq(prompt: str):
    if not groq_client:
        return None

    try:
        response = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0.2,
            timeout=20,
        )

        text = response.choices[0].message.content

        if text:
            return text

    except Exception as e:
        print("Groq error:", e)

    return None


# =========================
# MAIN
# =========================
def ask_ai(
    text: str,
    raw: bool = False
) -> str:

    text = text.strip()

    if not text:
        return "Nebyl zadán dotaz."

    cache = load_cache()
    cache_key = make_cache_key(text)

    if cache_key in cache:
        return cache[cache_key]

    # pokud je raw=True, bereme text jako hotový prompt
    if raw:
        prompt = text
    else:
        prompt = f"""
Odpověz česky stručně a přesně.

DOTAZ:
{text}
"""

    answer = ask_openrouter(prompt)

    if not answer:
        answer = ask_gemini(prompt)

    if not answer:
        answer = ask_groq(prompt)

    if not answer:
        answer = local_answer(text)

    if not answer:
        answer = (
            "AI je momentálně nedostupná. "
            "Zkuste to prosím později."
        )

    cache[cache_key] = answer
    save_cache(cache)

    return answer