from fastapi import APIRouter
from pydantic import BaseModel
from typing import List
import re

from app.services.search import ask_ai
from app.services.storage import get_documents
from app.services.retrieval import find_relevant_chunks

router = APIRouter(
    prefix="/api/v1/chat",
    tags=["Chat"]
)


class MessageItem(BaseModel):
    role: str
    text: str


class ChatRequest(BaseModel):
    question: str
    insurer: str | None = None
    document_title: str | None = None
    messages: List[MessageItem] = []


# =========================
# QUERY EXPAND
# =========================
def expand_question(question: str) -> str:
    q = question.lower()

    extra = []

    mapping = {
        "alkohol": [
            "alkohol",
            "alkoholu",
            "opilost",
            "promile",
            "pod vlivem",
            "pod vplyvom",
            "výluky",
            "vylúčenia",
        ],
        "drogy": [
            "drogy",
            "omamné látky",
            "psychotropní látky",
            "výluky",
        ],
        "úraz": [
            "úraz",
            "uraz",
            "nehoda",
            "zranění",
            "trvalé následky",
        ],
        "nemoc": [
            "nemoc",
            "choroba",
            "hospitalizace",
            "léčba",
            "ošetření",
        ],
        "storno": [
            "storno",
            "zrušení cesty",
            "prerušenie cesty",
        ],
        "zavazad": [
            "zavazadla",
            "batožina",
            "krádež kufru",
        ],
        "odpovědnost": [
            "odpovědnost",
            "škoda třetí osobě",
        ],
    }

    for trigger, words in mapping.items():
        if trigger in q:
            extra.extend(words)

    if extra:
        question += " " + " ".join(extra)

    return question


# =========================
# UNIVERSAL ROUTER
# =========================
def preferred_chunks(
    question: str,
    chunks: list
):
    q = question.lower()

    intents = {
        "vyluky": {
            "query": [
                "výluka",
                "výluky",
                "vyloučeno",
                "nevztahuje",
                "nevzťahuje",
                "alkohol",
                "drogy",
                "odmítnutí plnění",
            ],
            "target": [
                "výluky",
                "vylúčenia",
                "nevzťahuje",
                "nevztahuje",
                "nehradí",
            ],
        },

        "plneni": {
            "query": [
                "plnění",
                "plnenie",
                "limit",
                "částka",
                "kolik",
                "pojistná suma",
            ],
            "target": [
                "pojistné plnění",
                "poistné plnenie",
                "poistné sumy",
                "limity",
            ],
        },

        "uraz": {
            "query": [
                "úraz",
                "uraz",
                "smrt",
                "invalidita",
                "zlomenina",
            ],
            "target": [
                "úraz",
                "smrť",
                "trvalé následky",
            ],
        },

        "nemoc": {
            "query": [
                "nemoc",
                "choroba",
                "lékař",
                "hospitalizace",
                "léčba",
            ],
            "target": [
                "léčebné",
                "liečebné",
                "choroba",
                "hospital",
            ],
        },

        "storno": {
            "query": [
                "storno",
                "zrušení",
                "přerušení",
                "prerušenie",
            ],
            "target": [
                "storno",
                "zrušenie",
                "prerušenie",
            ],
        },

        "zavazadla": {
            "query": [
                "zavazadlo",
                "zavazadla",
                "kufr",
                "batožina",
            ],
            "target": [
                "batožina",
                "zavazad",
                "krádež",
            ],
        },

        "odpovednost": {
            "query": [
                "odpovědnost",
                "odpovednost",
                "škoda třetí osobě",
            ],
            "target": [
                "odpovědnost",
                "zodpovednosť",
                "škoda",
            ],
        },

        "povinnosti": {
            "query": [
                "povinnost",
                "doklady",
                "nahlásit",
                "oznámit",
                "co doložit",
            ],
            "target": [
                "povinnosti",
                "doklady",
                "oznámenie",
            ],
        },
    }

    selected_targets = []

    for item in intents.values():
        if any(word in q for word in item["query"]):
            selected_targets.extend(
                item["target"]
            )

    if not selected_targets:
        return []

    results = []

    for chunk in chunks:
        if isinstance(chunk, dict):
            text = (
                chunk.get("section", "")
                + " "
                + chunk.get("part", "")
                + " "
                + chunk.get("text", "")
            ).lower()
        else:
            text = str(chunk).lower()

        score = 0

        for word in selected_targets:
            if word in text:
                score += 1

        if score > 0:
            results.append(
                (score, chunk)
            )

    results.sort(
        key=lambda x: x[0],
        reverse=True
    )

    return [
        chunk
        for score, chunk in results[:3]
    ]


# =========================
# SECTION DETECTOR
# =========================
def detect_section(text: str) -> str:
    patterns = [
        r"(Výluky[^\n]*)",
        r"(Vylúčenia[^\n]*)",
        r"(Článek\s+\d+[^\n]*)",
        r"(Článok\s+\d+[^\n]*)",
        r"(Čl\.\s*\d+[^\n]*)",
    ]

    for pattern in patterns:
        found = re.search(
            pattern,
            text,
            re.IGNORECASE
        )

        if found:
            return found.group(1).strip()

    return "Neuvedeno"


@router.post("/")
def ask(data: ChatRequest):
    documents = get_documents()

    if data.insurer:
        documents = [
            d for d in documents
            if d.get("insurer") == data.insurer
        ]

    if (
        data.document_title
        and data.document_title != "all"
    ):
        documents = [
            d for d in documents
            if d.get("document_title")
            == data.document_title
        ]

    expanded_question = expand_question(
        data.question
    )

    selected_chunks = []
    source_document = ""
    source_section = ""
    source_article = ""
    source_paragraph = ""
    source_quote = ""

    for doc in documents:
        chunks = doc.get("chunks", [])

        best = preferred_chunks(
            data.question,
            chunks
        )

        if not best:
            best = find_relevant_chunks(
                question=expanded_question,
                chunks=chunks,
                limit=2
            )

        if best:
            selected_chunks.extend(best)

            if not source_document:
                source_document = doc.get(
                    "document_title",
                    ""
                )

                first = best[0]

                if isinstance(first, dict):
                    text = first.get(
                        "text",
                        ""
                    )

                    source_section = (
                        first.get(
                            "section",
                            ""
                        )
                        or detect_section(text)
                    )

                    source_article = str(
                        first.get(
                            "article",
                            ""
                        )
                    )

                    source_paragraph = str(
                        first.get(
                            "paragraph",
                            ""
                        )
                    )

                    source_quote = text

                else:
                    text = str(first)
                    source_quote = text
                    source_section = detect_section(
                        text
                    )

    if not selected_chunks:
        selected_chunks = [
            d.get(
                "text_content",
                ""
            )
            for d in documents
        ]

    context_parts = []

    for item in selected_chunks:
        if isinstance(item, dict):
            context_parts.append(
                item.get("text", "")
            )
        else:
            context_parts.append(item)

    context = "\n\n".join(
        context_parts
    )[:6000]

    history_items = data.messages[-4:]
    history_text = ""

    for item in history_items:
        role = (
            "Uživatel"
            if item.role == "user"
            else "AI"
        )

        history_text += (
            f"{role}: {item.text}\n"
        )

    prompt = f"""
Jsi senior expert na pojistné podmínky.

Odpovídáš klientovi pojišťovny jednoduše,
profesionálně a srozumitelně.

Používej POUZE informace ze ZDROJE.

Pokud odpověď ve zdroji není, napiš přesně:
V poskytnutém dokumentu jsem tuto informaci nenašel.

Nikdy si nic nevymýšlej.
Nevkládej domněnky.
Nevytvářej právní výklad nad rámec textu.

Odpovídej vždy ČESKY
(i když je zdroj slovensky).

Přelož slovenské formulace přirozeně do češtiny.

Piš stručně, věcně a profesionálně.

Použij přesně tento MARKDOWN formát:

Piš jako zkušený poradce v chatu.
Začni přirozeně:
"Jasně, našel jsem to."
"Níže je to podstatné."
"Podle podmínek platí:"

## Klíčové body

- maximálně 4 body
- každý bod krátká věta
- pouze relevantní informace

## Výluky / omezení

Tuto sekci napiš POUZE pokud ji zdroj skutečně obsahuje
a vztahuje se k dotazu.

## Prakticky pro vás

Jedna krátká užitečná rada pro klienta.

DALŠÍ PRAVIDLA:

- Nepiš dlouhé odstavce.
- Neopakuj stejné informace.
- Necituj celé věty ze zdroje.
- Nepoužívej frázi:
  "V poskytnutém dokumentu jsem tuto informaci nenašel"
  pokud zdroj obsahuje částečnou odpověď.
- Pokud je odpověď jen částečná, napiš to normálně lidsky.
- Nepiš sekci Výluky / omezení jen jako prázdnou výplň.
- Pokud něco není jasné, napiš to opatrně.

HISTORIE:
{history_text}

ZDROJ:
{context}

DOTAZ:
{data.question}
"""

    answer = ask_ai(
        prompt,
        raw=True
    )

    return {
        "question": data.question,
        "answer": answer,
        "source_document": source_document,
        "source_section": source_section,
        "source_article": source_article,
        "source_paragraph": source_paragraph,
        "source_quote": source_quote
    }