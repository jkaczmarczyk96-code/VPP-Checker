from typing import List, Dict, Any, Union
import re


# =========================
# TEXT HELPERS
# =========================
def normalize_text(text: str) -> str:
    return text.lower().strip()


def tokenize(text: str) -> List[str]:
    words = re.findall(
        r"\w+",
        text.lower()
    )

    return [
        w for w in words
        if len(w) >= 3
    ]


def get_chunk_text(
    chunk: Union[str, Dict[str, Any]]
) -> str:
    if isinstance(chunk, dict):
        return chunk.get("text", "")
    return str(chunk)


def get_chunk_field(
    chunk: Union[str, Dict[str, Any]],
    field: str
) -> str:
    if isinstance(chunk, dict):
        return str(
            chunk.get(field, "")
        )
    return ""


# =========================
# DOMAIN QUERY EXPANSION
# =========================
def expand_words(words: List[str]) -> List[str]:
    expanded = list(words)

    mapping = {
        "alkohol": [
            "alkohol",
            "alkoholu",
            "opilost",
            "promile",
            "pod",
            "vplyvom",
            "intoxik",
            "omam",
            "drogy",
        ],
        "plnění": [
            "plnenie",
            "plnění",
            "pois",
        ],
        "vyloučeno": [
            "výluky",
            "vylúčenia",
            "nevzťahuje",
            "nevzťahuje",
            "vyloučen",
        ],
        "úraz": [
            "úraz",
            "uraz",
            "zranění",
            "nehoda",
        ],
        "nemoc": [
            "choroba",
            "nemoc",
            "hospital",
            "liečba",
        ],
        "storno": [
            "storno",
            "zrušenie",
            "zruseni",
            "cesty",
        ],
        "zavazadlo": [
            "batožina",
            "batozina",
            "zavazad",
        ],
    }

    joined = " ".join(words)

    for trigger, extra in mapping.items():
        if trigger in joined:
            expanded.extend(extra)

    return list(set(expanded))


# =========================
# SCORING
# =========================
def keyword_score(
    words: List[str],
    text: str
) -> int:
    score = 0

    for word in words:
        count = text.count(word)

        if count > 0:
            score += count * 12

    return score


def overlap_score(
    q_words: List[str],
    c_words: List[str]
) -> int:
    common = set(q_words) & set(c_words)
    return len(common) * 14


def heading_bonus(
    words: List[str],
    heading: str
) -> int:
    if not heading:
        return 0

    heading = normalize_text(heading)

    score = 0

    for word in words:
        if word in heading:
            score += 70

    return score


def article_bonus(
    article: str
) -> int:
    if article:
        return 5
    return 0


def negative_penalty(
    heading: str
) -> int:
    if not heading:
        return 0

    heading = normalize_text(heading)

    bad = [
        "stanovenie výšky poistného",
        "povinnosti",
        "závěreč",
        "závereč",
        "administr",
        "oznámen",
        "doručov",
    ]

    score = 0

    for word in bad:
        if word in heading:
            score -= 60

    return score


# =========================
# MAIN
# =========================
def find_relevant_chunks(
    question: str,
    chunks: List[Union[str, Dict[str, Any]]],
    limit: int = 3
):
    if not question or not chunks:
        return []

    q_words = tokenize(question)
    q_words = expand_words(q_words)

    scored = []

    for chunk in chunks:
        text = get_chunk_text(chunk)
        text_n = normalize_text(text)
        text_words = tokenize(text)

        section = get_chunk_field(
            chunk,
            "section"
        )

        part = get_chunk_field(
            chunk,
            "part"
        )

        article = get_chunk_field(
            chunk,
            "article"
        )

        score = 0

        score += keyword_score(
            q_words,
            text_n
        )

        score += overlap_score(
            q_words,
            text_words
        )

        score += heading_bonus(
            q_words,
            section
        )

        score += heading_bonus(
            q_words,
            part
        )

        score += article_bonus(
            article
        )

        score += negative_penalty(
            section
        )

        scored.append(
            (score, chunk)
        )

    scored.sort(
        key=lambda x: x[0],
        reverse=True
    )

    best = [
        chunk
        for score, chunk in scored
        if score > 0
    ]

    return best[:limit]