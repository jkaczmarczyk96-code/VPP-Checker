from fastapi import FastAPI
from pydantic import BaseModel
from transformers import pipeline

app = FastAPI()

# 🔥 AI MODEL (odborník na pojištění)
llm = pipeline(
    "text2text-generation",
    model="google/flan-t5-base",
    max_length=512,
    do_sample=False
)

class Query(BaseModel):
    question: str
    contexts: list

@app.post("/ask")
def ask(q: Query):

    context_text = "\n\n".join(q.contexts)

    prompt = f"""
Jsi odborník na pojistné podmínky.

ODPOVÍDEJ POUZE NA ZÁKLADĚ POSKYTNUTÉHO TEXTU.

PRAVIDLA:
- Nepoužívej žádné znalosti mimo text
- Pokud odpověď není v textu, napiš: "Informace není v dokumentu"
- Používej citace [str. X]
- Cituj přesné věty

TEXT:
{context_text}

OTÁZKA:
{q.question}

ODPOVĚĎ:
"""

    result = llm(prompt)[0]["generated_text"]

    return {"answer": result}