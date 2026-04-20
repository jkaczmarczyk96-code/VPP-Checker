"use client";

import { useEffect, useRef, useState } from "react";
import ReactMarkdown from "react-markdown";

/* ================= TYPES ================= */

type Message = {
  role: "user" | "ai";
  text: string;
  source?: string;
  sourceSection?: string;
  sources?: {
    document?: string;
    section?: string;
    article?: string;
    quote?: string;
  }[];
};

type ChatResponse = {
  answer?: string;
  source_document?: string;
  source_section?: string;
};

type InsurersResponse = {
  insurers?: string[];
};

type DocumentsResponse = {
  documents?: {
    document_title: string;
  }[];
};

/* ================= HELPERS ================= */

function normalizeMessageText(text: string) {
  if (!text) return "";

  try {
    const parsed = JSON.parse(text);
    if (parsed?.answer) return parsed.answer;
    return text;
  } catch {
    return text;
  }
}

/* ================= COMPONENT ================= */

const WELCOME_MESSAGE: Message = {
  role: "ai",
  text: "Dobrý den. Jsem připraven analyzovat vaše pojistné podmínky a najít přesné informace.",
};

export default function Home() {
  const api = process.env.NEXT_PUBLIC_API_URL;

  const [question, setQuestion] = useState("");
  const [loading, setLoading] = useState(false);
  const [thinking, setThinking] = useState(false);

  const [insurer, setInsurer] = useState("");
  const [documentName, setDocumentName] = useState("");

  const [insurers, setInsurers] = useState<string[]>([]);
  const [documents, setDocuments] = useState<string[]>([]);

  const [messages, setMessages] = useState<Message[]>([WELCOME_MESSAGE]);

  const [copiedIndex, setCopiedIndex] = useState<number | null>(null);
  const [ratedIndexes, setRatedIndexes] = useState<number[]>([]);
  const [openCommentIndex, setOpenCommentIndex] = useState<number | null>(null);
  const [feedbackComment, setFeedbackComment] = useState("");

  const chatRef = useRef<HTMLDivElement>(null);
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  /* ================= LOAD ================= */

  useEffect(() => {
    loadInsurers();
  }, []);

  useEffect(() => {
    loadDocuments();
    setDocumentName("");
  }, [insurer]);

  async function loadInsurers() {
    try {
      const res = await fetch(`${api}/admin/insurers`, {
        credentials: "include",
      });
      const data: InsurersResponse = await res.json();
      setInsurers(data.insurers || []);
    } catch {}
  }

  async function loadDocuments() {
    if (!insurer) {
      setDocuments([]);
      return;
    }

    try {
      const res = await fetch(
        `${api}/admin/documents?insurer=${encodeURIComponent(insurer)}`,
        { credentials: "include" }
      );

      const data: DocumentsResponse = await res.json();

      const titles = Array.from(
        new Set((data.documents || []).map((d) => d.document_title))
      );

      setDocuments(titles);
    } catch {}
  }

  /* ================= CHAT ================= */

  async function sendQuestion() {
    if (!question.trim() || !insurer || !documentName || loading) return;

    const userText = question.trim();

    const history = [
      ...messages,
      { role: "user" as const, text: userText },
    ];

    setMessages(history);
    setQuestion("");
    setLoading(true);
    setThinking(true);

    try {
      const res = await fetch(`${api}/chat/`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          question: userText,
          insurer,
          document_title: documentName,
          messages: history,
          stream: true,
        }),
      });

      /* ================= FALLBACK ================= */

      if (!res.body) {
        const data: ChatResponse = await res.json();

        setMessages((prev) => [
          ...prev,
          {
            role: "ai",
            text: normalizeMessageText(data.answer || "Bez odpovědi."),
            sources: data.source_document
              ? [
                  {
                    document: data.source_document,
                    section: data.source_section,
                  },
                ]
              : [],
          },
        ]);

        setLoading(false);
        setThinking(false);
        return;
      }

      /* ================= STREAM ================= */

      const reader = res.body.getReader();
      const decoder = new TextDecoder("utf-8");

      let fullText = "";
      let finalSources: any[] = [];

      setMessages((prev) => [
        ...prev,
        {
          role: "ai",
          text: "",
          sources: [],
        },
      ]);

      while (true) {
        const { value, done } = await reader.read();
        if (done) break;

        fullText += decoder.decode(value, { stream: true });

        setMessages((prev) => {
          const copy = [...prev];

          copy[copy.length - 1] = {
            role: "ai",
            text: normalizeMessageText(fullText),
            sources: finalSources,
          };

          return copy;
        });
      }

      /* ================= FINAL SOURCE FIX ================= */
      try {
        const parsed = JSON.parse(fullText);

        if (parsed?.source_document) {
          finalSources = [
            {
              document: parsed.source_document,
              section: parsed.source_section,
              quote: parsed.source_quote,
            },
          ];
        }

        setMessages((prev) => {
          const copy = [...prev];
          copy[copy.length - 1] = {
            ...copy[copy.length - 1],
            sources: finalSources,
          };
          return copy;
        });
      } catch {}
    } catch {
      setMessages((prev) => [
        ...prev,
        {
          role: "ai",
          text: "Chyba spojení se serverem.",
        },
      ]);
    } finally {
      setLoading(false);
      setThinking(false);
      textareaRef.current?.focus();
    }
  }

  /* ================= FEEDBACK ================= */

  async function sendFeedback(
    index: number,
    rating: "up" | "down",
    comment = ""
  ) {
    if (ratedIndexes.includes(index)) return;

    const aiMessage = messages[index];
    const userMessage = messages[index - 1];

    await fetch(`${api}/feedback`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        question: userMessage?.text || "",
        answer: aiMessage.text,
        rating,
        comment,
        insurer,
        document_title: documentName,
      }),
    });

    setRatedIndexes((prev) => [...prev, index]);
    setOpenCommentIndex(null);
    setFeedbackComment("");
  }

  function copyText(text: string, index: number) {
    navigator.clipboard.writeText(text);
    setCopiedIndex(index);
    setTimeout(() => setCopiedIndex(null), 1500);
  }

  const canSend = question.trim() && insurer && documentName && !loading;

  /* ================= UI ================= */

  return (
    <main className="min-h-screen bg-zinc-950 text-white flex flex-col">
      <div className="mx-auto w-full max-w-5xl px-4 md:px-6 flex flex-col h-screen">

        {/* HEADER */}
        <header className="py-6 border-b border-zinc-800">
          <h1 className="text-3xl font-semibold">VPP Checker</h1>

          <div className="grid md:grid-cols-2 gap-4 py-4">
            <select
              value={insurer}
              onChange={(e) => setInsurer(e.target.value)}
              className="h-12 rounded-xl bg-zinc-900 border border-zinc-800 px-4 text-sm"
            >
              <option value="">Vyberte pojišťovnu</option>
              {insurers.map((item) => (
                <option key={item} value={item}>
                  {item}
                </option>
              ))}
            </select>

            <select
              value={documentName}
              onChange={(e) => setDocumentName(e.target.value)}
              className="h-12 rounded-xl bg-zinc-900 border border-zinc-800 px-4 text-sm"
            >
              <option value="">Vyberte dokument</option>
              {documents.map((item) => (
                <option key={item} value={item}>
                  {item}
                </option>
              ))}
            </select>
          </div>
        </header>

        {/* CHAT */}
        <section ref={chatRef} className="flex-1 overflow-y-auto py-8 space-y-6">

          {messages.map((msg, index) => (
            <div
              key={index}
              className={`flex ${
                msg.role === "user" ? "justify-end" : "justify-start"
              }`}
            >
              <div
                className={
                  msg.role === "user"
                    ? "max-w-[80%] rounded-3xl bg-zinc-800 px-5 py-4"
                    : "max-w-[90%] rounded-3xl bg-zinc-900 border border-zinc-800 px-6 py-5"
                }
              >

                {/* HEADER AI */}
                {msg.role === "ai" && (
                  <div className="flex justify-between mb-3 text-xs text-zinc-400">
                    <span>🤖 VPP Asistent</span>

                    <div className="flex gap-3">
                      <button onClick={() => sendFeedback(index, "up")}>👍</button>
                      <button onClick={() => setOpenCommentIndex(index)}>👎</button>
                      <button onClick={() => copyText(msg.text, index)}>
                        {copiedIndex === index ? "✓" : "⧉"}
                      </button>
                    </div>
                  </div>
                )}

                {/* TEXT */}
                <div className="prose prose-invert max-w-none">
                  <ReactMarkdown>
                    {normalizeMessageText(msg.text)}
                  </ReactMarkdown>
                </div>

                {/* SOURCES */}
               (msg.sources?.length ?? 0) > 0
                  <div className="mt-4 space-y-2">
                    {msg.sources.map((s, i) => (
                      <details
                        key={i}
                        className="rounded-xl border border-zinc-800 bg-zinc-950 p-3"
                      >
                        <summary className="cursor-pointer text-sm text-zinc-400">
                          📄 Zdroj {i + 1}
                        </summary>

                        <div className="mt-2 text-sm text-zinc-300 space-y-2">
                          {s.document && <div>{s.document}</div>}
                          {s.section && <div>{s.section}</div>}
                          {s.quote && (
                            <div className="mt-2 p-3 bg-zinc-900 border border-zinc-800 rounded-lg whitespace-pre-wrap">
                              {s.quote}
                            </div>
                          )}
                        </div>
                      </details>
                    ))}
                  </div>
                )}

                {/* COMMENT */}
                {openCommentIndex === index && (
                  <div className="mt-3 space-y-2">
                    <textarea
                      value={feedbackComment}
                      onChange={(e) => setFeedbackComment(e.target.value)}
                      className="w-full bg-zinc-950 border border-zinc-800 p-2 rounded-xl"
                    />

                    <button
                      onClick={() =>
                        sendFeedback(index, "down", feedbackComment)
                      }
                      className="bg-white text-black px-3 py-1 rounded-lg"
                    >
                      Odeslat
                    </button>
                  </div>
                )}

              </div>
            </div>
          ))}

          {thinking && (
            <div className="flex justify-start">
              <div className="rounded-3xl bg-zinc-900 border border-zinc-800 px-5 py-4 text-zinc-400">
                VPP Asistent přemýšlí...
              </div>
            </div>
          )}

        </section>

        {/* INPUT */}
        <footer className="py-5 border-t border-zinc-800 flex gap-3">
          <textarea
            value={question}
            rows={1}
            onChange={(e) => setQuestion(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "Enter" && !e.shiftKey) {
                e.preventDefault();
                sendQuestion();
              }
            }}
            className="flex-1 rounded-3xl bg-zinc-900 border border-zinc-800 px-5 py-3 resize-none overflow-hidden"
          />

          <button
            disabled={!canSend}
            onClick={sendQuestion}
            className="px-6 rounded-3xl bg-white text-black"
          >
            Odeslat
          </button>
        </footer>

      </div>
    </main>
  );
}
