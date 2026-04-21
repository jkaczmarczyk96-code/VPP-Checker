"use client";

import { useEffect, useRef, useState } from "react";
import ReactMarkdown from "react-markdown";

type Message = {
  role: "user" | "ai";
  text: string;
  sources?: {
    document?: string;
    section?: string;
    quote?: string;
  }[];
};

type ChatResponse = {
  answer?: string;
  source_document?: string;
  source_section?: string;
  source_quote?: string;
};

type DocumentsApiItem = {
  document_title?: string;
};

const WELCOME_MESSAGE: Message = {
  role: "ai",
  text: "Dobrý den. Jsem připraven analyzovat vaše pojistné podmínky a najít přesné informace.",
};

function normalizeMessageText(text: string) {
  if (!text) return "";

  try {
    const parsed = JSON.parse(text);

    if (parsed?.answer) {
      return parsed.answer;
    }

    return text;
  } catch {
    return text;
  }
}

export default function Home() {
  const api = process.env.NEXT_PUBLIC_API_URL;

  const [question, setQuestion] = useState("");
  const [loading, setLoading] = useState(false);
  const [thinking, setThinking] = useState(false);

  const [insurer, setInsurer] = useState("");
  const [documentName, setDocumentName] = useState("");

  const [insurers, setInsurers] = useState<string[]>([]);
  const [documents, setDocuments] = useState<string[]>([]);

  const [messages, setMessages] =
    useState<Message[]>([WELCOME_MESSAGE]);

  const [error, setError] = useState("");

  const [copiedIndex, setCopiedIndex] =
    useState<number | null>(null);

  const [ratedIndexes, setRatedIndexes] =
    useState<number[]>([]);

  const [openCommentIndex, setOpenCommentIndex] =
    useState<number | null>(null);

  const [feedbackComment, setFeedbackComment] =
    useState("");

  const [toast, setToast] =
    useState("");

  const [feedbackLoading, setFeedbackLoading] =
    useState<number | null>(null);

  const chatRef = useRef<HTMLDivElement>(null);

  const textareaRef =
    useRef<HTMLTextAreaElement>(null);

  /* ================= LOAD ================= */

  useEffect(() => {
    loadInsurers();
  }, []);

  useEffect(() => {
    loadDocuments();
    setDocumentName("");
  }, [insurer]);

  useEffect(() => {
    chatRef.current?.scrollTo({
      top: chatRef.current.scrollHeight,
      behavior: "smooth",
    });
  }, [messages, thinking]);

  async function loadInsurers() {
    try {
      const res = await fetch(
        `${api}/api/v1/public/insurers`
      );

      const data = await res.json();

      setInsurers(
        Array.isArray(data.insurers)
          ? data.insurers
          : []
      );
    } catch (e) {
      console.error(e);

      setError(
        "Nepodařilo se načíst pojišťovny."
      );
    }
  }

  async function loadDocuments() {
    if (!insurer) {
      setDocuments([]);
      return;
    }

    try {
      const res = await fetch(
        `${api}/api/v1/public/documents?insurer=${encodeURIComponent(
          insurer
        )}`
      );

      const data = await res.json();

      const docs =
        (data.documents ||
          []) as DocumentsApiItem[];

      const titles = Array.from(
        new Set(
          docs.map((d) =>
            String(
              d.document_title ||
                ""
            )
          )
        )
      ).filter(
        (item): item is string =>
          item.trim() !== ""
      );

      setDocuments(titles);
    } catch (e) {
      console.error(e);

      setError(
        "Nepodařilo se načíst dokumenty."
      );
    }
  }

  /* ================= CHAT ================= */

  async function sendQuestion() {
    if (
      !question.trim() ||
      !insurer ||
      !documentName ||
      loading
    ) {
      return;
    }

    const userText = question.trim();

    const history = [
      ...messages,
      {
        role: "user" as const,
        text: userText,
      },
    ];

    setMessages(history);
    setQuestion("");
    setLoading(true);
    setThinking(true);
    setError("");

    try {
      const res = await fetch(
        `${api}/api/v1/chat/`,
        {
          method: "POST",
          headers: {
            "Content-Type":
              "application/json",
          },
          body: JSON.stringify({
            question: userText,
            insurer,
            document_title:
              documentName,
            messages: history,
          }),
        }
      );

      const data: ChatResponse =
        await res.json();

      setMessages((prev) => [
        ...prev,
        {
          role: "ai",
          text:
            normalizeMessageText(
              data.answer ||
                "Bez odpovědi."
            ),
          sources:
            data.source_document
              ? [
                  {
                    document:
                      data.source_document,
                    section:
                      data.source_section,
                    quote:
                      data.source_quote,
                  },
                ]
              : [],
        },
      ]);
    } catch (e) {
      console.error(e);

      setMessages((prev) => [
        ...prev,
        {
          role: "ai",
          text:
            "Chyba spojení se serverem.",
        },
      ]);
    } finally {
      setLoading(false);
      setThinking(false);
      textareaRef.current?.focus();
    }
  }

  /* ================= TOAST ================= */

  function showToast(text: string) {
    setToast(text);

    setTimeout(() => {
      setToast("");
    }, 1800);
  }

  /* ================= FEEDBACK ================= */

  async function sendFeedback(
    index: number,
    rating: "up" | "down",
    comment = ""
  ) {
    if (
      ratedIndexes.includes(index) ||
      feedbackLoading === index
    ) {
      return;
    }

    const aiMessage = messages[index];

    const userMessage =
      messages[index - 1];

    try {
      setFeedbackLoading(index);

      await fetch(
        `${api}/api/v1/feedback`,
        {
          method: "POST",
          headers: {
            "Content-Type":
              "application/json",
          },
          body: JSON.stringify({
            question:
              userMessage?.text || "",
            answer:
              aiMessage?.text || "",
            rating,
            comment,
            insurer,
            document_title:
              documentName,
          }),
        }
      );

      setRatedIndexes((prev) => [
        ...prev,
        index,
      ]);

      setOpenCommentIndex(null);
      setFeedbackComment("");

      showToast(
        rating === "up"
          ? "Děkujeme za feedback."
          : "Feedback byl odeslán."
      );
    } catch (e) {
      console.error(e);

      showToast(
        "Feedback se nepodařilo odeslat."
      );
    } finally {
      setFeedbackLoading(null);
    }
  }

  function copyText(
    text: string,
    index: number
  ) {
    navigator.clipboard.writeText(
      text
    );

    setCopiedIndex(index);

    showToast("Zkopírováno.");

    setTimeout(() => {
      setCopiedIndex(null);
    }, 1500);
  }

  const canSend =
    question.trim() &&
    insurer &&
    documentName &&
    !loading;

  /* ================= UI ================= */

  return (
    <main className="min-h-screen bg-zinc-950 text-white flex flex-col">
      <div className="mx-auto w-full max-w-5xl px-4 md:px-6 flex flex-col h-screen">

        <header className="py-6 border-b border-zinc-800">
          <h1 className="text-3xl font-semibold">
            VPP Checker
          </h1>

          <p className="text-zinc-400 mt-2 text-sm">
            AI analýza pojistných
            podmínek
          </p>

          <div className="grid md:grid-cols-2 gap-4 py-4">

            <select
              value={insurer}
              onChange={(e) =>
                setInsurer(
                  e.target.value
                )
              }
              className="h-12 rounded-xl bg-zinc-900 border border-zinc-800 px-4 text-sm"
            >
              <option value="">
                Vyberte pojišťovnu
              </option>

              {insurers.map(
                (item) => (
                  <option
                    key={item}
                    value={item}
                  >
                    {item}
                  </option>
                )
              )}
            </select>

            <select
              value={documentName}
              disabled={!insurer}
              onChange={(e) =>
                setDocumentName(
                  e.target.value
                )
              }
              className="h-12 rounded-xl bg-zinc-900 border border-zinc-800 px-4 text-sm disabled:opacity-40"
            >
              <option value="">
                Vyberte dokument
              </option>

              {documents.map(
                (item) => (
                  <option
                    key={item}
                    value={item}
                  >
                    {item}
                  </option>
                )
              )}
            </select>

          </div>

          {error && (
            <div className="mt-2 text-sm text-red-400">
              {error}
            </div>
          )}
        </header>

        <section
          ref={chatRef}
          className="flex-1 overflow-y-auto py-8 space-y-6"
        >

          {messages.map(
            (msg, index) => (
              <div
                key={index}
                className={`flex ${
                  msg.role ===
                  "user"
                    ? "justify-end"
                    : "justify-start"
                }`}
              >
                <div
                  className={
                    msg.role ===
                    "user"
                      ? "max-w-[80%] rounded-3xl bg-zinc-800 px-5 py-4"
                      : "max-w-[90%] rounded-3xl bg-zinc-900 border border-zinc-800 px-6 py-5"
                  }
                >

                  {msg.role ===
                    "ai" && (
                    <div className="flex justify-between mb-3 text-xs text-zinc-400">

                      <span>
                        🤖 VPP Asistent
                      </span>

                      <div className="flex gap-3">

                        <button
                          disabled={
                            ratedIndexes.includes(
                              index
                            ) ||
                            feedbackLoading ===
                              index
                          }
                          onClick={() =>
                            sendFeedback(
                              index,
                              "up"
                            )
                          }
                          className="hover:text-white disabled:opacity-40 disabled:cursor-not-allowed active:scale-95 transition"
                        >
                          {feedbackLoading ===
                          index
                            ? "..."
                            : "👍"}
                        </button>

                        <button
                          disabled={
                            ratedIndexes.includes(
                              index
                            ) ||
                            feedbackLoading ===
                              index
                          }
                          onClick={() =>
                            setOpenCommentIndex(
                              index
                            )
                          }
                          className="hover:text-white disabled:opacity-40 disabled:cursor-not-allowed active:scale-95 transition"
                        >
                          👎
                        </button>

                        <button
                          onClick={() =>
                            copyText(
                              msg.text,
                              index
                            )
                          }
                          className="hover:text-white active:scale-95 transition"
                        >
                          {copiedIndex ===
                          index
                            ? "✓"
                            : "⧉"}
                        </button>

                      </div>
                    </div>
                  )}

                  <div className="prose prose-invert max-w-none">
                    <ReactMarkdown>
                      {msg.text}
                    </ReactMarkdown>
                  </div>

                  {msg.sources &&
                    msg.sources.length >
                      0 && (
                      <div className="mt-4 space-y-2">
                        {msg.sources.map(
                          (
                            s,
                            i
                          ) => (
                            <details
                              key={i}
                              className="rounded-xl border border-zinc-800 bg-zinc-950 p-3"
                            >
                              <summary className="cursor-pointer text-sm text-zinc-400">
                                📄 Zdroj{" "}
                                {i + 1}
                              </summary>

                              <div className="mt-2 text-sm text-zinc-300 space-y-2">

                                {s.document && (
                                  <div>
                                    {
                                      s.document
                                    }
                                  </div>
                                )}

                                {s.section && (
                                  <div>
                                    {
                                      s.section
                                    }
                                  </div>
                                )}

                                {s.quote && (
                                  <div className="p-3 bg-zinc-900 border border-zinc-800 rounded-lg whitespace-pre-wrap">
                                    {
                                      s.quote
                                    }
                                  </div>
                                )}

                              </div>
                            </details>
                          )
                        )}
                      </div>
                    )}

                  {openCommentIndex ===
                    index && (
                    <div className="mt-4 space-y-2">

                      <textarea
                        value={
                          feedbackComment
                        }
                        onChange={(e) =>
                          setFeedbackComment(
                            e.target.value
                          )
                        }
                        placeholder="Co bylo špatně?"
                        className="w-full rounded-xl bg-zinc-950 border border-zinc-800 p-3 text-sm"
                      />

                      <button
                        disabled={
                          feedbackLoading ===
                          index
                        }
                        onClick={() =>
                          sendFeedback(
                            index,
                            "down",
                            feedbackComment
                          )
                        }
                        className="bg-white text-black px-4 py-2 rounded-xl text-sm active:scale-95 transition disabled:opacity-40"
                      >
                        Odeslat feedback
                      </button>

                    </div>
                  )}

                </div>
              </div>
            )
          )}

          {thinking && (
            <div className="flex justify-start">
              <div className="rounded-3xl bg-zinc-900 border border-zinc-800 px-5 py-4 text-zinc-400 animate-pulse">
                VPP Asistent
                přemýšlí...
              </div>
            </div>
          )}

        </section>

        <footer className="py-5 border-t border-zinc-800 flex gap-3">

          <textarea
            ref={textareaRef}
            rows={1}
            value={question}
            onChange={(e) =>
              setQuestion(
                e.target.value
              )
            }
            onKeyDown={(e) => {
              if (
                e.key ===
                  "Enter" &&
                !e.shiftKey
              ) {
                e.preventDefault();
                sendQuestion();
              }
            }}
            placeholder="Napište dotaz..."
            className="flex-1 rounded-3xl bg-zinc-900 border border-zinc-800 px-5 py-3 resize-none"
          />

          <button
            disabled={!canSend}
            onClick={sendQuestion}
            className="px-6 rounded-3xl bg-white text-black disabled:opacity-40 active:scale-95 transition"
          >
            {loading
              ? "..."
              : "Odeslat"}
          </button>

        </footer>

      </div>

      {toast && (
        <div className="fixed bottom-6 left-1/2 -translate-x-1/2 z-50 rounded-2xl bg-white text-black px-5 py-3 text-sm shadow-xl">
          {toast}
        </div>
      )}
    </main>
  );
}
