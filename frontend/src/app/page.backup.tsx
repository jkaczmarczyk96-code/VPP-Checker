"use client";

import { useEffect, useRef, useState } from "react";
import ReactMarkdown from "react-markdown";

type Message = {
  role: "user" | "ai";
  text: string;
  source?: string;
  sourceSection?: string;
  sourceArticle?: string;
  sourceParagraph?: string;
  sourceQuote?: string;
};

type ChatResponse = {
  answer?: string;
  source_document?: string;
  source_section?: string;
  source_article?: string;
  source_paragraph?: string;
  source_quote?: string;
};

type InsurersResponse = {
  insurers?: string[];
};

type DocumentsResponse = {
  documents?: {
    document_title: string;
  }[];
};

const DEFAULT_MESSAGE: Message = {
  role: "ai",
  text: "Dobrý den. Jsem připraven projít vaše pojistné podmínky a najít v nich podstatné informace.",
};

export default function Home() {
  const [question, setQuestion] = useState("");
  const [loading, setLoading] = useState(false);

  const [insurer, setInsurer] = useState("");
  const [documentName, setDocumentName] = useState("");

  const [insurers, setInsurers] = useState<string[]>([]);
  const [documents, setDocuments] = useState<string[]>([]);

  const [messages, setMessages] = useState<Message[]>([
    DEFAULT_MESSAGE,
  ]);

  const [copiedIndex, setCopiedIndex] =
    useState<number | null>(null);

  const chatRef =
    useRef<HTMLDivElement>(null);

  const inputRef =
    useRef<HTMLInputElement>(null);

  const api =
    process.env.NEXT_PUBLIC_API_URL;

  useEffect(() => {
    chatRef.current?.scrollTo({
      top: chatRef.current.scrollHeight,
      behavior: "smooth",
    });
  }, [messages, loading]);

  useEffect(() => {
    inputRef.current?.focus();
  }, [loading]);

  const cleanSection = (
    value?: string
  ) => {
    if (!value) return "";

    const text = value.trim();

    if (
      text.toLowerCase() ===
        "neuvedeno" ||
      text.length < 4
    ) {
      return "";
    }

    return text.replace(/\s+/g, " ");
  };

  useEffect(() => {
    const loadInsurers =
      async () => {
        try {
          const res = await fetch(
            `${api}/admin/insurers`,
            {
              credentials:
                "include",
            }
          );

          const data: InsurersResponse =
            await res.json();

          setInsurers(
            data.insurers || []
          );
        } catch {}
      };

    loadInsurers();
  }, [api]);

  useEffect(() => {
    const loadDocuments =
      async () => {
        if (!insurer) {
          setDocuments([]);
          setDocumentName("");
          return;
        }

        try {
          const res = await fetch(
            `${api}/admin/documents?insurer=${encodeURIComponent(
              insurer
            )}`,
            {
              credentials:
                "include",
            }
          );

          const data: DocumentsResponse =
            await res.json();

          const titles =
            Array.from(
              new Set(
                (
                  data.documents ||
                  []
                ).map(
                  (d) =>
                    d.document_title
                )
              )
            );

          setDocuments(
            titles
          );
        } catch {}
      };

    loadDocuments();
  }, [api, insurer]);

  async function sendQuestion() {
    if (!question.trim())
      return;

    if (!insurer ||
        !documentName)
      return;

    if (loading) return;

    const userText =
      question.trim();

    const updated: Message[] =
      [
        ...messages,
        {
          role: "user",
          text: userText,
        },
      ];

    setMessages(updated);
    setQuestion("");
    setLoading(true);

    try {
      const res = await fetch(
        `${api}/chat/`,
        {
          method: "POST",
          headers: {
            "Content-Type":
              "application/json",
          },
          body: JSON.stringify(
            {
              question:
                userText,
              insurer,
              document_title:
                documentName,
              messages:
                updated,
            }
          ),
        }
      );

      const data: ChatResponse =
        await res.json();

      setMessages(
        (prev) => [
          ...prev,
          {
            role: "ai",
            text:
              data.answer ||
              "Nepodařilo se získat odpověď.",
            source:
              data.source_document ||
              "",
            sourceSection:
              cleanSection(
                data.source_section
              ),
            sourceArticle:
              data.source_article ||
              "",
            sourceParagraph:
              data.source_paragraph ||
              "",
            sourceQuote:
              data.source_quote ||
              "",
          },
        ]
      );
    } catch {
      setMessages(
        (prev) => [
          ...prev,
          {
            role: "ai",
            text: "Nepodařilo se spojit se serverem.",
          },
        ]
      );
    } finally {
      setLoading(false);
    }
  }

  return (
    <main className="min-h-screen bg-gradient-to-b from-white to-gray-50 text-gray-900">
      <div className="max-w-6xl mx-auto px-5 py-8">

        {/* Header */}
        <div className="mb-8">
          <div className="inline-flex items-center gap-2 rounded-full border bg-white px-4 py-2 text-sm shadow-sm">
            <span>✨</span>
            <span className="font-medium">
              VPP Checker
            </span>
          </div>

          <h1 className="text-4xl font-semibold mt-5 tracking-tight">
            AI asistent pro
            pojistné podmínky
          </h1>

          <p className="text-gray-500 mt-2 text-lg">
            Rychlé odpovědi,
            přesné citace,
            profesionální výstup.
          </p>
        </div>

        {/* Filters */}
        <div className="grid md:grid-cols-2 gap-4 mb-6">
          <select
            value={insurer}
            onChange={(e) =>
              setInsurer(
                e.target.value
              )
            }
            className="h-14 rounded-2xl border bg-white px-4 shadow-sm"
          >
            <option value="">
              Vyberte pojišťovnu
            </option>

            {insurers.map(
              (item, i) => (
                <option
                  key={i}
                  value={item}
                >
                  {item}
                </option>
              )
            )}
          </select>

          <select
            value={documentName}
            onChange={(e) =>
              setDocumentName(
                e.target.value
              )
            }
            className="h-14 rounded-2xl border bg-white px-4 shadow-sm"
          >
            <option value="">
              Vyberte dokument
            </option>

            {documents.map(
              (item, i) => (
                <option
                  key={i}
                  value={item}
                >
                  {item}
                </option>
              )
            )}
          </select>
        </div>

        {/* Chat */}
        <div
          ref={chatRef}
          className="h-[650px] overflow-y-auto rounded-3xl border bg-white shadow-sm px-6 py-6 space-y-6"
        >
          {messages.map(
            (
              msg,
              index
            ) => (
              <div
                key={index}
                className={
                  msg.role ===
                  "user"
                    ? "flex justify-end"
                    : "flex justify-start"
                }
              >
                <div
                  className={
                    msg.role ===
                    "user"
                      ? "max-w-[75%] rounded-3xl bg-black text-white px-5 py-4 shadow-sm"
                      : "max-w-[88%] rounded-3xl border bg-white px-6 py-5 shadow-sm hover:shadow-md transition"
                  }
                >
                  {msg.role ===
                    "ai" && (
                    <div className="flex items-center justify-between mb-4">
                      <div className="flex items-center gap-2 text-sm text-gray-500">
                        <span>
                          ✨
                        </span>
                        <span className="font-medium">
                          VPP
                          Asistent
                        </span>
                      </div>

                      <button
                        onClick={() => {
                          navigator.clipboard.writeText(
                            msg.text
                          );

                          setCopiedIndex(
                            index
                          );

                          setTimeout(
                            () =>
                              setCopiedIndex(
                                null
                              ),
                            1500
                          );
                        }}
                        className="text-xs text-gray-400 hover:text-black transition"
                      >
                        {copiedIndex ===
                        index
                          ? "Zkopírováno"
                          : "Kopírovat"}
                      </button>
                    </div>
                  )}

                  <div className="prose prose-base max-w-none prose-headings:font-semibold prose-headings:mt-6 prose-headings:mb-3 prose-p:mb-4 prose-li:mb-1 prose-strong:font-semibold leading-7">
                    <ReactMarkdown>
                      {msg.text}
                    </ReactMarkdown>
                  </div>

                  {msg.role ===
                    "ai" &&
                    msg.source && (
                      <div className="mt-6 border-t pt-5">
                        <details>
                          <summary className="cursor-pointer text-sm font-medium text-gray-700 hover:text-black">
                            📄 Zdroj a citace
                          </summary>

                          <div className="mt-4 space-y-2 text-sm text-gray-600">
                            <div>
                              <strong>
                                Zdroj:
                              </strong>{" "}
                              {
                                msg.source
                              }
                            </div>

                            {msg.sourceSection && (
                              <div>
                                <strong>
                                  Sekce:
                                </strong>{" "}
                                {
                                  msg.sourceSection
                                }
                              </div>
                            )}

                            {msg.sourceArticle && (
                              <div>
                                <strong>
                                  Článek:
                                </strong>{" "}
                                {
                                  msg.sourceArticle
                                }
                              </div>
                            )}

                            {msg.sourceParagraph && (
                              <div>
                                <strong>
                                  Odstavec:
                                </strong>{" "}
                                {
                                  msg.sourceParagraph
                                }
                              </div>
                            )}

                            {msg.sourceQuote && (
                              <div className="mt-3 rounded-2xl border bg-gray-50 p-4 whitespace-pre-wrap leading-6">
                                {
                                  msg.sourceQuote
                                }
                              </div>
                            )}
                          </div>
                        </details>
                      </div>
                    )}
                </div>
              </div>
            )
          )}

          {loading && (
            <div className="flex justify-start">
              <div className="rounded-3xl border bg-white px-5 py-4 shadow-sm text-gray-500">
                <div className="flex items-center gap-2">
                  <span className="animate-pulse">
                    ✨
                  </span>
                  <span>
                    Analyzuji podmínky...
                  </span>
                </div>
              </div>
            </div>
          )}
        </div>

        {/* Input */}
        <div className="mt-6 flex gap-3">
          <input
            ref={inputRef}
            value={question}
            onChange={(e) =>
              setQuestion(
                e.target.value
              )
            }
            onKeyDown={(e) => {
              if (
                e.key ===
                "Enter"
              ) {
                sendQuestion();
              }
            }}
            placeholder="Napište dotaz k pojistným podmínkám..."
            className="h-14 flex-1 rounded-2xl border bg-white px-5 shadow-sm outline-none focus:ring-2 focus:ring-black/10"
          />

          <button
            onClick={
              sendQuestion
            }
            disabled={
              loading
            }
            className="h-14 rounded-2xl bg-black px-7 text-white shadow-sm hover:opacity-90 disabled:opacity-50"
          >
            {loading
              ? "Čekám..."
              : "Odeslat"}
          </button>
        </div>
      </div>
    </main>
  );
}