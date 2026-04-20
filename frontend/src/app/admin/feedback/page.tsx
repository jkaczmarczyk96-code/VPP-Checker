"use client";

import { useEffect, useMemo, useState } from "react";

type FeedbackItem = {
  id: number;
  created_at: string;
  question: string;
  answer: string;
  rating: "up" | "down";
  comment: string;
  insurer: string;
  document_title: string;
};

type Insight = {
  category: string;
  count: number;
};

export default function FeedbackPage() {
  const api = process.env.NEXT_PUBLIC_API_URL;

  const [items, setItems] =
    useState<FeedbackItem[]>([]);

  const [loading, setLoading] =
    useState(true);

  const [filter, setFilter] =
    useState<"all" | "up" | "down">("all");

  const [search, setSearch] =
    useState("");

  const [selected, setSelected] =
    useState<FeedbackItem | null>(null);

  const [selectedIds, setSelectedIds] =
    useState<number[]>([]);

  const [insights, setInsights] =
    useState<Insight[]>([]);

  useEffect(() => {
    loadFeedback();
  }, []);

  async function loadFeedback() {
    try {
      const res = await fetch(
        `${api}/feedback`,
        {
          credentials:
            "include",
        }
      );

      const data =
        await res.json();

      setItems(
        data.items || []
      );
    } catch {}

    setLoading(false);
  }

  const filteredItems =
    useMemo(() => {
      const q =
        search.toLowerCase();

      return items.filter(
        (item) => {
          const ratingOk =
            filter === "all"
              ? true
              : item.rating ===
                filter;

          const text = `
${item.question}
${item.answer}
${item.comment}
${item.insurer}
${item.document_title}
          `.toLowerCase();

          const searchOk =
            !q ||
            text.includes(q);

          return (
            ratingOk &&
            searchOk
          );
        }
      );
    }, [
      items,
      filter,
      search,
    ]);

  const total =
    filteredItems.length;

  const positive =
    filteredItems.filter(
      (x) =>
        x.rating === "up"
    ).length;

  const negative =
    filteredItems.filter(
      (x) =>
        x.rating === "down"
    ).length;

  const successRate =
    total === 0
      ? 0
      : Math.round(
          (positive /
            total) *
            100
        );

  const worstDocs =
    useMemo(() => {
      const map:
        Record<
          string,
          number
        > = {};

      filteredItems
        .filter(
          (x) =>
            x.rating ===
            "down"
        )
        .forEach(
          (item) => {
            const key =
              item.document_title ||
              "Bez názvu";

            map[key] =
              (map[key] ||
                0) + 1;
          }
        );

      return Object.entries(
        map
      )
        .sort(
          (
            a,
            b
          ) =>
            b[1] -
            a[1]
        )
        .slice(0, 5);
    }, [filteredItems]);

  function runAIInsights() {
    const negativeItems =
      filteredItems.filter(
        (x) =>
          x.rating ===
          "down"
      );

    const categories:
      Record<
        string,
        number
      > = {
      "Nepřesná odpověď": 0,
      "Chybějící zdroj": 0,
      "Příliš obecné": 0,
      "Špatné UI": 0,
      "Jiné": 0,
    };

    negativeItems.forEach(
      (item) => {
        const text =
          `${item.comment} ${item.answer}`.toLowerCase();

        if (
          text.includes(
            "zdroj"
          ) ||
          text.includes(
            "citace"
          )
        ) {
          categories[
            "Chybějící zdroj"
          ]++;
        } else if (
          text.includes(
            "obec"
          ) ||
          text.includes(
            "málo"
          )
        ) {
          categories[
            "Příliš obecné"
          ]++;
        } else if (
          text.includes(
            "špat"
          ) ||
          text.includes(
            "nefung"
          )
        ) {
          categories[
            "Špatné UI"
          ]++;
        } else if (
          text.includes(
            "chyb"
          ) ||
          text.includes(
            "nepřes"
          )
        ) {
          categories[
            "Nepřesná odpověď"
          ]++;
        } else {
          categories[
            "Jiné"
          ]++;
        }
      }
    );

    const result =
      Object.entries(
        categories
      )
        .map(
          (
            item
          ) => ({
            category:
              item[0],
            count:
              item[1],
          })
        )
        .filter(
          (x) =>
            x.count > 0
        )
        .sort(
          (
            a,
            b
          ) =>
            b.count -
            a.count
        );

    setInsights(
      result
    );
  }

  function toggleRow(
    id: number
  ) {
    setSelectedIds(
      (prev) =>
        prev.includes(
          id
        )
          ? prev.filter(
              (
                x
              ) =>
                x !==
                id
            )
          : [
              ...prev,
              id,
            ]
    );
  }

  function toggleAll() {
    if (
      selectedIds.length ===
      filteredItems.length
    ) {
      setSelectedIds(
        []
      );
    } else {
      setSelectedIds(
        filteredItems.map(
          (x) =>
            x.id
        )
      );
    }
  }

  function bulkDelete() {
    const ok =
      window.confirm(
        `Smazat ${selectedIds.length} feedbacků?`
      );

    if (!ok) return;

    setItems(
      (prev) =>
        prev.filter(
          (
            x
          ) =>
            !selectedIds.includes(
              x.id
            )
        )
    );

    setSelectedIds(
      []
    );
  }

  function tabClass(
    value:
      | "all"
      | "up"
      | "down"
  ) {
    return filter ===
      value
      ? "px-4 py-2 rounded-xl bg-white text-black text-sm"
      : "px-4 py-2 rounded-xl bg-zinc-900 border border-zinc-800 text-sm text-zinc-300";
  }

  return (
    <main className="min-h-screen bg-zinc-950 text-white p-6">
      <div className="max-w-7xl mx-auto">

        <div className="mb-8">
          <h1 className="text-3xl font-semibold">
            Feedback Intelligence
          </h1>

          <p className="text-zinc-400 mt-2">
            AI řízení kvality
            odpovědí
          </p>
        </div>

        {loading ? (
          <div>
            Načítám...
          </div>
        ) : (
          <>
            {/* KPI */}
            <div className="grid md:grid-cols-4 gap-4 mb-6">
              <Card
                title="Celkem"
                value={String(
                  total
                )}
              />
              <Card
                title="👍"
                value={String(
                  positive
                )}
              />
              <Card
                title="👎"
                value={String(
                  negative
                )}
              />
              <Card
                title="Úspěšnost"
                value={`${successRate}%`}
              />
            </div>

            {/* TOOLS */}
            <div className="flex flex-col md:flex-row gap-4 mb-6">

              <div className="flex gap-3">
                <button
                  onClick={() =>
                    setFilter(
                      "all"
                    )
                  }
                  className={tabClass(
                    "all"
                  )}
                >
                  Vše
                </button>

                <button
                  onClick={() =>
                    setFilter(
                      "up"
                    )
                  }
                  className={tabClass(
                    "up"
                  )}
                >
                  👍
                </button>

                <button
                  onClick={() =>
                    setFilter(
                      "down"
                    )
                  }
                  className={tabClass(
                    "down"
                  )}
                >
                  👎
                </button>
              </div>

              <input
                value={search}
                onChange={(
                  e
                ) =>
                  setSearch(
                    e.target
                      .value
                  )
                }
                placeholder="Hledat..."
                className="flex-1 rounded-xl bg-zinc-900 border border-zinc-800 px-4 py-2 text-sm"
              />

              <button
                onClick={
                  runAIInsights
                }
                className="rounded-xl bg-white text-black px-4 py-2 text-sm"
              >
                AI Insights
              </button>

              <button
                onClick={
                  bulkDelete
                }
                disabled={
                  selectedIds.length ===
                  0
                }
                className="rounded-xl bg-red-600 px-4 py-2 text-sm disabled:opacity-40"
              >
                Smazat vybrané
              </button>
            </div>

            {/* INSIGHTS */}
            {insights.length >
              0 && (
              <div className="rounded-2xl border border-zinc-800 bg-zinc-900 p-6 mb-6">
                <h2 className="text-xl font-semibold mb-4">
                  AI Root Cause
                </h2>

                <div className="space-y-3">
                  {insights.map(
                    (
                      item
                    ) => (
                      <BarRow
                        key={
                          item.category
                        }
                        label={
                          item.category
                        }
                        value={
                          item.count
                        }
                        max={
                          insights[0]
                            ?.count ||
                          1
                        }
                      />
                    )
                  )}
                </div>
              </div>
            )}

            {/* WORST DOCS */}
            <div className="rounded-2xl border border-zinc-800 bg-zinc-900 p-6 mb-6">
              <h2 className="text-xl font-semibold mb-4">
                Nejproblemovější dokumenty
              </h2>

              <div className="space-y-3">
                {worstDocs.map(
                  (
                    item
                  ) => (
                    <BarRow
                      key={
                        item[0]
                      }
                      label={
                        item[0]
                      }
                      value={
                        item[1]
                      }
                      max={
                        worstDocs[0]?.[1] ||
                        1
                      }
                    />
                  )
                )}
              </div>
            </div>

            {/* TABLE */}
            <div className="overflow-x-auto rounded-2xl border border-zinc-800">
              <table className="w-full text-sm">
                <thead className="bg-zinc-900">
                  <tr className="text-left">
                    <th className="p-4">
                      <input
                        type="checkbox"
                        checked={
                          selectedIds.length ===
                            filteredItems.length &&
                          filteredItems.length >
                            0
                        }
                        onChange={
                          toggleAll
                        }
                      />
                    </th>

                    <th className="p-4">
                      Datum
                    </th>

                    <th className="p-4">
                      Rating
                    </th>

                    <th className="p-4">
                      Dotaz
                    </th>
                  </tr>
                </thead>

                <tbody>
                  {filteredItems.map(
                    (
                      item
                    ) => (
                      <tr
                        key={
                          item.id
                        }
                        className="border-t border-zinc-800 hover:bg-zinc-900"
                      >
                        <td className="p-4">
                          <input
                            type="checkbox"
                            checked={selectedIds.includes(
                              item.id
                            )}
                            onChange={() =>
                              toggleRow(
                                item.id
                              )
                            }
                          />
                        </td>

                        <td
                          className="p-4 cursor-pointer"
                          onClick={() =>
                            setSelected(
                              item
                            )
                          }
                        >
                          {new Date(
                            item.created_at
                          ).toLocaleString(
                            "cs-CZ"
                          )}
                        </td>

                        <td className="p-4 text-xl">
                          {item.rating ===
                          "up"
                            ? "👍"
                            : "👎"}
                        </td>

                        <td
                          className="p-4 truncate max-w-xl cursor-pointer"
                          onClick={() =>
                            setSelected(
                              item
                            )
                          }
                        >
                          {
                            item.question
                          }
                        </td>
                      </tr>
                    )
                  )}
                </tbody>
              </table>
            </div>

            {/* MODAL */}
            {selected && (
              <div className="fixed inset-0 bg-black/70 flex items-center justify-center z-50 p-6">
                <div className="w-full max-w-4xl rounded-2xl border border-zinc-800 bg-zinc-900 p-6 max-h-[90vh] overflow-y-auto">

                  <div className="flex justify-between mb-6">
                    <h2 className="text-xl font-semibold">
                      Detail feedbacku
                    </h2>

                    <button
                      onClick={() =>
                        setSelected(
                          null
                        )
                      }
                    >
                      Zavřít
                    </button>
                  </div>

                  <Block
                    title="Dotaz"
                    text={
                      selected.question
                    }
                  />

                  <Block
                    title="AI odpověď"
                    text={
                      selected.answer
                    }
                  />

                  <Block
                    title="Komentář"
                    text={
                      selected.comment ||
                      "-"
                    }
                  />
                </div>
              </div>
            )}
          </>
        )}
      </div>
    </main>
  );
}

/* COMPONENTS */

function Card({
  title,
  value,
}: {
  title: string;
  value: string;
}) {
  return (
    <div className="rounded-2xl border border-zinc-800 bg-zinc-900 p-5">
      <div className="text-sm text-zinc-400">
        {title}
      </div>

      <div className="text-3xl font-semibold mt-2">
        {value}
      </div>
    </div>
  );
}

function BarRow({
  label,
  value,
  max,
}: {
  label: string;
  value: number;
  max: number;
}) {
  const width =
    (value / max) * 100;

  return (
    <div>
      <div className="flex justify-between text-sm mb-1">
        <span>{label}</span>
        <span>{value}</span>
      </div>

      <div className="h-2 rounded-full bg-zinc-800 overflow-hidden">
        <div
          className="h-full bg-white"
          style={{
            width: `${width}%`,
          }}
        />
      </div>
    </div>
  );
}

function Block({
  title,
  text,
}: {
  title: string;
  text: string;
}) {
  return (
    <div className="mb-5">
      <div className="text-sm text-zinc-400 mb-2">
        {title}
      </div>

      <div className="rounded-xl bg-zinc-950 p-4 whitespace-pre-wrap">
        {text}
      </div>
    </div>
  );
}