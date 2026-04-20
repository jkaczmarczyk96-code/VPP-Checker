"use client";

import { useEffect, useState } from "react";
import Link from "next/link";

export default function AdminPage() {
  const api = process.env.NEXT_PUBLIC_API_URL;

  const [file, setFile] = useState<File | null>(null);
  const [message, setMessage] = useState("");
  const [files, setFiles] = useState<any[]>([]);

  const [insurer, setInsurer] = useState("");
  const [documentTitle, setDocumentTitle] = useState("");

  const [insurers, setInsurers] = useState<string[]>([]);
  const [newInsurer, setNewInsurer] = useState("");

  const [password, setPassword] = useState("");
  const [loggedIn, setLoggedIn] = useState(false);
  const [checkingAuth, setCheckingAuth] = useState(true);

  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);

  /* ================= LOAD ================= */

  const loadFiles = async () => {
    const res = await fetch(`${api}/admin/documents`, {
      credentials: "include",
    });

    const data = await res.json();
    setFiles(data.documents || []);
  };

  const loadInsurers = async () => {
    const res = await fetch(`${api}/admin/insurers`, {
      credentials: "include",
    });

    const data = await res.json();
    setInsurers(data.insurers || []);
  };

  const checkAuth = async () => {
    try {
      const res = await fetch(`${api}/admin/me`, {
        credentials: "include",
      });

      if (res.ok) {
        setLoggedIn(true);
        await loadFiles();
        await loadInsurers();
      } else {
        setLoggedIn(false);
      }
    } catch {
      setLoggedIn(false);
    }

    setCheckingAuth(false);
  };

  useEffect(() => {
    checkAuth();
  }, []);

  /* ================= AUTH ================= */

  const handleLogin = async () => {
    if (!password.trim()) {
      setError("Zadej heslo.");
      return;
    }

    try {
      setLoading(true);
      setError("");

      const res = await fetch(`${api}/admin/login`, {
        method: "POST",
        credentials: "include",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ password }),
      });

      const data = await res.json();

      if (!res.ok) {
        setError(data.detail || "Login selhal.");
        setLoading(false);
        return;
      }

      setPassword("");
      setLoggedIn(true);

      await loadFiles();
      await loadInsurers();
    } catch {
      setError("Backend není dostupný.");
    }

    setLoading(false);
  };

  const logout = async () => {
    await fetch(`${api}/admin/logout`, {
      method: "POST",
      credentials: "include",
    });

    setLoggedIn(false);
    setFiles([]);
    setInsurers([]);
  };

  /* ================= CRUD ================= */

  const addInsurer = async () => {
    if (!newInsurer.trim()) return;

    await fetch(`${api}/admin/insurers`, {
      method: "POST",
      credentials: "include",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        name: newInsurer,
      }),
    });

    setNewInsurer("");
    await loadInsurers();
    setMessage("Pojišťovna přidána.");
  };

  const deleteInsurer = async () => {
    if (!insurer) return;

    await fetch(
      `${api}/admin/insurers/${encodeURIComponent(insurer)}`,
      {
        method: "DELETE",
        credentials: "include",
      }
    );

    setInsurer("");
    await loadInsurers();
    setMessage("Pojišťovna smazána.");
  };

  const uploadFile = async () => {
    if (!file) {
      setMessage("Vyber soubor.");
      return;
    }

    if (!insurer.trim()) {
      setMessage("Vyber pojišťovnu.");
      return;
    }

    if (!documentTitle.trim()) {
      setMessage("Vyplň název VPP.");
      return;
    }

    const formData = new FormData();
    formData.append("file", file);
    formData.append("insurer", insurer);
    formData.append("document_title", documentTitle);

    const res = await fetch(`${api}/upload`, {
      method: "POST",
      credentials: "include",
      body: formData,
    });

    const data = await res.json();

    if (res.ok) {
      setMessage("Soubor úspěšně nahrán.");
      setDocumentTitle("");
      setFile(null);
      await loadFiles();
      await loadInsurers();
    } else {
      setMessage(data.detail || "Chyba uploadu.");
    }
  };

  const deleteFile = async (filename: string) => {
    const confirmed = window.confirm(
      `Opravdu smazat soubor "${filename}"?`
    );

    if (!confirmed) return;

    const res = await fetch(`${api}/upload/${filename}`, {
      method: "DELETE",
      credentials: "include",
    });

    if (res.ok) {
      setMessage("Soubor smazán.");
      await loadFiles();
      await loadInsurers();
    } else {
      setMessage("Mazání se nepodařilo.");
    }
  };

  /* ================= UI ================= */

  if (checkingAuth) {
    return (
      <main className="min-h-screen bg-zinc-950 text-white p-8">
        Ověřuji přihlášení...
      </main>
    );
  }

  if (!loggedIn) {
    return (
      <main className="min-h-screen bg-zinc-950 text-white p-8">
        <div className="max-w-xl mx-auto">
          <div className="bg-zinc-900 border border-zinc-800 rounded-2xl p-8">
            <h1 className="text-3xl font-semibold mb-6">
              Admin login
            </h1>

            <input
              type="password"
              value={password}
              placeholder="Heslo"
              onChange={(e) =>
                setPassword(e.target.value)
              }
              className="w-full rounded-xl bg-zinc-950 border border-zinc-800 px-4 py-3"
            />

            <button
              onClick={handleLogin}
              className="mt-4 bg-white text-black px-6 py-3 rounded-xl"
            >
              {loading
                ? "Přihlašuji..."
                : "Přihlásit se"}
            </button>

            {error && (
              <p className="mt-4 text-red-400">
                {error}
              </p>
            )}
          </div>
        </div>
      </main>
    );
  }

  return (
    <main className="min-h-screen bg-zinc-950 text-white p-8">
      <div className="max-w-6xl mx-auto">
        <div className="bg-zinc-900 border border-zinc-800 rounded-2xl p-8">

          {/* HEADER */}
          <div className="flex justify-between items-center mb-8">
            <h1 className="text-4xl font-semibold">
              Admin panel
            </h1>

            <div className="flex gap-3">
              <Link
                href="/admin/feedback"
                className="bg-zinc-800 hover:bg-zinc-700 px-5 py-2 rounded-xl"
              >
                Feedback
              </Link>

              <button
                onClick={logout}
                className="bg-white text-black px-5 py-2 rounded-xl"
              >
                Odhlásit
              </button>
            </div>
          </div>

          {/* INSURERS */}
          <div className="mb-10">
            <h2 className="text-2xl font-semibold mb-4">
              Správa pojišťoven
            </h2>

            <div className="flex gap-3 mb-4">
              <select
                value={insurer}
                onChange={(e) =>
                  setInsurer(e.target.value)
                }
                className="rounded-xl bg-zinc-950 border border-zinc-800 px-4 py-3"
              >
                <option value="">
                  Vyber pojišťovnu
                </option>

                {insurers.map((item, index) => (
                  <option
                    key={`${item}-${index}`}
                    value={item}
                  >
                    {item}
                  </option>
                ))}
              </select>

              <button
                onClick={deleteInsurer}
                className="border border-zinc-700 px-5 py-3 rounded-xl"
              >
                Smazat
              </button>
            </div>

            <div className="flex gap-3">
              <input
                type="text"
                placeholder="Nová pojišťovna"
                value={newInsurer}
                onChange={(e) =>
                  setNewInsurer(e.target.value)
                }
                className="flex-1 rounded-xl bg-zinc-950 border border-zinc-800 px-4 py-3"
              />

              <button
                onClick={addInsurer}
                className="bg-white text-black px-5 py-3 rounded-xl"
              >
                Přidat
              </button>
            </div>
          </div>

          {/* UPLOAD */}
          <div className="mb-10">
            <h2 className="text-2xl font-semibold mb-4">
              Nahrát dokument
            </h2>

            <div className="space-y-4">
              <select
                value={insurer}
                onChange={(e) =>
                  setInsurer(e.target.value)
                }
                className="w-full rounded-xl bg-zinc-950 border border-zinc-800 px-4 py-3"
              >
                <option value="">
                  Vyber pojišťovnu
                </option>

                {insurers.map((item, index) => (
                  <option
                    key={`upload-${item}-${index}`}
                    value={item}
                  >
                    {item}
                  </option>
                ))}
              </select>

              <input
                type="text"
                placeholder="Název VPP"
                value={documentTitle}
                onChange={(e) =>
                  setDocumentTitle(
                    e.target.value
                  )
                }
                className="w-full rounded-xl bg-zinc-950 border border-zinc-800 px-4 py-3"
              />

              <input
                type="file"
                accept=".docx,.pdf"
                onChange={(e) =>
                  setFile(
                    e.target.files
                      ? e.target.files[0]
                      : null
                  )
                }
                className="w-full rounded-xl bg-zinc-950 border border-zinc-800 px-4 py-3"
              />

              <button
                onClick={uploadFile}
                className="bg-white text-black px-6 py-3 rounded-xl"
              >
                Nahrát soubor
              </button>
            </div>
          </div>

          {/* MESSAGE */}
          {message && (
            <div className="mb-10 rounded-xl bg-zinc-800 px-4 py-3">
              {message}
            </div>
          )}

          {/* FILES */}
          <div>
            <h2 className="text-2xl font-semibold mb-4">
              Nahrané dokumenty
            </h2>

            <div className="space-y-3">
              {files.map((item, index) => (
                <div
                  key={`${item.file_name}-${index}`}
                  className="rounded-xl border border-zinc-800 p-4"
                >
                  <div className="font-semibold">
                    {item.document_title}
                  </div>

                  <div className="text-sm text-zinc-400 mt-1">
                    Pojišťovna: {item.insurer}
                  </div>

                  <div className="text-sm text-zinc-400">
                    Soubor: {item.file_name}
                  </div>

                  <button
                    onClick={() =>
                      deleteFile(item.file_name)
                    }
                    className="mt-3 border border-zinc-700 px-4 py-2 rounded-xl"
                  >
                    Smazat
                  </button>
                </div>
              ))}
            </div>
          </div>

        </div>
      </div>
    </main>
  );
}