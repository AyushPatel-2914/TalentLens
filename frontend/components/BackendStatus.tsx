"use client";

import { useEffect, useState } from "react";
import { API_URL } from "@/lib/api";

type State = "checking" | "ok" | "waking";

// Free hosting (Hugging Face Spaces) puts the backend to sleep when idle. Instead of failing
// silently, tell the visitor what is happening and retry until it answers.
export function BackendStatus() {
  const [state, setState] = useState<State>("checking");
  const [seconds, setSeconds] = useState(0);

  useEffect(() => {
    let stopped = false;
    const started = Date.now();
    async function check() {
      try {
        const ctrl = new AbortController();
        const t = setTimeout(() => ctrl.abort(), 8000);
        const res = await fetch(`${API_URL}/health`, { signal: ctrl.signal });
        clearTimeout(t);
        if (res.ok) {
          if (!stopped) setState("ok");
          return;
        }
      } catch {
        /* not reachable yet */
      }
      if (stopped) return;
      setState("waking");
      setSeconds(Math.round((Date.now() - started) / 1000));
      setTimeout(check, 10000);
    }
    check();
    return () => {
      stopped = true;
    };
  }, []);

  if (state === "ok") return null;
  return (
    <div className="card small" role="status" style={{ marginBottom: 16, background: "var(--warn-soft)" }}>
      {state === "checking"
        ? "Connecting to the backend…"
        : `The backend is waking up (free hosting sleeps when idle). This takes 1–3 minutes; retrying automatically… ${seconds}s`}
    </div>
  );
}
