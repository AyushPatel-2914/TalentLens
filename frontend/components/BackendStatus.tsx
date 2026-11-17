"use client";

import { useEffect, useState } from "react";
import { getApiUrl } from "@/lib/api";

type State = "checking" | "ok" | "offline";

// The public demo backend runs on demand (on the author's machine, behind a Cloudflare tunnel).
// Instead of failing silently, tell the visitor what is happening and keep retrying.
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
        const res = await fetch(`${getApiUrl()}/health`, { signal: ctrl.signal });
        clearTimeout(t);
        if (res.ok) {
          if (!stopped) setState("ok");
          return;
        }
      } catch {
        /* not reachable yet */
      }
      if (stopped) return;
      const elapsed = Math.round((Date.now() - started) / 1000);
      setSeconds(elapsed);
      setState(elapsed >= 20 ? "offline" : "checking");
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
      {state === "checking" ? (
        "Connecting to the backend…"
      ) : (
        <>
          The live backend isn&apos;t reachable right now. It runs on demand, so ask for a fresh demo link, or see the{" "}
          <a href="https://github.com/AyushPatel-2914/TalentLens" style={{ color: "var(--accent)" }}>source and evaluation on GitHub</a>.
          Retrying automatically… {seconds}s
        </>
      )}
    </div>
  );
}
