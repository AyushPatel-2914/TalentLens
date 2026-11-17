"use client";

import { useState } from "react";
import { AgentPanel } from "@/components/AgentPanel";
import { MatchPanel } from "@/components/MatchPanel";

const TABS = [
  { id: "job", label: "Job → candidates" },
  { id: "resume", label: "Resume → jobs" },
  { id: "agent", label: "Recruiter agent" },
] as const;

export default function Home() {
  const [tab, setTab] = useState<(typeof TABS)[number]["id"]>("job");
  return (
    <main className="container">
      <header className="top">
        <h1>TalentLens</h1>
        <p>Explainable job matching: hybrid retrieval, cross-encoder re-ranking, verified evidence.</p>
      </header>
      <nav className="tabs" role="tablist">
        {TABS.map((t) => (
          <button key={t.id} role="tab" aria-selected={tab === t.id} className={tab === t.id ? "active" : ""}
                  onClick={() => setTab(t.id)}>
            {t.label}
          </button>
        ))}
      </nav>
      {/* Panels stay mounted so results survive switching tabs. */}
      <div hidden={tab !== "job"}><MatchPanel mode="job" /></div>
      <div hidden={tab !== "resume"}><MatchPanel mode="resume" /></div>
      <div hidden={tab !== "agent"}><AgentPanel /></div>
    </main>
  );
}
