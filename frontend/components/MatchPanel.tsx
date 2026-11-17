"use client";

import { useState } from "react";
import { api, type MatchJobResponse, type MatchResumeResponse } from "@/lib/api";
import samples from "@/lib/samples.json";
import { ResultCard } from "./ResultCard";

export function MatchPanel({ mode }: { mode: "job" | "resume" }) {
  const options = mode === "job" ? samples.jobs : samples.resumes;
  const [text, setText] = useState(options[0].text);
  const [quotes, setQuotes] = useState(true);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [jobRes, setJobRes] = useState<MatchJobResponse | null>(null);
  const [resumeRes, setResumeRes] = useState<MatchResumeResponse | null>(null);

  async function run() {
    setLoading(true);
    setError(null);
    try {
      if (mode === "job") setJobRes(await api.matchJob(text, 10, quotes));
      else setResumeRes(await api.matchResume(text, 10));
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setLoading(false);
    }
  }

  const timings = mode === "job" ? jobRes?.timings_ms : resumeRes?.timings_ms;

  return (
    <div className="stack">
      <div className="card stack">
        <label className="muted small" htmlFor={`input-${mode}`}>
          {mode === "job" ? "Paste a job description" : "Paste a resume"}
        </label>
        <div className="row small">
          <span className="muted">Examples:</span>
          {options.map((o) => (
            <button key={o.label} className="secondary small" onClick={() => setText(o.text)} disabled={loading}>
              {o.label}
            </button>
          ))}
        </div>
        <textarea id={`input-${mode}`} value={text} onChange={(e) => setText(e.target.value)} />
        <div className="row">
          <button className="primary" onClick={run} disabled={loading || text.trim().length < 30}>
            {loading ? "Matching…" : mode === "job" ? "Find candidates" : "Find jobs"}
          </button>
          {mode === "job" && (
            <label className="row small muted" style={{ gap: 6 }}>
              <input type="checkbox" checked={quotes} onChange={(e) => setQuotes(e.target.checked)} />
              Evidence quotes for the top 3 (LLM, verified)
            </label>
          )}
          {timings && (
            <span className="muted small">
              {timings.served_from_cache ? "served from cache" : `${Math.round(Number(timings.total))} ms`}
            </span>
          )}
        </div>
      </div>

      {error && <div className="error">{error}</div>}

      {mode === "job" && jobRes && (
        <>
          <div className="muted small">
            Requirements read from the job ({jobRes.job._source}):{" "}
            {(jobRes.job.required_skills ?? []).join(", ") || "none found"}
            {jobRes.job.min_years_experience ? ` · ${jobRes.job.min_years_experience}+ years` : ""}
          </div>
          {jobRes.results.map((r, i) => (
            <ResultCard
              key={r.resume_id}
              rank={i + 1}
              title={`${r.resume_id} · ${r.candidate.job_titles[0] ?? r.candidate.category}`}
              subtitle={[
                r.candidate.category,
                r.candidate.total_years_experience != null ? `${r.candidate.total_years_experience} yrs` : null,
                r.candidate.seniority && r.candidate.seniority !== "unknown" ? r.candidate.seniority : null,
                r.candidate.education_level && r.candidate.education_level !== "unknown" ? r.candidate.education_level.replace("_", " ") : null,
              ].filter(Boolean).join(" · ")}
              e={r}
            />
          ))}
        </>
      )}

      {mode === "resume" &&
        resumeRes?.results.map((r, i) => (
          <ResultCard
            key={r.job_id}
            rank={i + 1}
            title={r.title}
            subtitle={[r.company, r.location, r.category].filter(Boolean).join(" · ")}
            e={r}
          />
        ))}
    </div>
  );
}
