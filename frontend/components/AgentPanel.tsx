"use client";

import { useState } from "react";
import { api, type AgentResponse, type Draft } from "@/lib/api";
import samples from "@/lib/samples.json";

const EXAMPLES = samples.agent;

function DraftCard({ draft, onChange }: { draft: Draft; onChange: (d: Draft) => void }) {
  const [body, setBody] = useState(draft.body);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function decide(approve: boolean) {
    setBusy(true);
    setError(null);
    try {
      onChange(approve ? await api.approve(draft.id, body !== draft.body ? body : undefined) : await api.reject(draft.id));
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setBusy(false);
    }
  }

  const pending = draft.status === "PENDING_APPROVAL";
  return (
    <div className="draft stack">
      <div className="row" style={{ justifyContent: "space-between" }}>
        <strong>Outreach to {draft.candidate_id}</strong>
        <span className={`badge ${draft.status}`}>{draft.status.replace("_", " ")}</span>
      </div>
      <div className="small"><span className="muted">Subject:</span> {draft.subject}</div>
      <textarea value={body} onChange={(e) => setBody(e.target.value)} disabled={!pending} style={{ minHeight: 120 }} />
      {pending && (
        <div className="row">
          <button className="primary" onClick={() => decide(true)} disabled={busy}>Approve</button>
          <button className="danger" onClick={() => decide(false)} disabled={busy}>Reject</button>
          <span className="muted small">Nothing is sent until you approve.</span>
        </div>
      )}
      {error && <div className="error">{error}</div>}
    </div>
  );
}

export function AgentPanel() {
  const [request, setRequest] = useState(EXAMPLES[0]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<AgentResponse | null>(null);

  async function run() {
    setLoading(true);
    setError(null);
    setResult(null);
    try {
      setResult(await api.agent(request));
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setLoading(false);
    }
  }

  function updateDraft(d: Draft) {
    setResult((r) => (r ? { ...r, drafts: r.drafts.map((x) => (x.id === d.id ? d : x)) } : r));
  }

  return (
    <div className="stack">
      <div className="card stack">
        <label className="muted small" htmlFor="agent-input">Ask the recruiter agent</label>
        <input id="agent-input" type="text" value={request} onChange={(e) => setRequest(e.target.value)}
               onKeyDown={(e) => e.key === "Enter" && !loading && run()} />
        <div className="row">
          <button className="primary" onClick={run} disabled={loading || request.trim().length < 5}>
            {loading ? "Working…" : "Run agent"}
          </button>
          {EXAMPLES.map((ex) => (
            <button key={ex} className="secondary small" onClick={() => setRequest(ex)} disabled={loading}>
              {ex.length > 42 ? ex.slice(0, 42) + "…" : ex}
            </button>
          ))}
        </div>
      </div>

      {error && <div className="error">{error}</div>}

      {result && (
        <>
          <div className="card stack">
            <div className="row muted small">
              <span>{result.llm_calls} LLM calls</span>
              <span>{result.steps.filter((s) => s.kind === "tool_call").length} tool calls</span>
              <span>{result.input_tokens + result.output_tokens} tokens</span>
              <span>stopped: {result.stopped_reason}</span>
            </div>
            <div className="answer">{result.final_answer}</div>
          </div>
          {result.drafts.map((d) => (
            <DraftCard key={d.id} draft={d} onChange={updateDraft} />
          ))}
          <div className="card steps">
            <strong style={{ fontFamily: "inherit" }}>Tool log</strong>
            {result.steps.filter((s) => s.kind === "tool_call").map((s, i) => (
              <details key={i}>
                <summary>
                  step {s.step} · {s.tool}({s.arguments}) {s.error ? "· ERROR" : ""} · {Math.round(s.latency_ms)} ms
                </summary>
                <pre>{s.error ?? s.result}</pre>
              </details>
            ))}
          </div>
        </>
      )}
    </div>
  );
}
