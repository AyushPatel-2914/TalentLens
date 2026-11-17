import type { Explanation } from "@/lib/api";

type Props = { title: string; subtitle?: string; rank: number; e: Explanation; extra?: React.ReactNode };

export function ResultCard({ title, subtitle, rank, e, extra }: Props) {
  const pct = Math.round(e.score * 100);
  return (
    <div className="card stack">
      <div className="result-head">
        <div>
          <h3>
            {rank}. {title}
          </h3>
          {subtitle && <div className="muted small">{subtitle}</div>}
        </div>
        <div className="score" title={JSON.stringify(e.score_breakdown.weights_used)}>
          <div className="row small" style={{ justifyContent: "space-between" }}>
            <span className="muted">Match score</span>
            <strong>{pct}</strong>
          </div>
          <div className="score-bar" aria-label={`score ${pct} of 100`}>
            <div style={{ width: `${pct}%` }} />
          </div>
        </div>
      </div>

      {(e.required_skills.matched.length > 0 || e.required_skills.missing.length > 0) && (
        <div className="chips">
          {e.required_skills.matched.map((s) => (
            <span key={`m-${s}`} className="chip ok">✓ {s}</span>
          ))}
          {e.required_skills.missing.map((s) => (
            <span key={`x-${s}`} className="chip missing">✗ {s}</span>
          ))}
          {e.nice_to_have_matched.map((s) => (
            <span key={`n-${s}`} className="chip nice">+ {s}</span>
          ))}
        </div>
      )}

      <div className="facts">
        <div><span>Experience</span>{e.experience.summary}</div>
        <div><span>Seniority</span>{e.seniority.candidate ?? "unknown"} vs {e.seniority.job ?? "unknown"} ({e.seniority.fit})</div>
        <div><span>Location</span>{e.location.fit}</div>
        <div>
          <span>Score parts</span>
          {Object.entries(e.score_breakdown.components)
            .filter(([, v]) => v !== null)
            .map(([k, v]) => `${k} ${Math.round((v as number) * 100)}`)
            .join(" · ")}
        </div>
      </div>

      {e.evidence_quotes.length > 0 && (
        <div className="stack" style={{ gap: 6 }}>
          {e.evidence_quotes.map((q, i) => (
            <blockquote key={i}>
              “{q.quote}”<span className="supports">verified in resume · supports: {q.supports}</span>
            </blockquote>
          ))}
        </div>
      )}
      {e.quotes_dropped_unverified > 0 && (
        <div className="muted small">{e.quotes_dropped_unverified} LLM quote(s) dropped: not found word-for-word in the resume.</div>
      )}
      {e.quotes_note && <div className="muted small">{e.quotes_note}</div>}
      {extra}
    </div>
  );
}
