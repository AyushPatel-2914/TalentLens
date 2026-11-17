// Thin typed client for the TalentLens FastAPI backend.

const DEFAULT_API_URL = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";
const STORAGE_KEY = "talentlens_api_url";

// Only these backends may be chosen from a link, so a crafted ?api= link can't point the page elsewhere.
const ALLOWED_API = /^(https:\/\/[a-z0-9-]+\.trycloudflare\.com|http:\/\/(localhost|127\.0\.0\.1):\d+)$/;

// The public demo backend runs on the author's laptop behind a Cloudflare quick tunnel, whose URL
// changes on every start. The share link carries it: https://<frontend>/?api=https://xyz.trycloudflare.com
// It is remembered in this browser, so later visits without ?api= still work.
export function getApiUrl(): string {
  if (typeof window === "undefined") return DEFAULT_API_URL;
  const fromLink = new URLSearchParams(window.location.search).get("api")?.trim().replace(/\/+$/, "");
  try {
    if (fromLink && ALLOWED_API.test(fromLink)) {
      localStorage.setItem(STORAGE_KEY, fromLink);
      return fromLink;
    }
    const saved = localStorage.getItem(STORAGE_KEY);
    if (saved && ALLOWED_API.test(saved)) return saved;
  } catch {
    /* storage blocked: fall through */
  }
  return fromLink && ALLOWED_API.test(fromLink) ? fromLink : DEFAULT_API_URL;
}

export type SkillMatch = { skill: string; matched: boolean; via: string | null };

export type Explanation = {
  score: number;
  score_breakdown: { components: Record<string, number | null>; weights_used: Record<string, number> };
  required_skills: { matched: string[]; missing: string[]; coverage: number | null };
  nice_to_have_matched: string[];
  experience: { fit: number; summary: string };
  seniority: { job: string | null; candidate: string | null; fit: string };
  location: { job: string | null; candidate: string | null; fit: string };
  evidence_quotes: { quote: string; supports: string }[];
  quotes_dropped_unverified: number;
  quotes_note?: string;
};

export type CandidateResult = Explanation & {
  resume_id: string;
  reranker_logit: number | null;
  candidate: {
    category: string;
    job_titles: string[];
    total_years_experience: number | null;
    education_level: string | null;
    seniority: string | null;
    structured_by: string;
    summary: string;
  };
};

export type JobResult = Explanation & {
  job_id: string;
  title: string;
  company: string | null;
  location: string | null;
  category: string;
};

export type MatchJobResponse = {
  job: { title?: string; required_skills?: string[]; min_years_experience?: number | null; _source?: string };
  results: CandidateResult[];
  timings_ms: Record<string, number | boolean>;
  cached: boolean;
};

export type MatchResumeResponse = { results: JobResult[]; timings_ms: Record<string, number | boolean>; cached: boolean };

export type Draft = {
  id: string;
  candidate_id: string;
  job_ref: string;
  subject: string;
  body: string;
  status: "PENDING_APPROVAL" | "APPROVED" | "REJECTED";
};

export type AgentStep = {
  step: number;
  kind: string;
  tool: string | null;
  arguments: string | null;
  result: string | null;
  error: string | null;
  latency_ms: number;
};

export type AgentResponse = {
  request: string;
  final_answer: string;
  stopped_reason: string;
  steps: AgentStep[];
  drafts: Draft[];
  llm_calls: number;
  input_tokens: number;
  output_tokens: number;
  cost_usd: number;
};

async function post<T>(path: string, body: unknown): Promise<T> {
  const res = await fetch(`${getApiUrl()}${path}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body ?? {}),
  });
  const data = await res.json().catch(() => ({}));
  if (!res.ok) {
    const detail = Array.isArray(data?.detail) ? data.detail.map((d: { msg: string }) => d.msg).join("; ") : data?.detail;
    throw new Error(`${data?.error ?? `HTTP ${res.status}`}${detail ? `: ${detail}` : ""}`);
  }
  return data as T;
}

export const api = {
  matchJob: (job_text: string, top_k = 10, with_quotes = true) =>
    post<MatchJobResponse>("/match/job", { job_text, top_k, with_quotes }),
  matchResume: (resume_text: string, top_k = 10) => post<MatchResumeResponse>("/match/resume", { resume_text, top_k }),
  agent: (request: string) => post<AgentResponse>("/agent", { request }),
  approve: (id: string, edited_body?: string) => post<Draft>(`/outreach/${id}/approve`, { edited_body }),
  reject: (id: string) => post<Draft>(`/outreach/${id}/reject`, {}),
};
