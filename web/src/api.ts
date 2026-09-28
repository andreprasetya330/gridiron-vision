import type { Health, Play, PlaySummary } from "./types";

const BASE = "/api";

async function get<T>(path: string, signal?: AbortSignal): Promise<T> {
  const response = await fetch(`${BASE}${path}`, { signal });
  if (!response.ok) {
    throw new Error(`${response.status} ${response.statusText} on ${path}`);
  }
  return (await response.json()) as T;
}

async function readError(response: Response): Promise<string> {
  try {
    const body = (await response.json()) as { detail?: unknown };
    if (typeof body.detail === "string") return body.detail;
    if (Array.isArray(body.detail)) {
      return body.detail.map((item) => (typeof item === "string" ? item : JSON.stringify(item))).join("; ");
    }
  } catch {
    /* fall through */
  }
  return `${response.status} ${response.statusText}`;
}

export interface IngestFields {
  file: File;
  defense_team: string;
  offense_team?: string;
  league?: string;
  down?: string;
  distance?: string;
  week?: string;
}

export const api = {
  health: () => get<Health>("/health"),

  teams: (source = "auto") => {
    const query = new URLSearchParams();
    if (source && source !== "auto") query.set("source", source);
    const suffix = query.toString() ? `?${query}` : "";
    return get<{ teams: { team: string; plays: number }[] }>(`/teams${suffix}`);
  },

  plays: (
    params: {
      team?: string;
      coverage?: string;
      week?: number;
      limit?: number;
      disguised_only?: boolean;
      source?: string;
    },
    signal?: AbortSignal,
  ) => {
    const query = new URLSearchParams();
    if (params.team) query.set("team", params.team);
    if (params.coverage) query.set("coverage", params.coverage);
    if (params.week !== undefined) query.set("week", String(params.week));
    if (params.disguised_only) query.set("disguised_only", "true");
    if (params.source) query.set("source", params.source);
    query.set("limit", String(params.limit ?? 300));
    return get<{ total: number; plays: PlaySummary[] }>(`/plays?${query}`, signal);
  },

  play: (playId: string, signal?: AbortSignal) =>
    get<Play>(`/plays/${encodeURIComponent(playId)}`, signal),

  ingest: async (fields: IngestFields): Promise<{ plays: Play[]; play_ids: string[] }> => {
    const body = new FormData();
    body.append("file", fields.file);
    body.append("defense_team", fields.defense_team.trim());
    if (fields.offense_team?.trim()) body.append("offense_team", fields.offense_team.trim());
    body.append("league", fields.league || "ncaa");
    if (fields.down) body.append("down", fields.down);
    if (fields.distance) body.append("distance", fields.distance);
    if (fields.week) body.append("week", fields.week);
    const response = await fetch(`${BASE}/film/ingest`, { method: "POST", body });
    if (!response.ok) {
      throw new Error(await readError(response));
    }
    return (await response.json()) as { plays: Play[]; play_ids: string[] };
  },

  reports: () => get<{ reports: { slug: string; path: string }[] }>("/reports"),

  report: (slug: string) =>
    get<{ slug: string; markdown: string; evidence: unknown }>(`/reports/${slug}`),
};
