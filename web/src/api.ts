import type { Health, Play, PlaySummary } from "./types";

const BASE = "/api";

async function get<T>(path: string): Promise<T> {
  const response = await fetch(`${BASE}${path}`);
  if (!response.ok) {
    throw new Error(`${response.status} ${response.statusText} on ${path}`);
  }
  return (await response.json()) as T;
}

export const api = {
  health: () => get<Health>("/health"),

  teams: () => get<{ teams: { team: string; plays: number }[] }>("/teams"),

  plays: (params: { team?: string; coverage?: string; week?: number; limit?: number }) => {
    const query = new URLSearchParams();
    if (params.team) query.set("team", params.team);
    if (params.coverage) query.set("coverage", params.coverage);
    if (params.week !== undefined) query.set("week", String(params.week));
    query.set("limit", String(params.limit ?? 300));
    return get<{ total: number; plays: PlaySummary[] }>(`/plays?${query}`);
  },

  play: (playId: string) => get<Play>(`/plays/${encodeURIComponent(playId)}`),

  reports: () => get<{ reports: { slug: string; path: string }[] }>("/reports"),

  report: (slug: string) =>
    get<{ slug: string; markdown: string; evidence: unknown }>(`/reports/${slug}`),
};
