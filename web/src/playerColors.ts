/** One color per defender, the way broadcast tracking boxes are painted. */

const DEFENSE_PALETTE = [
  "#e11d48",
  "#f97316",
  "#eab308",
  "#22c55e",
  "#14b8a6",
  "#3b82f6",
  "#8b5cf6",
  "#ec4899",
  "#06b6d4",
  "#84cc16",
  "#fb7185",
];

export const OFFENSE_COLOR = "#64748b";
export const RUSHER_COLOR = "#f97316";

export function playerColor(
  trackId: string,
  side: "offense" | "defense",
  role?: string | null,
): string {
  if (side === "offense") return OFFENSE_COLOR;
  if (role === "blitz") return RUSHER_COLOR;
  let hash = 0;
  for (let i = 0; i < trackId.length; i += 1) {
    hash = (hash * 33 + trackId.charCodeAt(i)) >>> 0;
  }
  return DEFENSE_PALETTE[hash % DEFENSE_PALETTE.length];
}

export function playerLabel(
  jersey: number | null,
  side: "offense" | "defense",
  role?: string | null,
): string {
  if (jersey !== null) return String(jersey);
  if (side !== "defense") return "";
  if (role === "blitz") return "RUSH";
  if (role === "deep_zone") return "S";
  if (role === "man") return "CB";
  if (role === "underneath_zone") return "LB";
  return "D";
}
