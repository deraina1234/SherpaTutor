import type { InMsg, Proposal } from "./socket";

export type DecisionMsg = Extract<InMsg, { type: "decision" }>;
export type HintMsg = Extract<InMsg, { type: "hint" }>;

export type Deliberation = {
  trigger: string | null; // why the agents met, e.g. "failed_run" or "heartbeat"
  rounds: Record<number, Proposal[]>;
  decision: DecisionMsg | null;
  hint: HintMsg | null;
};

export const emptyDeliberation: Deliberation = { trigger: null, rounds: {}, decision: null, hint: null };

// Round 0 starts a new deliberation; everything else adds to the current one.
// A background check-in (demo mode) keeps the hint already on screen; a new hint replaces it.
export function applyMessage(d: Deliberation, m: InMsg): Deliberation {
  switch (m.type) {
    case "round":
      return m.round === 0
        ? { trigger: m.trigger ?? null, rounds: { 0: m.proposals }, decision: null, hint: m.trigger === "heartbeat" ? d.hint : null }
        : { ...d, rounds: { ...d.rounds, [m.round]: m.proposals } };
    case "decision":
      return { ...d, decision: m };
    case "hint":
      return { ...d, hint: m };
  }
}