import type { InMsg, Proposal } from "./socket";

export type DecisionMsg = Extract<InMsg, { type: "decision" }>;
export type HintMsg = Extract<InMsg, { type: "hint" }>;

export type Deliberation = {
  rounds: Record<number, Proposal[]>;
  decision: DecisionMsg | null;
  hint: HintMsg | null;
};

export const emptyDeliberation: Deliberation = { rounds: {}, decision: null, hint: null };

// Round 0 starts a new deliberation; everything else adds to the current one.
export function applyMessage(d: Deliberation, m: InMsg): Deliberation {
  switch (m.type) {
    case "round":
      return m.round === 0
        ? { rounds: { 0: m.proposals }, decision: null, hint: null }
        : { ...d, rounds: { ...d.rounds, [m.round]: m.proposals } };
    case "decision":
      return { ...d, decision: m };
    case "hint":
      return { ...d, hint: m };
  }
}