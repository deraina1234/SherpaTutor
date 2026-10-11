import { useState } from "react";
import type { Proposal } from "./socket";
import type { Deliberation, DecisionMsg } from "./deliberation";
import ChatBox from "./ChatBox";

const AGENTS = [
  {
    name: "Learning",
    color: "#60a5fa",
    objective: "Protect productive struggle and long-term learning.",
    sees: "Learner level, hint history, time on task",
  },
  {
    name: "Momentum",
    color: "#f472b6",
    objective: "Minimize time stuck and frustration.",
    sees: "Edit and idle timing, repeated errors, failed runs, learner messages",
  },
  {
    name: "Integrity",
    color: "#a78bfa",
    objective: "Don't leak the solution; follow task rules.",
    sees: "Reference solution and task policy",
  },
];

const LEVEL_NAMES = ["No help", "Nudge", "Conceptual hint", "Point to the area", "Near-solution"];

const TAG_LABELS: Record<string, string> = {
  repeated_error: "Same error repeating",
  long_idle: "Idle for a while",
  concept_gap: "Missing a concept",
  solution_leak_risk: "Hint could give away the answer",
  active_progress: "Making progress",
  learner_requested_help: "Asked for help",
  recent_hint_given: "Just got a hint",
};

const tagLabel = (t: string) =>
  TAG_LABELS[t] ?? t.replace(/_/g, " ").replace(/^./, (c) => c.toUpperCase());

const OUTCOMES: Record<string, { label: string; color: string }> = {
  consensus: { label: "Consensus", color: "#4ade80" },
  soft_disagreement: { label: "Soft disagreement", color: "#fbbf24" },
  deadlock: { label: "Deadlock", color: "#f87171" },
};

const mono = { fontFamily: "ui-monospace, Menlo, monospace", fontSize: 12 };
const HATCH =
  "repeating-linear-gradient(45deg, #2a2a2a, #2a2a2a 4px, #353535 4px, #353535 8px)";

function Ladder({ level, cap, color }: { level: number; cap: number | null; color: string }) {
  return (
    <div style={{ display: "flex", gap: 3 }}>
      {[0, 1, 2, 3, 4].map((i) => {
        const blocked = cap !== null && i > cap;
        const active = i === level;
        const atCap = cap !== null && i === cap;
        return (
          <div
            key={i}
            title={blocked ? `Level ${i} is above the Integrity cap` : LEVEL_NAMES[i]}
            style={{
              flex: 1,
              height: 22,
              boxSizing: "border-box",
              display: "grid",
              placeItems: "center",
              borderRadius: 4,
              fontSize: 11,
              fontWeight: active ? 700 : 400,
              color: active ? "#111" : blocked ? "#666" : "#aaa",
              backgroundColor: active ? color : "#2f2f2f",
              backgroundImage: blocked && !active ? HATCH : "none",
              borderRight: atCap ? "2px solid #f87171" : "2px solid transparent",
              transition: "background-color 0.4s",
            }}
          >
            {i}
          </div>
        );
      })}
    </div>
  );
}

function ConfidenceBar({ value, color }: { value: number; color: string }) {
  const pct = Math.round(Math.max(0, Math.min(1, value)) * 100);
  return (
    <div>
      <div style={{ display: "flex", justifyContent: "space-between", fontSize: 11, color: "#999" }}>
        <span>Confidence</span>
        <span style={mono}>{pct}%</span>
      </div>
      <div style={{ height: 6, background: "#2f2f2f", borderRadius: 3, overflow: "hidden", marginTop: 3 }}>
        <div
          style={{
            height: "100%",
            width: `${pct}%`,
            background: color,
            transition: "width 0.4s",
          }}
        />
      </div>
    </div>
  );
}

function History({ entries, color }: { entries: { round: number; level: number }[]; color: string }) {
  return (
    <div style={{ display: "flex", alignItems: "center", gap: 4, flexWrap: "wrap" }}>
      {entries.map((e, i) => {
        const changed = i > 0 && e.level !== entries[i - 1].level;
        return (
          <span key={e.round} style={{ display: "inline-flex", alignItems: "center", gap: 4 }}>
            {i > 0 && <span style={{ color: "#666" }}>→</span>}
            <span
              title={`Round ${e.round}`}
              style={{
                ...mono,
                padding: "1px 6px",
                borderRadius: 4,
                background: changed ? color : "#2f2f2f",
                color: changed ? "#111" : "#ccc",
              }}
            >
              R{e.round} · {e.level}
            </span>
          </span>
        );
      })}
    </div>
  );
}

function statusOf(entries: { level: number }[]): { text: string; color: string } {
  if (entries.length < 2) return { text: "proposed", color: "#999" };
  const diff = entries[entries.length - 1].level - entries[entries.length - 2].level;
  if (diff === 0) return { text: "held", color: "#999" };
  return diff > 0
    ? { text: `moved up ${diff}`, color: "#4ade80" }
    : { text: `moved down ${-diff}`, color: "#fbbf24" };
}

function AgentCard({
  meta,
  entries,
  latest,
  cap,
}: {
  meta: (typeof AGENTS)[number];
  entries: { round: number; level: number }[];
  latest: Proposal | undefined;
  cap: number | null;
}) {
  const status = statusOf(entries);
  const isIntegrity = meta.name === "Integrity";
  const clamped = !isIntegrity && latest !== undefined && cap !== null && latest.level > cap;
  return (
    <div style={{ background: "#262626", borderRadius: 8, padding: 10, display: "grid", gap: 8 }}>
      <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
        <span style={{ width: 10, height: 10, borderRadius: 5, background: meta.color }} />
        <strong style={{ fontSize: 14 }}>{meta.name}</strong>
        {latest && (
          <span style={{ marginLeft: "auto", fontSize: 11, color: status.color }}>{status.text}</span>
        )}
      </div>
      <div style={{ fontSize: 12, color: "#999", lineHeight: 1.4 }}>{meta.objective}</div>

      {latest ? (
        <>
          <div style={{ display: "flex", alignItems: "baseline", gap: 8, flexWrap: "wrap" }}>
            {isIntegrity ? (
              <>
                <span style={{ fontSize: 20, fontWeight: 700, color: meta.color, lineHeight: 1 }}>
                  Sets cap: L{cap ?? latest.level}
                </span>
                <span style={{ fontSize: 12, color: "#ccc" }}>
                  {LEVEL_NAMES[cap ?? latest.level] ?? ""} at most
                </span>
              </>
            ) : (
              <>
                <span style={{ fontSize: 26, fontWeight: 700, color: meta.color, lineHeight: 1 }}>
                  L{latest.level}
                </span>
                <span style={{ fontSize: 12, color: "#ccc" }}>{LEVEL_NAMES[latest.level] ?? ""}</span>
                {clamped && (
                  <span
                    title={`Integrity allows at most L${cap}, so this vote counts as L${cap}`}
                    style={{
                      fontSize: 11,
                      fontWeight: 600,
                      padding: "1px 8px",
                      borderRadius: 10,
                      background: "#3b1d1d",
                      color: "#f87171",
                      border: "1px solid #7f2d2d",
                    }}
                  >
                    clamped to L{cap}
                  </span>
                )}
              </>
            )}
          </div>
          <ConfidenceBar value={latest.confidence} color={meta.color} />
          <Ladder level={isIntegrity ? -1 : latest.level} cap={cap} color={meta.color} />
          {latest.tags.length > 0 && (
            <div style={{ display: "flex", gap: 4, flexWrap: "wrap" }}>
              {latest.tags.map((t) => (
                <span
                  key={t}
                  title={t}
                  style={{
                    fontSize: 11,
                    padding: "1px 8px",
                    borderRadius: 10,
                    background: "#333",
                    color: "#ccc",
                  }}
                >
                  {tagLabel(t)}
                </span>
              ))}
            </div>
          )}
          {!isIntegrity && <History entries={entries} color={meta.color} />}
        </>
      ) : (
        <div style={{ fontSize: 12, color: "#666" }}>Watching. No position yet.</div>
      )}

      <details style={{ fontSize: 12, color: "#999" }}>
        <summary style={{ cursor: "pointer" }}>Private information</summary>
        <div style={{ marginTop: 4 }}>{meta.sees}</div>
      </details>
    </div>
  );
}

export default function AgentPanel({
  d,
  onAskMore,
  onChat,
  traceCase,
  tracing,
  traceError,
  onTrace,
}: {
  d: Deliberation;
  onAskMore: () => void;
  onChat: (text: string) => void;
  traceCase: number | null; // failing case the learner can trace, or null when not offered
  tracing: boolean;
  traceError: string | null;
  onTrace: () => void;
}) {
  // Remember which decision the learner already asked about, so the button only fires once per decision
  const [askedFor, setAskedFor] = useState<DecisionMsg | null>(null);

  const roundNums = Object.keys(d.rounds)
    .map(Number)
    .sort((a, b) => a - b);
  const lastRound = roundNums.length ? roundNums[roundNums.length - 1] : null;
  const deliberating = lastRound !== null && d.decision === null;

  function proposalIn(round: number, agent: string): Proposal | undefined {
    return d.rounds[round]?.find((p) => p.agent.toLowerCase() === agent.toLowerCase());
  }

  // Integrity's hard cap: an explicit `cap` if the backend sends one, otherwise its own level
  const integrity = lastRound !== null ? proposalIn(lastRound, "Integrity") : undefined;
  const cap = integrity ? (integrity.cap ?? integrity.level) : null;

  const outcome = d.decision ? (OUTCOMES[d.decision.outcome] ?? { label: d.decision.outcome, color: "#999" }) : null;
  const canAskMore =
    d.decision !== null &&
    (d.decision.outcome === "soft_disagreement" || d.decision.outcome === "deadlock");
  const asked = d.decision !== null && askedFor === d.decision;

  return (
    <div
      style={{
        width: 340,
        flexShrink: 0,
        borderLeft: "1px solid #333",
        padding: 12,
        overflow: "auto",
        display: "flex",
        flexDirection: "column",
        gap: 10,
      }}
    >
      <div style={{ display: "flex", alignItems: "baseline", justifyContent: "space-between" }}>
        <strong style={{ fontSize: 15 }}>Agents</strong>
        <span style={{ fontSize: 12, color: deliberating ? "#fbbf24" : "#999" }}>
          {deliberating
            ? `Deliberating · round ${lastRound}`
            : d.decision
              ? `Decided after ${roundNums.length} round${roundNums.length === 1 ? "" : "s"}`
              : "Watching"}
        </span>
      </div>

      {outcome && d.decision && (
        <div
          style={{
            borderRadius: 8,
            padding: 10,
            background: "#262626",
            borderLeft: `4px solid ${outcome.color}`,
            display: "grid",
            gap: 6,
          }}
        >
          <div style={{ display: "flex", justifyContent: "space-between", alignItems: "baseline" }}>
            <strong style={{ color: outcome.color }}>{outcome.label}</strong>
            <span style={{ fontSize: 12, color: "#ccc" }}>
              {d.decision.level === 0 ? "No automatic hint" : `Acting at L${d.decision.level}`}
            </span>
          </div>
          <div style={{ fontSize: 13, color: "#ccc", lineHeight: 1.45 }}>{d.decision.explanation}</div>
          {canAskMore && (
            <button
              disabled={asked}
              onClick={() => {
                setAskedFor(d.decision);
                onAskMore();
              }}
              style={{ justifySelf: "start" }}
            >
              {asked
                ? "Requested"
                : d.decision.outcome === "deadlock"
                  ? "Want a nudge?"
                  : "Ask for more help"}
            </button>
          )}
        </div>
      )}

      {d.hint && (
        <div style={{ borderRadius: 8, padding: 10, background: "#1f2d24", border: "1px solid #2f4a39" }}>
          <div style={{ fontSize: 11, color: "#4ade80", marginBottom: 4 }}>
            HINT · LEVEL {d.hint.level}
          </div>
          <div style={{ fontSize: 14, lineHeight: 1.45 }}>{d.hint.text}</div>
          {traceCase !== null && (
            <button onClick={onTrace} disabled={tracing} style={{ marginTop: 8 }}>
              {tracing ? "Tracing…" : `See what your code does on Case ${traceCase + 1}`}
            </button>
          )}
          {traceError && <div style={{ fontSize: 12, color: "#f87171", marginTop: 6 }}>{traceError}</div>}
        </div>
      )}

      <div style={{ fontSize: 11, color: "#777" }}>
        Levels 0–4. Hatched cells are above the Integrity cap; the red line marks the cap.
      </div>

      {AGENTS.map((meta) => {
        const entries = roundNums
          .map((r) => ({ round: r, p: proposalIn(r, meta.name) }))
          .filter((e): e is { round: number; p: Proposal } => e.p !== undefined)
          .map((e) => ({ round: e.round, level: e.p.level }));
        return (
          <AgentCard
            key={meta.name}
            meta={meta}
            entries={entries}
            latest={lastRound !== null ? proposalIn(lastRound, meta.name) : undefined}
            cap={cap}
          />
        );
      })}

      <ChatBox onSend={onChat} />
    </div>
  );
}