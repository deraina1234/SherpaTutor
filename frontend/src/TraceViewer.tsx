import { useEffect, useState } from "react";
import type { Problem } from "./problems";
import CallTreeView from "./CallTreeView";

// Shapes produced by TRACE_HARNESS in public/pyodide-worker.js.
export type Snap = { kind: "list"; items: string[] } | { kind: "value"; repr: string };

export type TraceStep = {
  event: "line" | "return";
  line: number;
  fn: string;
  depth: number;
  locals: Record<string, Snap>;
  returnValue: string | null;
  call: number | null; // id of the TraceCall this step belongs to
};

// One function call, for the call tree. Arguments are captured when the call starts.
export type TraceCall = {
  id: number;
  parent: number | null;
  fn: string;
  args: string[];
  argNames: string[];
  firstStep: number;
  returnStep: number | null;
  returnValue: string | null;
  updatedShared: boolean; // changed a closure variable or dict/set argument (e.g. saved an answer)
  label?: string | null; // value of the problem's traceLabel in this call, if one worked
  goal?: boolean | null; // value of the problem's traceGoal in this call
};

export type TraceResult = {
  steps: TraceStep[];
  calls?: TraceCall[]; // absent from an older cached worker
  returned: string | null;
  matches: boolean | null; // returned value == expected, compared in Python
  error: { message: string; line: number | null } | null;
  truncated: boolean;
};

// Integer variables with these names get a pointer label under the list box they index.
const POINTER_NAMES = new Set(["i", "j", "k", "lo", "hi", "left", "right", "l", "r", "mid", "index", "idx", "start", "pos", "end"]);

const mono = { fontFamily: "ui-monospace, Menlo, monospace", fontSize: 13 };

function sameSnap(a: Snap | undefined, b: Snap | undefined): boolean {
  return JSON.stringify(a) === JSON.stringify(b);
}

function pointersFor(locals: Record<string, Snap>, length: number): Record<number, string[]> {
  const out: Record<number, string[]> = {};
  for (const [name, snap] of Object.entries(locals)) {
    if (!POINTER_NAMES.has(name) || snap.kind !== "value" || !/^\d+$/.test(snap.repr)) continue;
    const idx = Number(snap.repr);
    if (idx < length) (out[idx] ??= []).push(name);
  }
  return out;
}

function ListBoxes({ items, pointers, changed }: { items: string[]; pointers: Record<number, string[]>; changed: boolean }) {
  if (items.length === 0) return <span style={{ ...mono, color: changed ? "#fbbf24" : "#ddd" }}>[]</span>;
  return (
    <div style={{ display: "flex", gap: 3, flexWrap: "wrap" }}>
      {items.map((item, i) => (
        <div key={i} style={{ display: "grid", justifyItems: "center", gap: 2 }}>
          <div
            style={{
              ...mono,
              minWidth: 28,
              padding: "3px 6px",
              textAlign: "center",
              borderRadius: 4,
              background: pointers[i] ? "#1e3a5f" : "#2f2f2f",
              border: `1px solid ${changed ? "#fbbf24" : "#444"}`,
              color: "#eee",
            }}
          >
            {item}
          </div>
          <div style={{ fontSize: 10, color: "#777" }}>{i}</div>
          {pointers[i] && <div style={{ fontSize: 10, color: "#60a5fa", fontWeight: 600 }}>{pointers[i].join(",")}</div>}
        </div>
      ))}
    </div>
  );
}

export default function TraceViewer({
  problem,
  caseIndex,
  code,
  result,
  onClose,
}: {
  problem: Problem;
  caseIndex: number;
  code: string;
  result: TraceResult;
  onClose: () => void;
}) {
  const { steps } = result;
  const calls = result.calls ?? [];
  const recursive = calls.length > 1;
  const last = Math.max(steps.length - 1, 0);
  const [step, setStep] = useState(0);
  const [mode, setMode] = useState<"calls" | "lines">(recursive ? "calls" : "lines");

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") onClose();
      else if (mode !== "lines") return; // the call tree handles its own keys
      else if (e.key === "ArrowRight") setStep((s) => Math.min(s + 1, last));
      else if (e.key === "ArrowLeft") setStep((s) => Math.max(s - 1, 0));
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [last, onClose, mode]);

  const testCase = problem.cases[caseIndex];
  const current = steps[step];
  const previous = step > 0 ? steps[step - 1] : undefined;
  const sameFrame = previous !== undefined && current !== undefined && previous.fn === current.fn && previous.depth === current.depth;
  const atEnd = step === last;
  const errorHere = atEnd && result.error !== null;
  const lines = code.replace(/\n$/, "").split("\n");
  const maxDepth = steps.reduce((m, s) => Math.max(m, s.depth), 0);

  return (
    <div
      onClick={onClose}
      style={{ position: "fixed", inset: 0, background: "rgba(0,0,0,0.6)", display: "grid", placeItems: "center", zIndex: 50 }}
    >
      <div
        onClick={(e) => e.stopPropagation()}
        style={{
          width: recursive ? 1100 : 900,
          maxWidth: "94vw",
          height: mode === "calls" ? "88vh" : "80vh",
          background: "#1e1e1e",
          border: "1px solid #333",
          borderRadius: 10,
          display: "grid",
          gridTemplateRows: mode === "calls" ? "auto 1fr" : "auto 1fr auto auto auto",
          overflow: "hidden",
        }}
      >
        {/* Header */}
        <div style={{ padding: "12px 16px", borderBottom: "1px solid #333", display: "flex", alignItems: "baseline", gap: 12 }}>
          <strong style={{ fontSize: 15 }}>Your code on Case {caseIndex + 1}</strong>
          <span style={{ ...mono, color: "#ccc" }}>
            {problem.params.map((p, i) => `${p} = ${testCase.args[i]}`).join(", ")}
          </span>
          <span style={{ ...mono, color: "#999" }}>expected {testCase.expected}</span>
          {recursive && (
            <div style={{ display: "flex", gap: 6, marginLeft: "auto" }}>
              {(["calls", "lines"] as const).map((m) => (
                <button
                  key={m}
                  onClick={() => setMode(m)}
                  style={{
                    background: mode === m ? "#3a3a3a" : "#2a2a2a",
                    color: "#ddd",
                    border: "none",
                    borderRadius: 6,
                    padding: "4px 10px",
                    cursor: "pointer",
                  }}
                >
                  {m === "calls" ? `Call tree (${calls.length} calls)` : `Line by line (${steps.length} steps)`}
                </button>
              ))}
            </div>
          )}
          <button onClick={onClose} title="Close (Esc)" style={{ marginLeft: recursive ? 0 : "auto" }}>
            ✕
          </button>
        </div>

        {mode === "calls" && <CallTreeView result={result} code={code} expected={testCase.expected} problem={problem} />}

        {mode === "lines" && (
        <>
        {/* Code and variables */}
        <div style={{ display: "grid", gridTemplateColumns: recursive ? "1fr 1.1fr" : "1.2fr 1fr", minHeight: 0 }}>
          <div style={{ overflow: "auto", padding: "10px 0", borderRight: "1px solid #333" }}>
            {current && current.depth > 0 && (
              <div style={{ fontSize: 11, color: "#a78bfa", padding: "0 16px 6px" }}>
                in {current.fn} (call depth {current.depth})
              </div>
            )}
            {lines.map((text, i) => {
              const lineNo = i + 1;
              const active = current?.line === lineNo;
              const color = active && errorHere ? "#f87171" : active ? "#fbbf24" : "transparent";
              return (
                <div
                  key={i}
                  style={{
                    ...mono,
                    display: "flex",
                    whiteSpace: "pre",
                    background: active ? (errorHere ? "#3b1d1d" : "#3a3320") : "transparent",
                    borderLeft: `3px solid ${color}`,
                  }}
                >
                  <span style={{ width: 36, textAlign: "right", paddingRight: 12, color: "#666", flexShrink: 0 }}>{lineNo}</span>
                  <span>{text}</span>
                </div>
              );
            })}
          </div>

          <div style={{ overflow: "auto", padding: "10px 16px", display: "grid", alignContent: "start", gap: 10 }}>
            <div style={{ fontSize: 12, color: "#999" }}>
              {current
                ? current.event === "return"
                  ? `${current.fn} is returning ${current.returnValue}`
                  : `About to run line ${current.line}`
                : "No steps recorded"}
            </div>
            {current &&
              Object.entries(current.locals).map(([name, snap]) => {
                const changed = sameFrame && !sameSnap(previous?.locals[name], snap);
                return (
                  <div key={name} style={{ display: "grid", gap: 4 }}>
                    <span style={{ ...mono, color: changed ? "#fbbf24" : "#9cdcfe" }}>{name}</span>
                    {snap.kind === "list" ? (
                      <ListBoxes items={snap.items} pointers={pointersFor(current.locals, snap.items.length)} changed={changed} />
                    ) : (
                      <span style={{ ...mono, color: changed ? "#fbbf24" : "#ddd" }}>{snap.repr}</span>
                    )}
                  </div>
                );
              })}
          </div>
        </div>

        {/* Outcome, shown on the last step */}
        <div style={{ padding: atEnd ? "8px 16px" : 0, borderTop: atEnd ? "1px solid #333" : "none" }}>
          {atEnd && result.truncated && (
            <div style={{ ...mono, color: "#fbbf24" }}>
              Showing the first {steps.length} steps
              {maxDepth > 0 ? ` (calls nested up to ${maxDepth} deep)` : ""}. If this run never finishes, look for
              an infinite loop or far more work than this input needs.
            </div>
          )}
          {atEnd && result.error && <div style={{ ...mono, color: "#f87171" }}>{result.error.message}</div>}
          {atEnd && result.returned !== null && (
            <div style={{ ...mono, color: result.matches ? "#4ade80" : "#f87171" }}>
              returned {result.returned} · expected {testCase.expected}
            </div>
          )}
        </div>

        {/* Controls */}
        <div style={{ padding: "10px 16px", borderTop: "1px solid #333", display: "flex", alignItems: "center", gap: 10 }}>
          <button onClick={() => setStep((s) => Math.max(s - 1, 0))} disabled={step === 0}>
            ◀ Prev
          </button>
          <input
            type="range"
            min={0}
            max={last}
            value={step}
            onChange={(e) => setStep(Number(e.target.value))}
            style={{ flex: 1 }}
            disabled={steps.length < 2}
          />
          <button onClick={() => setStep((s) => Math.min(s + 1, last))} disabled={atEnd}>
            Next ▶
          </button>
          <span style={{ ...mono, fontSize: 12, color: "#999", minWidth: 110, textAlign: "right" }}>
            Step {steps.length ? step + 1 : 0} of {steps.length}
          </span>
        </div>
        <div style={{ fontSize: 11, color: "#777", padding: "0 16px 10px" }}>
          This is your code, not the solution. Use ← → to step, Esc to close.
        </div>
        </>
        )}
      </div>
    </div>
  );
}
