import { useEffect, useMemo, useRef, useState } from "react";
import type { Problem } from "./problems";
import type { TraceCall, TraceResult } from "./TraceViewer";

// Steps through a recursive function one call at a time, drawing the calls as a tree.
// Works for any recursion; a problem can add a node label and a goal (see problems.ts).
// Calls that repeat an earlier subproblem are marked, since that repeated work is what
// memoization removes.

const PLAY_MS = 700;
const LEVEL_GAP = 64;
const NODE_H = 30;
const MAX_LABEL = 12;

const mono = { fontFamily: "ui-monospace, Menlo, monospace", fontSize: 13 };

const COLORS = {
  plain: { fill: "#1e1e1e", stroke: "#666", text: "#ddd" },
  saved: { fill: "#16341f", stroke: "#4ade80", text: "#86efac" },
  deadEnd: { fill: "#3b1d1d", stroke: "#f87171", text: "#fca5a5" },
  focus: { fill: "#3a2f12", stroke: "#f59e0b", text: "#fcd34d" },
};
const REPEAT = "#f59e0b";

function compact(repr: string): string {
  return repr.startsWith("[") || repr.startsWith("(") ? repr.replace(/,\s+/g, ",") : repr;
}

function labelOf(c: TraceCall): string {
  if (c.parent === null) return c.fn;
  const text = c.label ?? c.args.map(compact).join(" ");
  return text.length > MAX_LABEL ? text.slice(0, MAX_LABEL - 1) + "…" : text;
}

const nodeWidth = (label: string) => Math.max(NODE_H, label.length * 7.4 + 16);

// Dashed edge: the child kept the same list arguments as its parent and only moved on
// (e.g. index + 1). Solid edge: a list argument changed, so a choice was added or removed.
function movedOn(parent: TraceCall, child: TraceCall): boolean {
  if (parent.fn !== child.fn) return false;
  const lists = child.args.map((a, i) => (a.startsWith("[") ? i : -1)).filter((i) => i >= 0);
  return lists.length > 0 && lists.every((i) => child.args[i] === parent.args[i]);
}

// Which subproblem a call works on. With a problem label (e.g. the running sum), list
// arguments are replaced by it, so [1, 2] and [2, 1] at the same index count as the same.
function subproblemKey(c: TraceCall): string {
  if (c.parent === null) return `root:${c.id}`;
  const rest = c.label != null ? c.args.filter((a) => !a.startsWith("[")) : c.args;
  return [c.fn, c.label ?? "", ...rest].join("|");
}

function findRepeats(calls: TraceCall[]) {
  const keys = calls.map(subproblemKey);
  const first = new Map<string, number>();
  const count = new Map<string, number>();
  const repeatOf: (number | null)[] = [];
  const redundant: boolean[] = [];
  for (const c of calls) {
    const key = keys[c.id];
    repeatOf[c.id] = first.get(key) ?? null;
    if (!first.has(key)) first.set(key, c.id);
    count.set(key, (count.get(key) ?? 0) + 1);
    // Calls arrive parent-first, so a call inside a repeated subtree is redundant work too.
    redundant[c.id] = repeatOf[c.id] !== null || (c.parent !== null && redundant[c.parent]);
  }
  const top = [...count.entries()]
    .filter(([, n]) => n > 1)
    .sort((a, b) => b[1] - a[1])
    .slice(0, 3)
    .map(([key, n]) => ({ key, n, sample: first.get(key)!, last: keys.lastIndexOf(key) }));
  return { keys, repeatOf, redundantCount: redundant.filter(Boolean).length, top };
}

// Full description of a subproblem for the "most repeated" list: the label plus the
// arguments it doesn't cover (e.g. "11, index=0"), so two different subproblems with
// the same label don't read the same.
function subproblemText(c: TraceCall): string {
  if (c.label == null) return labelOf(c);
  const rest = c.argNames
    .map((name, i) => (c.args[i].startsWith("[") ? null : `${name}=${c.args[i]}`))
    .filter((x) => x !== null);
  return [c.label, ...rest].join(", ");
}

type Placed = { x: number; y: number; w: number; label: string };

function layout(calls: TraceCall[]) {
  const kids = new Map<number | null, number[]>();
  for (const c of calls) kids.set(c.parent, [...(kids.get(c.parent) ?? []), c.id]);
  const labels = calls.map(labelOf);
  const gap = Math.max(...labels.map(nodeWidth)) + 12;
  const placed = new Map<number, Placed>();
  let nextLeaf = 0;
  let maxDepth = 0;
  const place = (id: number, depth: number) => {
    maxDepth = Math.max(maxDepth, depth);
    const children = kids.get(id) ?? [];
    children.forEach((ch) => place(ch, depth + 1));
    const x = children.length
      ? (placed.get(children[0])!.x + placed.get(children[children.length - 1])!.x) / 2
      : 20 + gap / 2 + gap * nextLeaf++;
    placed.set(id, { x, y: 24 + NODE_H / 2 + depth * LEVEL_GAP, w: nodeWidth(labels[id]), label: labels[id] });
  };
  (kids.get(null) ?? []).forEach((root) => place(root, 0));
  return {
    kids,
    placed,
    width: 40 + gap * Math.max(nextLeaf, 1),
    height: 24 + NODE_H + maxDepth * LEVEL_GAP + 34,
  };
}

export default function CallTreeView({
  result,
  code,
  expected,
  problem,
}: {
  result: TraceResult;
  code: string;
  expected: string;
  problem: Problem;
}) {
  const calls = useMemo(() => result.calls ?? [], [result]);
  const { steps } = result;
  const n = calls.length;
  const { kids, placed, width, height } = useMemo(() => layout(calls), [calls]);
  const repeats = useMemo(() => findRepeats(calls), [calls]);
  const [focusKey, setFocusKey] = useState<string | null>(null);
  const hasLabel = calls.some((c) => c.label != null);
  const hasGoal = calls.some((c) => c.goal != null);
  const [k, setK] = useState(0);
  const [playing, setPlaying] = useState(false);
  const isPlaying = playing && k < n - 1;

  useEffect(() => {
    if (!isPlaying) return;
    const t = window.setInterval(() => setK((prev) => Math.min(prev + 1, n - 1)), PLAY_MS);
    return () => window.clearInterval(t);
  }, [isPlaying, n]);

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "ArrowRight") setK((prev) => Math.min(prev + 1, n - 1));
      else if (e.key === "ArrowLeft") setK((prev) => Math.max(prev - 1, 0));
      else if (e.key === " ") {
        e.preventDefault();
        setPlaying((p) => !p);
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [n]);

  const currentRef = useRef<SVGGElement | null>(null);
  useEffect(() => {
    currentRef.current?.scrollIntoView({ block: "nearest", inline: "center" });
  }, [k]);

  const current = calls[k];
  // Everything that finishes before the next call starts is shown as finished.
  const nextTime = calls[k + 1]?.firstStep ?? Infinity;
  const finished = (c: TraceCall) => c.returnStep !== null && c.returnStep < nextTime;
  const isDeadEnd = (c: TraceCall) =>
    finished(c) && !(kids.get(c.id)?.length) && c.returnValue === "None" && !c.updatedShared;
  // Green: the problem's goal holds (e.g. sum == amount); without a goal, the call saved
  // something. Either can happen before the call finishes, so neither waits for it.
  const reached = (c: TraceCall) => (hasGoal ? c.goal === true : c.updatedShared);
  const colorOf = (c: TraceCall) =>
    focusKey !== null && repeats.keys[c.id] === focusKey
      ? COLORS.focus
      : reached(c)
        ? COLORS.saved
        : isDeadEnd(c)
          ? COLORS.deadEnd
          : COLORS.plain;

  const stack = new Set<number>();
  for (let id: number | null = current?.id ?? null; id !== null; id = calls[id].parent) stack.add(id);

  const lines = code.replace(/\n$/, "").split("\n");
  const lineText = (line: number | undefined) => (line ? lines[line - 1]?.trim() : undefined);
  const calledFrom = current && current.firstStep > 0 ? steps[current.firstStep - 1]?.line : undefined;
  // A call that is still running at this step hasn't returned yet: show what it calls next instead.
  const done = current !== undefined && finished(current);
  const returnLine = done && current.returnStep !== null ? steps[current.returnStep]?.line : undefined;
  const next = calls[k + 1];
  const nextCallLine = next && next.parent === current?.id && next.firstStep > 0 ? steps[next.firstStep - 1]?.line : undefined;
  const highlight = returnLine ?? nextCallLine ?? (current ? steps[current.firstStep]?.line : undefined);
  const highlightColor = current && reached(current) ? "#4ade80" : current && isDeadEnd(current) ? "#f87171" : "#fbbf24";
  const repeatsCall = current ? repeats.repeatOf[current.id] : null;

  if (!current) return <div style={{ color: "#999", padding: 16 }}>No calls recorded.</div>;

  return (
    <div style={{ display: "grid", gridTemplateRows: "auto auto auto 1fr auto", minHeight: 0, height: "100%" }}>
      {/* Controls */}
      <div style={{ display: "flex", alignItems: "center", gap: 8, padding: "10px 16px" }}>
        <button onClick={() => setK((p) => Math.max(p - 1, 0))} disabled={k === 0}>
          ◀ Prev
        </button>
        <button onClick={() => setK((p) => Math.min(p + 1, n - 1))} disabled={k === n - 1}>
          Next ▶
        </button>
        <button
          onClick={() => {
            if (k === n - 1) setK(0);
            setPlaying(!isPlaying);
          }}
        >
          {isPlaying ? "❚❚ Pause" : "▶ Play"}
        </button>
        <button
          onClick={() => {
            setPlaying(false);
            setK(0);
          }}
        >
          Reset
        </button>
        <span style={{ ...mono, color: "#999", marginLeft: 8 }}>
          Call {k + 1} of {n}
        </span>
      </div>

      {/* What this call did */}
      <div style={{ padding: "0 16px 10px", fontSize: 13, lineHeight: 1.6, color: "#ccc" }}>
        <div style={mono}>
          {current.fn}({current.argNames.map((name, i) => `${name}=${current.args[i]}`).join(", ")})
          {current.label != null && current.parent !== null && (
            <span style={{ color: "#999" }}>
              {" "}
              · {problem.traceLabelText ?? "label"} = {current.label}
            </span>
          )}
        </div>
        {repeatsCall !== null && (
          <div style={{ color: REPEAT }}>
            Same subproblem as{" "}
            <button onClick={() => setK(repeatsCall)} style={{ padding: "0 6px" }}>
              call {repeatsCall + 1}
            </button>{" "}
            — everything below this call repeats work that was already done.
          </div>
        )}
        {calledFrom !== undefined && current.parent !== null && (
          <div>
            Called from line {calledFrom}: <code style={{ color: "#ddd" }}>{lineText(calledFrom)}</code>
          </div>
        )}
        {!done && current.returnStep !== null && nextCallLine !== undefined ? (
          <div>
            Next, line {nextCallLine} calls{" "}
            <code style={{ color: "#ddd" }}>
              {next.fn}({next.args.join(", ")})
            </code>
            : <code style={{ color: "#ddd" }}>{lineText(nextCallLine)}</code>
          </div>
        ) : returnLine !== undefined ? (
          <div>
            Returned <code style={{ color: "#ddd" }}>{current.returnValue}</code> at line {returnLine}:{" "}
            <code style={{ color: "#ddd" }}>{lineText(returnLine)}</code>
            {current.updatedShared && <span style={{ color: "#4ade80" }}> · saved something (updated shared state)</span>}
            {isDeadEnd(current) && <span style={{ color: "#f87171" }}> · dead end</span>}
          </div>
        ) : (
          <div style={{ color: "#fbbf24" }}>Never returned: the trace stopped while this call was still running.</div>
        )}
        {k === n - 1 && result.returned !== null && (
          <div style={{ color: result.matches ? "#4ade80" : "#f87171" }}>
            Last call. The function returns {result.returned} (expected {expected}).
          </div>
        )}
        {k === n - 1 && result.truncated && (
          <div style={{ color: "#fbbf24" }}>Trace stopped early: these are the first {n} calls.</div>
        )}
      </div>

      {/* Repeated work across the whole trace */}
      <div
        style={{
          display: "flex",
          flexWrap: "wrap",
          alignItems: "center",
          gap: 8,
          padding: "8px 16px",
          borderTop: "1px solid #333",
          fontSize: 13,
          color: "#ccc",
        }}
      >
        {repeats.redundantCount > 0 ? (
          <>
            <span>
              <strong style={{ color: REPEAT }}>
                {repeats.redundantCount} of {n} calls
              </strong>{" "}
              redo a subproblem that was already worked out{result.truncated ? " (in the calls traced)" : ""}.
            </span>
            <span style={{ color: "#888" }}>Most repeated:</span>
            {repeats.top.map((t) => (
              <button
                key={t.key}
                onClick={() => {
                  const on = focusKey !== t.key;
                  setFocusKey(on ? t.key : null);
                  // Later repeats aren't drawn yet: step far enough to show every one of them.
                  if (on) {
                    setPlaying(false);
                    setK((prev) => Math.max(prev, t.last));
                  }
                }}
                title="Highlight every call working on this subproblem (steps forward to show them all)"
                style={{
                  ...mono,
                  fontSize: 12,
                  background: focusKey === t.key ? "#3a2f12" : "#2a2a2a",
                  color: focusKey === t.key ? "#fcd34d" : "#ddd",
                  border: `1px solid ${focusKey === t.key ? REPEAT : "#444"}`,
                  borderRadius: 10,
                  padding: "1px 8px",
                  cursor: "pointer",
                }}
              >
                {subproblemText(calls[t.sample])} ×{t.n}
              </button>
            ))}
          </>
        ) : (
          <span>Every call works on a new subproblem. No repeated work.</span>
        )}
      </div>

      {/* Code and tree */}
      <div style={{ display: "grid", gridTemplateColumns: "minmax(300px, 1fr) 1.5fr", minHeight: 0, borderTop: "1px solid #333" }}>
        <div style={{ overflow: "auto", padding: "10px 0", borderRight: "1px solid #333" }}>
          {lines.map((text, i) => {
            const active = highlight === i + 1;
            return (
              <div
                key={i}
                style={{
                  ...mono,
                  fontSize: 12,
                  display: "flex",
                  whiteSpace: "pre",
                  background: active ? "#2c2c2c" : "transparent",
                  borderLeft: `3px solid ${active ? highlightColor : "transparent"}`,
                }}
              >
                <span style={{ width: 30, textAlign: "right", paddingRight: 10, color: "#666", flexShrink: 0 }}>{i + 1}</span>
                <span>{text}</span>
              </div>
            );
          })}
        </div>

        <div style={{ overflow: "auto", minHeight: 0 }}>
          <svg width={width} height={height} style={{ display: "block" }}>
            {calls.slice(1, k + 1).map((c) => {
              if (c.parent === null) return null;
              const from = placed.get(c.parent)!;
              const to = placed.get(c.id)!;
              return (
                <line
                  key={`e${c.id}`}
                  x1={from.x}
                  y1={from.y + NODE_H / 2}
                  x2={to.x}
                  y2={to.y - NODE_H / 2}
                  stroke={stack.has(c.id) ? "#aaa" : "#555"}
                  strokeWidth={1.5}
                  strokeDasharray={movedOn(calls[c.parent], c) ? "5 4" : undefined}
                />
              );
            })}
            {calls.slice(0, k + 1).map((c) => {
              const p = placed.get(c.id)!;
              const color = colorOf(c);
              const isCurrent = c.id === k;
              const showReturn = finished(c) && c.returnValue !== null && c.returnValue !== "None";
              return (
                <g
                  key={c.id}
                  ref={isCurrent ? currentRef : undefined}
                  onClick={() => setK(c.id)}
                  style={{ cursor: "pointer" }}
                >
                  <title>
                    {`${c.fn}(${c.argNames.map((name, i) => `${name}=${c.args[i]}`).join(", ")})` +
                      (repeats.repeatOf[c.id] !== null ? `\nRepeats call ${repeats.repeatOf[c.id]! + 1}` : "")}
                  </title>
                  {repeats.repeatOf[c.id] !== null && (
                    <rect
                      x={p.x - p.w / 2 - 3}
                      y={p.y - NODE_H / 2 - 3}
                      width={p.w + 6}
                      height={NODE_H + 6}
                      rx={(NODE_H + 6) / 2}
                      fill="none"
                      stroke={REPEAT}
                      strokeWidth={1.5}
                      strokeDasharray="3 3"
                    />
                  )}
                  {isCurrent && (
                    <rect
                      x={p.x - p.w / 2 - 5}
                      y={p.y - NODE_H / 2 - 5}
                      width={p.w + 10}
                      height={NODE_H + 10}
                      rx={(NODE_H + 10) / 2}
                      fill="none"
                      stroke="#60a5fa"
                      strokeWidth={3}
                    />
                  )}
                  <rect
                    x={p.x - p.w / 2}
                    y={p.y - NODE_H / 2}
                    width={p.w}
                    height={NODE_H}
                    rx={NODE_H / 2}
                    fill={color.fill}
                    stroke={stack.has(c.id) && !isCurrent ? "#bbb" : color.stroke}
                    strokeWidth={1.5}
                  />
                  <text x={p.x} y={p.y + 4.5} textAnchor="middle" fill={color.text} style={{ ...mono, fontSize: 12 }}>
                    {p.label}
                  </text>
                  {showReturn && (
                    <text x={p.x} y={p.y + NODE_H / 2 + 14} textAnchor="middle" fill="#4ade80" style={{ ...mono, fontSize: 11 }}>
                      → {c.returnValue}
                    </text>
                  )}
                </g>
              );
            })}
          </svg>
        </div>
      </div>

      {/* Legend */}
      <div style={{ display: "flex", flexWrap: "wrap", gap: 16, padding: "8px 16px", borderTop: "1px solid #333", fontSize: 12, color: "#aaa" }}>
        <span>── a list argument changed (choice made)</span>
        <span>- - - same choices, moved on</span>
        {hasLabel && problem.traceLabelText && <span>number in node = {problem.traceLabelText}</span>}
        <span style={{ color: "#4ade80" }}>
          green = {hasGoal ? problem.traceGoalText ?? "goal reached" : "saved something (e.g. an answer)"}
        </span>
        <span style={{ color: REPEAT }}>amber dashes = repeats an earlier subproblem</span>
        <span style={{ color: "#f87171" }}>red = dead end</span>
        <span style={{ color: "#60a5fa" }}>blue ring = current call</span>
        <span>← → step · space play</span>
      </div>
    </div>
  );
}
