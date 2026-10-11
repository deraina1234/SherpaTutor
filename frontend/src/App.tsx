import { useCallback, useEffect, useRef, useState } from "react";
import Editor from "@monaco-editor/react";
import { PROBLEMS, type Problem } from "./problems";
import { Socket, type InMsg } from "./socket";
import AgentPanel from "./AgentPanel";
import { applyMessage, emptyDeliberation, type Deliberation, type HintMsg } from "./deliberation";
import { useRevealQueue } from "./useRevealQueue";
import TraceViewer, { type TraceResult } from "./TraceViewer";

// Hints at this level or above offer a step-by-step trace of the learner's own code.
const TRACE_MIN_LEVEL = 2;

// One case to trace, with the exact code that was run.
type TraceTarget = { problemId: string; code: string; caseIndex: number };

// What the last run left traceable: its failing cases, or every case if it timed out.
type LastRun = { problemId: string; code: string; failing: number[]; timedOut: boolean };

type CaseResult = { ok: boolean; actual: string | null; error: string | null };
type RunState = {
  timedOut: boolean;
  compileError: string | null;
  cases: CaseResult[];
  stdout: string;
};

function verdictOf(r: RunState): string {
  if (r.timedOut) return "Time Limit Exceeded";
  if (r.compileError || r.cases.some((c) => c.error)) return "Runtime Error";
  if (r.cases.every((c) => c.ok)) return "Accepted";
  return "Wrong Answer";
}

function firstErrorOf(r: RunState, p: Problem): string | null {
  if (r.timedOut) return "Timed out after 5s (infinite loop?)";
  if (r.compileError) return r.compileError;
  for (let i = 0; i < r.cases.length; i++) {
    const c = r.cases[i];
    if (c.error) return `Case ${i + 1}: ${c.error}`;
    if (!c.ok) return `Case ${i + 1}: expected ${p.cases[i].expected}, got ${c.actual}`;
  }
  return null;
}

const diffColor = { Easy: "#4ade80", Medium: "#fbbf24" } as const;

const mono = { fontFamily: "ui-monospace, Menlo, monospace", fontSize: 13 };
const box = {
  ...mono,
  background: "#2a2a2a",
  borderRadius: 6,
  padding: "8px 10px",
  whiteSpace: "pre-wrap" as const,
};

export default function App() {
  const [problemIndex, setProblemIndex] = useState(0);
  const [ready, setReady] = useState(false);
  const [running, setRunning] = useState(false);
  const [connected, setConnected] = useState(false);
  const [run, setRun] = useState<RunState | null>(null);
  const [selected, setSelected] = useState(0);
  const [delib, setDelib] = useState<Deliberation>(emptyDeliberation);
  const [lastRun, setLastRun] = useState<LastRun | null>(null);
  const [hintAtLastRun, setHintAtLastRun] = useState<HintMsg | null>(null);
  const [tracing, setTracing] = useState(false);
  const [traceError, setTraceError] = useState<string | null>(null);
  const [openTrace, setOpenTrace] = useState<{ target: TraceTarget; result: TraceResult } | null>(null);

  const problem = PROBLEMS[problemIndex];
  const applyMsg = useCallback((m: InMsg) => setDelib((prev) => applyMessage(prev, m)), []);
  const { push: pushMsg, clear: clearQueue } = useRevealQueue(applyMsg);

  const workerRef = useRef<Worker | null>(null);
  const timerRef = useRef<number | undefined>(undefined);
  const traceTimerRef = useRef<number | undefined>(undefined);
  const runCodeRef = useRef("");
  const pendingTraceRef = useRef<TraceTarget | null>(null);
  const socketRef = useRef<Socket | null>(null);

  // Per-problem code and last-sent snapshot, so switching problems keeps your work
  const codesRef = useRef<Record<string, string>>({});
  const snapshotsRef = useRef<Record<string, string>>({});

  // Callbacks and timers read the current problem from here. It is updated in an
  // effect (not during render) so React's rules about refs are respected.
  const problemRef = useRef<Problem>(problem);
  useEffect(() => {
    problemRef.current = problem;
  }, [problem]);

  function currentCode(p: Problem) {
    return codesRef.current[p.id] ?? p.starter;
  }

  const sendRunEvent = useCallback((r: RunState) => {
    const p = problemRef.current;
    // Send the code that ran with the result: edits otherwise reach the backend up to 5s later.
    const snapshot = runCodeRef.current;
    snapshotsRef.current[p.id] = snapshot;
    const passed = verdictOf(r) === "Accepted";
    // A hint about a bug that is now fixed would read as a claim that the passing code is wrong.
    if (passed) setDelib((prev) => ({ ...prev, hint: null }));
    socketRef.current?.send({
      type: "run",
      ts: Date.now(),
      problem_id: p.id,
      passed,
      first_error: firstErrorOf(r, p),
      snapshot,
    });
  }, []);

  const createWorker = useCallback(() => {
    workerRef.current?.terminate();
    const w = new Worker("/pyodide-worker.js");
    w.onmessage = (e) => {
      if (e.data.ready) {
        setReady(true);
        return;
      }
      if (e.data.kind === "trace") {
        window.clearTimeout(traceTimerRef.current);
        setTracing(false);
        const target = pendingTraceRef.current;
        if (e.data.ok && target) setOpenTrace({ target, result: e.data.trace });
        else setTraceError(e.data.ok ? null : "Couldn't trace your code.");
        return;
      }
      window.clearTimeout(timerRef.current);
      setRunning(false);
      const r: RunState = e.data.ok
        ? {
            timedOut: false,
            compileError: e.data.result.compile_error,
            cases: e.data.result.cases,
            stdout: e.data.result.stdout,
          }
        : { timedOut: false, compileError: e.data.error, cases: [], stdout: "" };
      setRun(r);
      sendRunEvent(r);
      // Cases that ran and gave a wrong answer or raised can be traced; a compile error has nothing to step through.
      const failing = r.compileError ? [] : r.cases.flatMap((c, i) => (c.ok ? [] : [i]));
      setLastRun({ problemId: problemRef.current.id, code: runCodeRef.current, failing, timedOut: false });
      // Show the first failing case, so its details and trace button are what the learner sees.
      if (failing.length > 0) setSelected(failing[0]);
    };
    workerRef.current = w;
  }, [sendRunEvent]);

  // Start Python once on load. (`ready` already starts as false.)
  useEffect(() => {
    createWorker();
    return () => workerRef.current?.terminate();
  }, [createWorker]);

  useEffect(() => {
    const s = new Socket((m) => {
      console.log("recv", m);
      pushMsg(m);
    }, setConnected);
    socketRef.current = s;
    s.send({ type: "problem", ts: Date.now(), problem_id: problemRef.current.id });

    const interval = window.setInterval(() => {
      const p = problemRef.current;
      const now = currentCode(p);
      const before = snapshotsRef.current[p.id] ?? p.starter;
      if (now === before) return;
      const a = before.split("\n");
      const b = now.split("\n");
      s.send({
        type: "edit",
        ts: Date.now(),
        problem_id: p.id,
        snapshot: now,
        diff_summary: {
          added: b.filter((l) => !a.includes(l)).length,
          deleted: a.filter((l) => !b.includes(l)).length,
        },
      });
      snapshotsRef.current[p.id] = now;
    }, 5000);

    return () => {
      window.clearInterval(interval);
      s.close();
    };
  }, [pushMsg]);

  function switchProblem(i: number) {
    clearQueue();
    setLastRun(null);
    setTraceError(null);
    setProblemIndex(i);
    setRun(null);
    setSelected(0);
    setDelib(emptyDeliberation);
    socketRef.current?.send({ type: "problem", ts: Date.now(), problem_id: PROBLEMS[i].id });
  }

  function runTests() {
    if (!ready || running || tracing) return;
    setRunning(true);
    // A new run hides the trace offer until the next qualifying hint.
    runCodeRef.current = currentCode(problem);
    setLastRun(null);
    setTraceError(null);
    setHintAtLastRun(delib.hint);
    workerRef.current?.postMessage({
      id: Date.now(),
      code: runCodeRef.current,
      fnName: problem.fnName,
      cases: problem.cases,
    });
    // Kill runaway code (e.g. infinite loops) after 5 seconds, then restart Python
    timerRef.current = window.setTimeout(() => {
      setReady(false);
      createWorker();
      setRunning(false);
      const r: RunState = { timedOut: true, compileError: null, cases: [], stdout: "" };
      setRun(r);
      sendRunEvent(r);
      // Which case hung is unknown, so any case can be traced; the step limit shows the runaway work.
      setLastRun({ problemId: problemRef.current.id, code: runCodeRef.current, failing: [], timedOut: true });
    }, 5000);
  }

  // Any case can be traced at any time: it runs the learner's own current code, so it can't leak a solution.
  function startTrace(caseIndex: number) {
    if (!ready || running || tracing) return;
    const traceTarget: TraceTarget = { problemId: problem.id, code: currentCode(problem), caseIndex };
    const c = problem.cases[caseIndex];
    setTracing(true);
    setTraceError(null);
    pendingTraceRef.current = traceTarget;
    workerRef.current?.postMessage({
      type: "trace",
      id: Date.now(),
      code: traceTarget.code,
      fnName: problem.fnName,
      args: c.args,
      expected: c.expected,
      spec: { label: problem.traceLabel ?? [], goal: problem.traceGoal ?? null },
    });
    // Same safety net as runs: kill and restart Python if tracing hangs
    traceTimerRef.current = window.setTimeout(() => {
      setReady(false);
      createWorker();
      setTracing(false);
      setTraceError("Tracing timed out after 5 seconds.");
    }, 5000);
  }

  const closeTrace = useCallback(() => setOpenTrace(null), []);

  // The agents also suggest a trace in the hint card when they decide on a level 2+ hint.
  const suggestedCase = lastRun?.failing[0] ?? (lastRun?.timedOut ? 0 : null);
  const traceCase =
    suggestedCase !== null &&
    lastRun?.problemId === problem.id &&
    delib.hint !== null &&
    delib.hint !== hintAtLastRun &&
    delib.hint.level >= TRACE_MIN_LEVEL
      ? suggestedCase
      : null;

  const verdict = run ? verdictOf(run) : null;
  const passedCount = run ? run.cases.filter((c) => c.ok).length : 0;
  const selCase = problem.cases[selected];
  const selResult = run?.cases[selected];
  const example = problem.cases[0];

  return (
    <div style={{ display: "flex", flexDirection: "column", height: "100vh" }}>
      {/* Toolbar */}
      <div
        style={{
          padding: 8,
          borderBottom: "1px solid #333",
          display: "flex",
          alignItems: "center",
          gap: 12,
        }}
      >
        <select
          value={problemIndex}
          disabled={running}
          onChange={(e) => switchProblem(Number(e.target.value))}
          style={{ maxWidth: 360 }}
        >
          {PROBLEMS.map((p, i) => (
            <option key={p.id} value={i}>
              {i === 0 ? "Warm-up" : i}. {p.title} ({p.difficulty})
            </option>
          ))}
        </select>
        <button onClick={runTests} disabled={!ready || running || tracing}>
          {!ready ? "Loading Python..." : running ? "Running..." : "Run"}
        </button>
        <span
          style={{ marginLeft: "auto", fontSize: 12, color: connected ? "#4ade80" : "#f87171" }}
        >
          {connected ? "● agents connected" : "○ agents offline"}
        </span>
      </div>

      {/* Main area: workspace on the left, agent sidebar on the right */}
      <div style={{ flex: 1, minHeight: 0, display: "flex" }}>
        <div style={{ flex: 1, minWidth: 0, display: "flex", flexDirection: "column" }}>
          {/* Description + editor */}
          <div style={{ flex: 1, minHeight: 0, display: "flex" }}>
            <div
              style={{
                width: 320,
                padding: 14,
                overflow: "auto",
                borderRight: "1px solid #333",
                fontSize: 14,
                lineHeight: 1.5,
              }}
            >
              <div style={{ fontSize: 17, fontWeight: 600 }}>{problem.title}</div>
              <div
                style={{ color: diffColor[problem.difficulty], fontSize: 12, margin: "2px 0 10px" }}
              >
                {problem.difficulty}
              </div>
              <div style={{ marginBottom: 12 }}>{problem.description}</div>
              <div style={{ fontSize: 12, color: "#999", marginBottom: 4 }}>Example</div>
              <div style={box}>
                {problem.params.map((p, i) => `${p} = ${example.args[i]}`).join("\n")}
                {"\n"}
                {"→ "}
                {example.expected}
              </div>
              <div style={{ fontSize: 12, color: "#999", marginTop: 12 }}>
                Function:{" "}
                <span style={mono}>
                  {problem.fnName}({problem.params.join(", ")})
                </span>
              </div>
            </div>

            <div style={{ flex: 1, minWidth: 0 }}>
              <Editor
                path={problem.id}
                defaultLanguage="python"
                defaultValue={problem.starter}
                theme="vs-dark"
                onChange={(value) => {
                  codesRef.current[problem.id] = value ?? "";
                }}
              />
            </div>
          </div>

          {/* Test panel */}
          <div
            style={{
              height: 260,
              borderTop: "1px solid #333",
              padding: 12,
              overflow: "auto",
              background: "#1e1e1e",
            }}
          >
            <div style={{ marginBottom: 10, fontWeight: 600 }}>
              {verdict === null ? (
                <span style={{ color: "#999" }}>Press Run to test your code.</span>
              ) : (
                <>
                  <span
                    style={{ color: verdict === "Accepted" ? "#4ade80" : "#f87171", fontSize: 16 }}
                  >
                    {verdict}
                  </span>
                  {run && run.cases.length > 0 && (
                    <span style={{ color: "#999", fontWeight: 400, marginLeft: 10 }}>
                      {passedCount} / {problem.cases.length} test cases passed
                    </span>
                  )}
                </>
              )}
            </div>

            {run?.timedOut && (
              <div style={{ ...box, color: "#f87171", marginBottom: 10 }}>
                Your code ran longer than 5 seconds. Check for an infinite loop.
              </div>
            )}
            {run?.compileError && (
              <div style={{ ...box, color: "#f87171", marginBottom: 10 }}>{run.compileError}</div>
            )}

            <div style={{ display: "flex", gap: 6, marginBottom: 10, flexWrap: "wrap" }}>
              {problem.cases.map((_, i) => {
                const r = run?.cases[i];
                const dot = r ? (r.ok ? "#4ade80" : "#f87171") : "#555";
                return (
                  <button
                    key={i}
                    onClick={() => setSelected(i)}
                    style={{
                      background: selected === i ? "#3a3a3a" : "#2a2a2a",
                      color: "#ddd",
                      border: "none",
                      borderRadius: 6,
                      padding: "4px 10px",
                      cursor: "pointer",
                    }}
                  >
                    <span style={{ color: dot }}>●</span> Case {i + 1}
                  </button>
                );
              })}
              <button
                onClick={() => startTrace(selected)}
                disabled={!ready || running || tracing}
                title="Watch your code run line by line on the selected case"
                style={{ marginLeft: "auto" }}
              >
                {tracing ? "Tracing…" : `▶ Step through Case ${selected + 1}`}
              </button>
            </div>
            {traceError && <div style={{ fontSize: 12, color: "#f87171", marginBottom: 8 }}>{traceError}</div>}

            <div style={{ display: "grid", gap: 8 }}>
              <div>
                <div style={{ fontSize: 12, color: "#999", marginBottom: 4 }}>Input</div>
                <div style={box}>
                  {problem.params.map((p, i) => `${p} = ${selCase.args[i]}`).join("\n")}
                </div>
              </div>
              <div>
                <div style={{ fontSize: 12, color: "#999", marginBottom: 4 }}>Expected</div>
                <div style={box}>{selCase.expected}</div>
              </div>
              {selResult && (
                <div>
                  <div style={{ fontSize: 12, color: "#999", marginBottom: 4 }}>Output</div>
                  <div style={{ ...box, color: selResult.ok ? "#4ade80" : "#f87171" }}>
                    {selResult.error ?? selResult.actual}
                  </div>
                </div>
              )}
              {run?.stdout && (
                <div>
                  <div style={{ fontSize: 12, color: "#999", marginBottom: 4 }}>Stdout</div>
                  <div style={box}>{run.stdout}</div>
                </div>
              )}
            </div>
          </div>
        </div>

        <AgentPanel
          d={delib}
          onAskMore={() => socketRef.current?.send({ type: "ask_nudge", ts: Date.now(), problem_id: problem.id })}
          onChat={(text) =>
            socketRef.current?.send({ type: "chat", ts: Date.now(), problem_id: problem.id, text })
          }
          traceCase={traceCase}
          tracing={tracing}
          traceError={traceError}
          onTrace={() => traceCase !== null && startTrace(traceCase)}
        />
      </div>

      {openTrace && openTrace.target.problemId === problem.id && (
        <TraceViewer
          problem={problem}
          caseIndex={openTrace.target.caseIndex}
          code={openTrace.target.code}
          result={openTrace.result}
          onClose={closeTrace}
        />
      )}
    </div>
  );
}