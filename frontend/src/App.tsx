import { useEffect, useRef, useState } from "react";
import Editor from "@monaco-editor/react";
import { PROBLEMS, type Problem } from "./problems";
import { Socket } from "./socket";
import AgentPanel, { applyMessage, emptyDeliberation, type Deliberation } from "./AgentPanel";

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

  const problem = PROBLEMS[problemIndex];

  const workerRef = useRef<Worker | null>(null);
  const timerRef = useRef<number | undefined>(undefined);
  const socketRef = useRef<Socket | null>(null);

  // Per-problem code and last-sent snapshot, so switching problems keeps your work
  const codesRef = useRef<Record<string, string>>({});
  const snapshotsRef = useRef<Record<string, string>>({});
  const problemRef = useRef<Problem>(problem);
  problemRef.current = problem;

  function currentCode(p: Problem) {
    return codesRef.current[p.id] ?? p.starter;
  }

  function sendRunEvent(r: RunState) {
    const p = problemRef.current;
    socketRef.current?.send({
      type: "run",
      ts: Date.now(),
      problem_id: p.id,
      passed: verdictOf(r) === "Accepted",
      first_error: firstErrorOf(r, p),
    });
  }

  function startWorker() {
    workerRef.current?.terminate();
    setReady(false);
    const w = new Worker("/pyodide-worker.js");
    w.onmessage = (e) => {
      if (e.data.ready) {
        setReady(true);
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
    };
    workerRef.current = w;
  }

  useEffect(() => {
    startWorker();
    return () => workerRef.current?.terminate();
  }, []);

  useEffect(() => {
    const s = new Socket((m) => {
      console.log("recv", m);
      setDelib((prev) => applyMessage(prev, m));
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
  }, []);

  function switchProblem(i: number) {
    setProblemIndex(i);
    setRun(null);
    setSelected(0);
    setDelib(emptyDeliberation);
    socketRef.current?.send({ type: "problem", ts: Date.now(), problem_id: PROBLEMS[i].id });
  }

  function runTests() {
    if (!ready || running) return;
    setRunning(true);
    workerRef.current?.postMessage({
      id: Date.now(),
      code: currentCode(problem),
      fnName: problem.fnName,
      cases: problem.cases,
    });
    // Kill runaway code (e.g. infinite loops) after 5 seconds
    timerRef.current = window.setTimeout(() => {
      startWorker();
      setRunning(false);
      const r: RunState = { timedOut: true, compileError: null, cases: [], stdout: "" };
      setRun(r);
      sendRunEvent(r);
    }, 5000);
  }

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
        <button onClick={runTests} disabled={!ready || running}>
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
                  codesRef.current[problemRef.current.id] = value ?? "";
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
            </div>

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
          onAskMore={() =>
            socketRef.current?.send({
              type: "ask_nudge",
              ts: Date.now(),
              problem_id: problemRef.current.id,
            })
          }
        />
      </div>
    </div>
  );
}