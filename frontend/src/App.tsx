import { useEffect, useRef, useState } from "react";
import Editor from "@monaco-editor/react";
import { SECOND_LARGEST_STARTER, SECOND_LARGEST_TESTS } from "./problems";

type RunResult = {
  passed: boolean;
  first_error: string | null;
  output: string;
};

export default function App() {
  const [code, setCode] = useState(SECOND_LARGEST_STARTER);
  const [ready, setReady] = useState(false);
  const [running, setRunning] = useState(false);
  const [result, setResult] = useState<RunResult | null>(null);

  const workerRef = useRef<Worker | null>(null);
  const timerRef = useRef<number | undefined>(undefined);

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
      const ok: boolean = e.data.ok;
      setResult({
        passed: ok,
        first_error: ok ? null : String(e.data.error).trim().split("\n").pop() ?? null,
        output: e.data.output ?? "",
      });
    };
    workerRef.current = w;
  }

  useEffect(() => {
    startWorker();
    return () => workerRef.current?.terminate();
  }, []);

  function run() {
    if (!ready || running) return;
    setRunning(true);
    workerRef.current?.postMessage({
      id: Date.now(),
      code: code + "\n" + SECOND_LARGEST_TESTS,
    });
    // Kill runaway code (e.g. infinite loops) after 5 seconds
    timerRef.current = window.setTimeout(() => {
      startWorker();
      setRunning(false);
      setResult({
        passed: false,
        first_error: "Timed out after 5s (infinite loop?)",
        output: "",
      });
    }, 5000);
  }

  return (
    <div style={{ display: "flex", height: "100vh" }}>
      <div style={{ flex: 2, display: "flex", flexDirection: "column" }}>
        <div style={{ padding: 8, borderBottom: "1px solid #333" }}>
          <button onClick={run} disabled={!ready || running}>
            {!ready ? "Loading Python..." : running ? "Running..." : "Run"}
          </button>
        </div>
        <div style={{ flex: 1 }}>
          <Editor
            defaultLanguage="python"
            defaultValue={SECOND_LARGEST_STARTER}
            theme="vs-dark"
            onChange={(value) => setCode(value ?? "")}
          />
        </div>
        <div style={{ padding: 8, borderTop: "1px solid #333", minHeight: 60 }}>
          {result === null ? (
            "Press Run to test your code."
          ) : result.passed ? (
            <span style={{ color: "#4ade80" }}>All tests passed</span>
          ) : (
            <span style={{ color: "#f87171" }}>{result.first_error}</span>
          )}
        </div>
      </div>
      <div style={{ flex: 1, padding: 12, borderLeft: "1px solid #333" }}>
        Agent panel goes here
      </div>
    </div>
  );
}