const { WebSocketServer } = require("ws");
const wss = new WebSocketServer({ port: 8000, path: "/ws" });

const learning = (level, confidence, tags) => ({ agent: "learning", level, confidence, tags });
const momentum = (level, confidence, tags) => ({ agent: "momentum", level, confidence, tags });
// Integrity sets a cap, not a vote: its level equals its cap.
const integrity = (cap, confidence, tags) => ({ agent: "integrity", level: cap, confidence, tags, cap });

// Each failed Run (or a chat containing "stuck"/"help") plays the next scenario.
const SCENARIOS = [
  {
    rounds: [
      [learning(1, 0.55, ["active_progress"]), momentum(2, 0.8, ["repeated_error"]), integrity(2, 0.9, ["solution_leak_risk"])],
      [learning(2, 0.65, ["active_progress"]), momentum(2, 0.8, ["repeated_error"]), integrity(2, 0.9, ["solution_leak_risk"])],
    ],
    decision: {
      outcome: "consensus",
      level: 2,
      explanation: "Learning relaxed once it saw the same error repeat. All three agents agree on a conceptual hint.",
    },
    hint: { level: 2, text: "Think about what happens to your result when the list has duplicate values." },
  },
  {
    rounds: [
      [learning(0, 0.7, ["active_progress"]), momentum(2, 0.6, ["repeated_error"]), integrity(2, 0.9, ["solution_leak_risk"])],
      [learning(1, 0.6, ["active_progress"]), momentum(2, 0.6, ["repeated_error"]), integrity(2, 0.9, ["solution_leak_risk"])],
      [learning(1, 0.55, ["active_progress"]), momentum(2, 0.65, ["repeated_error"]), integrity(2, 0.9, ["solution_leak_risk"])],
    ],
    decision: {
      outcome: "soft_disagreement",
      level: 1,
      explanation: "Learning wanted to wait because the learner is actively rewriting; Momentum saw repeated errors. Acting at the lower level and offering more.",
    },
    hint: { level: 1, text: "Look at your error message carefully." },
  },
  {
    rounds: [
      [learning(0, 0.75, ["active_progress"]), momentum(2, 0.7, ["long_idle"]), integrity(3, 0.9, [])],
      [learning(0, 0.7, ["active_progress"]), momentum(2, 0.7, ["long_idle"]), integrity(3, 0.9, [])],
    ],
    decision: {
      outcome: "deadlock",
      level: 0,
      explanation: "Learning and Momentum are two levels apart and neither moved. Keeping the disagreement and asking the learner.",
    },
    hint: null,
  },
  {
    // Momentum votes above Integrity's cap, so the UI shows "clamped to L2"
    rounds: [
      [learning(1, 0.6, ["active_progress"]), momentum(3, 0.85, ["repeated_error", "learner_requested_help"]), integrity(2, 0.9, ["solution_leak_risk"])],
      [learning(2, 0.65, ["active_progress"]), momentum(3, 0.85, ["repeated_error", "learner_requested_help"]), integrity(2, 0.9, ["solution_leak_risk"])],
    ],
    decision: {
      outcome: "consensus",
      level: 2,
      explanation: "Momentum wanted to point at the line (L3), but Integrity capped help at L2 to avoid giving away the solution. Clamped to L2, Learning agrees.",
    },
    hint: { level: 2, text: "Consider how your loop treats the first element compared to the rest." },
  },
];

let scenarioIndex = 0;

wss.on("connection", (ws) => {
  console.log("client connected");
  const send = (m) => ws.send(JSON.stringify(m));

  function playNext() {
    const s = SCENARIOS[scenarioIndex++ % SCENARIOS.length];
    s.rounds.forEach((proposals, i) => setTimeout(() => send({ type: "round", round: i, proposals }), i * 50));
    setTimeout(() => {
      send({ type: "decision", ...s.decision });
      if (s.hint) send({ type: "hint", ...s.hint });
    }, s.rounds.length * 50 + 50);
  }

  ws.on("message", (raw) => {
    const m = JSON.parse(raw);
    console.log("got", m.type, m.problem_id || "", m.text || "");

    if (m.type === "run" && !m.passed) playNext();

    if (m.type === "chat") {
      const t = (m.text || "").trim().toLowerCase();
      if (t && (t.includes("stuck") || t.includes("help"))) playNext();
    }

    if (m.type === "ask_nudge") {
      setTimeout(() => send({ type: "hint", level: 2, text: "Try sorting the values first, then think about which position you need." }), 400);
    }
  });
});
console.log("mock backend on ws://localhost:8000/ws");