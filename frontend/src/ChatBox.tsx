import { useState } from "react";

export default function ChatBox({ onSend }: { onSend: (text: string) => void }) {
  const [text, setText] = useState("");
  const [sent, setSent] = useState<string[]>([]);

  const submit = () => {
    const t = text.trim();
    if (!t) return;
    onSend(t);
    setSent((s) => [...s, t].slice(-4));
    setText("");
  };

  return (
    <div style={{ position: "sticky", bottom: 0, background: "#1e1e1e", borderTop: "1px solid #333", padding: 10 }}>
      {sent.map((m, i) => (
        <div
          key={i}
          style={{ fontSize: 12, color: "#ccc", background: "#2a2a2a", borderRadius: 8, padding: "4px 8px", marginBottom: 4 }}
        >
          {m}
        </div>
      ))}
      <div style={{ display: "flex", gap: 6, marginTop: 4 }}>
        <input
          value={text}
          onChange={(e) => setText(e.target.value)}
          onKeyDown={(e) => e.key === "Enter" && submit()}
          placeholder="Message the agents…"
          style={{ flex: 1, background: "#2a2a2a", color: "#ddd", border: "1px solid #444", borderRadius: 6, padding: "6px 8px", fontSize: 13 }}
        />
        <button
          onClick={submit}
          disabled={!text.trim()}
          style={{
            background: "#3b82f6",
            color: "#fff",
            border: "none",
            borderRadius: 6,
            padding: "6px 12px",
            cursor: text.trim() ? "pointer" : "default",
            opacity: text.trim() ? 1 : 0.5,
          }}
        >
          Send
        </button>
      </div>
    </div>
  );
}