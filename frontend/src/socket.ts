export type OutMsg =
  | { type: "problem"; ts: number; problem_id: string }
  | {
      type: "edit";
      ts: number;
      problem_id: string;
      snapshot: string;
      diff_summary: { added: number; deleted: number };
    }
  | {
      type: "run";
      ts: number;
      problem_id: string;
      passed: boolean;
      first_error: string | null;
      snapshot?: string; // the code that ran
    }
  | { type: "chat"; ts: number; problem_id: string; text: string }
  | { type: "ask_nudge"; ts: number; problem_id: string };

export type Proposal = {
  agent: string;
  level: number;
  confidence: number;
  tags: string[];
  cap?: number; // Integrity's hard cap, if the backend sends it separately
  responses?: Response[]; // this agent's verdicts on peers' arguments
};

export type Response = {
  to: string; // peer agent name, lowercase
  claim: string; // the reason tag the peer argued
  accepted: boolean;
  holds: number | null; // Jev's probability that the peer's argument holds
};

export type InMsg =
  | { type: "round"; round: number; proposals: Proposal[] }
  | { type: "decision"; outcome: string; level: number; explanation: string }
  | { type: "hint"; level: number; text: string };

const WS_URL = "ws://localhost:8000/ws";

export class Socket {
  private ws: WebSocket | null = null;
  private queue: OutMsg[] = [];
  private closed = false;
  private onMessage: (m: InMsg) => void;
  private onStatus: (connected: boolean) => void;

  constructor(
    onMessage: (m: InMsg) => void,
    onStatus: (connected: boolean) => void
  ) {
    this.onMessage = onMessage;
    this.onStatus = onStatus;
    this.connect();
  }

  private connect() {
    if (this.closed) return;
    const ws = new WebSocket(WS_URL);
    this.ws = ws;
    ws.onopen = () => {
      this.onStatus(true);
      this.queue.forEach((m) => ws.send(JSON.stringify(m)));
      this.queue = [];
    };
    ws.onmessage = (e) => {
      try {
        this.onMessage(JSON.parse(e.data));
      } catch {
        console.warn("Bad message", e.data);
      }
    };
    ws.onclose = () => {
      if (this.closed) return;
      this.onStatus(false);
      setTimeout(() => this.connect(), 1500);
    };
    ws.onerror = () => ws.close();
  }

  send(m: OutMsg) {
    console.log("send", m);
    if (this.ws?.readyState === WebSocket.OPEN) {
      this.ws.send(JSON.stringify(m));
    } else {
      this.queue.push(m);
      if (this.queue.length > 50) this.queue.shift();
    }
  }

  close() {
    this.closed = true;
    this.ws?.close();
  }
}