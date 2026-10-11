importScripts("https://cdn.jsdelivr.net/pyodide/v0.26.4/full/pyodide.js");

const pyodideReady = loadPyodide().then((p) => {
  self.postMessage({ ready: true });
  return p;
});

const HARNESS = `
import json, sys, io
cases = json.loads(cases_json)
ns = {}
out = []
buf = io.StringIO()
old = sys.stdout
sys.stdout = buf
compile_err = None
try:
    exec(user_code, ns)
except BaseException as e:
    compile_err = type(e).__name__ + ": " + str(e)
if compile_err is None:
    fn = ns.get(fn_name)
    for c in cases:
        try:
            if fn is None:
                raise NameError("function " + fn_name + " is not defined")
            args = [eval(a, {}) for a in c["args"]]
            actual = fn(*args)
            ok = actual == eval(c["expected"], {})
            out.append({"ok": ok, "actual": repr(actual), "error": None})
        except BaseException as e:
            out.append({"ok": False, "actual": None, "error": type(e).__name__ + ": " + str(e)})
sys.stdout = old
json.dumps({"compile_error": compile_err, "cases": out, "stdout": buf.getvalue()})
`;

// Step-by-step trace of the learner's own code on one test case. Records each line
// about to run (and each return) with the locals at that moment. Never runs or shows
// a reference solution.
const TRACE_HARNESS = `
import json, sys, io
MAX_STEPS = 1500
MAX_REPR = 80
MAX_LIST = 20
steps = []
spec = json.loads(trace_spec_json)
LABELS = spec.get("label") or []   # candidate expressions; the first that works in a call wins
GOAL = spec.get("goal")            # e.g. "label == amount"
calls = []        # one entry per function call, for the call tree
call_of = {}      # live frame -> its call id
shared_seen = {}  # live frame -> last seen state of its shared variables

def shared_state(frame):
    # Variables a call shares with other calls: closure variables (e.g. a "res" or "best"
    # it records answers in) and dict/set arguments (e.g. a memo). Plain list arguments
    # like a backtracking path are excluded: changing those is making a choice, not saving one.
    names = list(frame.f_code.co_freevars)
    names += [
        n for n in frame.f_code.co_varnames[: frame.f_code.co_argcount]
        if isinstance(frame.f_locals.get(n), (dict, set))
    ]
    return {n: short(frame.f_locals.get(n)) for n in names if n in frame.f_locals}

def note_shared(frame):
    if frame not in call_of:
        return
    now = shared_state(frame)
    if now != shared_seen.get(frame, now):
        calls[call_of[frame]]["updatedShared"] = True
    shared_seen[frame] = now

class TooManySteps(Exception):
    pass

def short(v):
    try:
        r = repr(v)
    except BaseException:
        r = "<unprintable>"
    return r if len(r) <= MAX_REPR else r[: MAX_REPR - 1] + "…"

def snapshot(v):
    # Short lists/tuples of simple values are drawn as index boxes; everything else is text.
    if isinstance(v, (list, tuple)) and len(v) <= MAX_LIST and all(
        isinstance(x, (int, float, str, bool, type(None))) for x in v
    ):
        return {"kind": "list", "items": [short(x) for x in v]}
    return {"kind": "value", "repr": short(v)}

def tracer(frame, event, arg):
    if frame.f_code.co_filename != "<user>":
        return None
    if event == "call":
        # Arguments are captured now: lists like a backtracking "subset" change afterwards.
        code = frame.f_code
        params = code.co_varnames[: code.co_argcount + code.co_kwonlyargcount]
        parent, f = None, frame.f_back
        while f is not None and parent is None:
            parent = call_of.get(f)
            f = f.f_back
        call_of[frame] = len(calls)
        calls.append({
            "id": len(calls),
            "parent": parent,
            "fn": code.co_name,
            "args": [short(frame.f_locals[p]) for p in params if p in frame.f_locals],
            "argNames": [p for p in params if p in frame.f_locals],
            "firstStep": len(steps),
            "returnStep": None,
            "returnValue": None,
            "updatedShared": False,
        })
        label, goal = None, None
        scope = dict(frame.f_locals)
        for expr in LABELS:
            try:
                label = eval(expr, frame.f_globals, scope)
                break
            except BaseException:
                continue
        if label is not None and GOAL:
            try:
                goal = bool(eval(GOAL, frame.f_globals, {**scope, "label": label}))
            except BaseException:
                goal = None
        calls[-1]["label"] = None if label is None else short(label)
        calls[-1]["goal"] = goal
        shared_seen[frame] = shared_state(frame)
        return tracer
    if event in ("line", "return"):
        if len(steps) >= MAX_STEPS:
            raise TooManySteps()
        note_shared(frame)
        depth, f = 0, frame.f_back
        while f is not None:
            depth += f.f_code.co_filename == "<user>"
            f = f.f_back
        steps.append({
            "event": event,
            "line": frame.f_lineno,
            "fn": frame.f_code.co_name,
            "depth": depth,
            # Functions, classes and modules are definitions, not state; leave them out.
            "locals": {
                k: snapshot(v)
                for k, v in frame.f_locals.items()
                if not k.startswith("__") and not callable(v) and type(v).__name__ != "module"
            },
            "returnValue": short(arg) if event == "return" else None,
            "call": call_of.get(frame),
        })
        if event == "return" and frame in call_of:
            c = calls[call_of.pop(frame)]
            c["returnStep"], c["returnValue"] = len(steps) - 1, short(arg)
            shared_seen.pop(frame, None)
            # Whatever this call changed must not be credited to its caller.
            f = frame.f_back
            while f is not None and f not in call_of:
                f = f.f_back
            if f is not None:
                shared_seen[f] = shared_state(f)
    return tracer

returned, matches, error, truncated = None, None, None, False
buf = io.StringIO()
old = sys.stdout
sys.stdout = buf
try:
    ns = {}
    exec(compile(user_code, "<user>", "exec"), ns)
    fn = ns.get(fn_name)
    if fn is None:
        raise NameError("function " + fn_name + " is not defined")
    args = [eval(a, {}) for a in json.loads(args_json)]
    sys.settrace(tracer)
    try:
        actual = fn(*args)
    finally:
        sys.settrace(None)
    returned = short(actual)
    matches = actual == eval(expected, {})
except TooManySteps:
    truncated = True
except BaseException as e:
    # An exception unwinding a frame is reported as "return None"; those steps would mislead.
    while steps and steps[-1]["event"] == "return":
        steps.pop()
    for c in calls:
        if c["returnStep"] is not None and c["returnStep"] >= len(steps):
            c["returnStep"], c["returnValue"] = None, None
    error = {"message": type(e).__name__ + ": " + str(e), "line": steps[-1]["line"] if steps else None}
finally:
    sys.stdout = old
json.dumps({"steps": steps, "calls": calls, "returned": returned, "matches": matches, "error": error, "truncated": truncated})
`;

self.onmessage = async (e) => {
  const pyodide = await pyodideReady;
  if (e.data.type === "trace") {
    const { id, code, fnName, args, expected, spec } = e.data;
    try {
      pyodide.globals.set("user_code", code);
      pyodide.globals.set("fn_name", fnName);
      pyodide.globals.set("args_json", JSON.stringify(args));
      pyodide.globals.set("expected", expected);
      pyodide.globals.set("trace_spec_json", JSON.stringify(spec ?? {}));
      const json = pyodide.runPython(TRACE_HARNESS);
      self.postMessage({ kind: "trace", id, ok: true, trace: JSON.parse(json) });
    } catch (err) {
      self.postMessage({ kind: "trace", id, ok: false, error: String(err) });
    }
    return;
  }

  const { id, code, fnName, cases } = e.data;
  try {
    pyodide.globals.set("user_code", code);
    pyodide.globals.set("fn_name", fnName);
    pyodide.globals.set("cases_json", JSON.stringify(cases));
    const json = pyodide.runPython(HARNESS);
    self.postMessage({ id, ok: true, result: JSON.parse(json) });
  } catch (err) {
    self.postMessage({ id, ok: false, error: String(err) });
  }
};