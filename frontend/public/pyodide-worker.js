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

self.onmessage = async (e) => {
  const { id, code, fnName, cases } = e.data;
  const pyodide = await pyodideReady;
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