importScripts("https://cdn.jsdelivr.net/pyodide/v0.26.4/full/pyodide.js");

const pyodideReady = loadPyodide().then((p) => {
  self.postMessage({ ready: true });
  return p;
});

self.onmessage = async (e) => {
  const { id, code } = e.data;
  const pyodide = await pyodideReady;
  try {
    pyodide.runPython(`
import sys, io
sys.stdout = io.StringIO()
`);
    await pyodide.runPythonAsync(code);
    const out = pyodide.runPython("sys.stdout.getvalue()");
    self.postMessage({ id, ok: true, output: out });
  } catch (err) {
    self.postMessage({ id, ok: false, error: String(err) });
  }
};