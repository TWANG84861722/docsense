"""Phone Q&A web service (LAN edition).

Run it on the Mac:  python server.py
Then, from a phone (on the same WiFi), open the http://<Mac's IP>:8000 that is printed on startup.

- The "brain" (embedding/reranking/index/papers) all lives on the Mac; the phone is just a thin
  client (an input box + an answer area).
- Reuses chat.py's existing condense_question + map_reduce; no duplicate retrieval logic.
- "Speaking a question" on the phone uses the keyboard's built-in dictation (its recognition is
  better than desktop Whisper), so no extra code is needed.
- Listens on the LAN only, with no authentication -- for use on a trusted home WiFi only; do not
  expose port 8000 to the public internet.

Dependencies (optional):  pip install fastapi uvicorn
"""
import socket
import logging
import threading
import time
import uuid

from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse, Response
from pydantic import BaseModel
import uvicorn

import chat
from config import MAX_HISTORY_TURNS

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")

app = FastAPI()

# Single-user personal tool: the server keeps one global conversation history (matching the
# terminal REPL's behavior).
_history = []

# Background job table: one Q&A is slow (possibly a minute+); if the phone just hangs waiting,
# iOS fetch times out and drops the connection at ~60s. So instead: ask → return a job_id
# immediately → the phone polls /result every few seconds. Each poll returns instantly and never
# times out.
_jobs = {}                 # job_id -> {status, answer, sources, standalone, error, started}
_job_lock = threading.Lock()   # models/retrieval aren't guaranteed thread-safe → run only one job at a time


class Ask(BaseModel):
    question: str


def _lan_ip():
    """Get this machine's LAN IP (to print the address the phone should visit)."""
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("8.8.8.8", 80))      # doesn't actually send a packet; just lets the OS pick the outbound interface's IP
        return s.getsockname()[0]
    except Exception:
        return "127.0.0.1"
    finally:
        s.close()


PAGE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>docsense</title>
<style>
  :root { color-scheme: light dark; }
  * { box-sizing: border-box; }
  body { font-family: -apple-system, system-ui, sans-serif; margin: 0; padding: 16px;
         max-width: 760px; margin: 0 auto; line-height: 1.6; }
  h1 { font-size: 20px; margin: 4px 0 12px; }
  #q { width: 100%; font-size: 17px; padding: 12px; border: 1px solid #bbb;
       border-radius: 10px; resize: vertical; min-height: 64px; }
  .row { display: flex; gap: 8px; margin-top: 10px; }
  button { font-size: 16px; padding: 10px 16px; border: 0; border-radius: 10px;
           background: #2563eb; color: #fff; }
  button.ghost { background: #6b7280; }
  button:disabled { opacity: .5; }
  #hint { color: #888; font-size: 13px; margin: 6px 2px; }
  #asked { font-weight: 600; margin: 16px 2px 2px; }
  #rewrite { color: #888; font-size: 13px; margin: 4px 2px 0; }
  #answer { white-space: pre-wrap; margin-top: 14px; padding: 14px;
            border: 1px solid #ddd; border-radius: 10px; min-height: 40px; }
  #sources { margin-top: 14px; }
  .src { font-size: 13px; color: #555; border-top: 1px solid #eee; padding: 8px 2px; }
  .tag { display: inline-block; background: #eef; color: #446; border-radius: 6px;
         padding: 0 6px; font-size: 12px; margin-left: 4px; }
  .spin { color: #2563eb; }
</style>
</head>
<body>
  <h1>📚 docsense — ask your library</h1>
  <textarea id="q" placeholder="Type, or tap 🎤 on the keyboard to speak (any language)…"></textarea>
  <div id="hint">Tip: use the keyboard's dictation for voice. Ask in any language — answers are in English.</div>
  <div class="row">
    <button id="ask" onclick="ask()">Ask</button>
    <button class="ghost" onclick="reset()">New topic</button>
  </div>
  <div id="asked"></div>
  <div id="rewrite"></div>
  <div id="answer"></div>
  <div id="sources"></div>

<script>
const sleep = ms => new Promise(r => setTimeout(r, ms));

function render(d, q) {
  if (d.standalone && d.standalone !== q)
    document.getElementById('rewrite').textContent = 'Search query → ' + d.standalone;
  document.getElementById('answer').textContent = d.answer || '(nothing relevant found)';
  const box = document.getElementById('sources');
  box.innerHTML = '';
  (d.sources || []).forEach((s, i) => {
    const div = document.createElement('div');
    div.className = 'src';
    let tag = s.type === 'figure' ? '<span class="tag">figure</span>'
            : s.type === 'table'  ? '<span class="tag">table</span>' : '';
    div.innerHTML = '[' + (i+1) + '] ' + s.paper + ' p.' + s.page +
                    (s.section ? ' · ' + s.section : '') + tag +
                    ' <span style="color:#999">(rerank=' + s.rerank.toFixed(3) + ')</span><br>' +
                    '<span style="color:#888">' + s.snippet + '…</span>';
    box.appendChild(div);
  });
}

async function ask() {
  const q = document.getElementById('q').value.trim();
  if (!q) return;
  const btn = document.getElementById('ask');
  const ans = document.getElementById('answer');
  btn.disabled = true;
  document.getElementById('asked').textContent = 'Q: ' + q;   // keep the question above the answer; it stays after asking
  document.getElementById('rewrite').textContent = '';
  document.getElementById('sources').innerHTML = '';
  ans.innerHTML = '<span class="spin">🤔 Searching & answering…</span>';
  try {
    // 1) Submit the job and get a job_id immediately (this step is fast and won't time out).
    const r = await fetch('/ask', {
      method: 'POST', headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({question: q})
    });
    const j = await r.json();
    if (j.error || !j.job_id) { ans.textContent = 'Error: ' + (j.error || 'no job id'); return; }
    // 2) Poll every N milliseconds until done/error (each poll returns instantly, never times out).
    while (true) {
      await sleep(3000);       // poll interval (ms): 3000 = ask once every 3 seconds
      const rr = await fetch('/result/' + j.job_id);
      const d = await rr.json();
      if (d.status === 'running') {
        ans.innerHTML = '<span class="spin">🤔 Searching & answering… ' + d.elapsed + 's</span>';
        continue;
      }
      if (d.status === 'error') { ans.textContent = 'Error: ' + d.error; return; }
      render(d, q);                       // done
      document.getElementById('q').value = '';
      return;
    }
  } catch (e) {
    ans.textContent = 'Request failed: ' + e;
  } finally {
    btn.disabled = false;
  }
}
async function reset() {
  await fetch('/reset', {method: 'POST'});
  document.getElementById('asked').textContent = '';
  document.getElementById('rewrite').textContent = '';
  document.getElementById('answer').textContent = '';
  document.getElementById('sources').innerHTML = '';
}
</script>
</body>
</html>"""


@app.get("/", response_class=HTMLResponse)
def home():
    # no-store: makes the phone browser fetch the latest page every time, so a code change isn't masked by an old cached page
    return HTMLResponse(PAGE, headers={"Cache-Control": "no-store"})


@app.get("/favicon.ico")
def favicon():
    return Response(status_code=204)      # the browser asks for a favicon; return an empty response so the log isn't spammed with 404s


@app.post("/reset")
def reset():
    _history.clear()
    return {"ok": True}


def _run_job(job_id, question):
    """Background thread: the actual slow work (condense + map_reduce); the result is written back to _jobs[job_id]."""
    with _job_lock:                       # run only one at a time: models/retrieval are not thread-safe
        try:
            standalone = chat.condense_question(question, _history)
            logging.info("Received question: %r  →  condensed to English: %r", question, standalone)
            answer, sources = chat.map_reduce(standalone)

            _history.append({"role": "user", "content": question})
            _history.append({"role": "assistant", "content": answer})
            if len(_history) > MAX_HISTORY_TURNS * 2:
                del _history[:-(MAX_HISTORY_TURNS * 2)]

            _jobs[job_id].update(
                status="done",
                standalone=standalone,
                answer=answer,
                sources=[
                    {
                        "paper": s["paper"], "page": s["page"],
                        "section": (s.get("section") or "").strip(),
                        "type": s.get("type", "text"),
                        "rerank": float(s.get("rerank_score", 0)),
                        "snippet": s["text"][:120].replace("\n", " ").strip(),
                    }
                    for s in sources
                ],
            )
        except Exception as e:
            logging.exception("job failed")
            _jobs[job_id].update(status="error", error=str(e))


@app.post("/ask")
def ask(req: Ask):
    """Return a job_id immediately; the work is handed to a background thread, and the phone polls /result/<job_id>."""
    question = req.question.strip()
    if not question:
        return JSONResponse({"error": "empty question"})
    job_id = uuid.uuid4().hex
    _jobs[job_id] = {"status": "running", "started": time.time()}
    threading.Thread(target=_run_job, args=(job_id, question), daemon=True).start()
    return {"job_id": job_id}


@app.get("/result/{job_id}")
def result(job_id: str):
    job = _jobs.get(job_id)
    if not job:
        return JSONResponse({"status": "error", "error": "unknown job"})
    out = {"status": job["status"], "elapsed": round(time.time() - job["started"])}
    if job["status"] == "done":
        out.update(answer=job["answer"], standalone=job["standalone"], sources=job["sources"])
        _jobs.pop(job_id, None)           # cleared once fetched, so we don't accumulate memory
    elif job["status"] == "error":
        out["error"] = job.get("error", "unknown")
        _jobs.pop(job_id, None)
    return out


if __name__ == "__main__":
    ip = _lan_ip()
    print("\n" + "=" * 56)
    print("  docsense phone Q&A is up. On your phone (same WiFi), open:")
    print(f"      http://{ip}:8000")
    print("  Stop: Ctrl-C")
    print("=" * 56 + "\n")
    uvicorn.run(app, host="0.0.0.0", port=8000)
