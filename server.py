"""
FastAPI server — OpenEnv HTTP API for the CRISPR Guide RNA environment.

Endpoints
---------
POST /reset          body: {"task_name": "single_guide_easy", "session_id": "<optional>"}
POST /step           body: {"guide_sequence": "ACGTACGTACGTACGTACGT", "rationale": "...", "session_id": "<optional>"}
GET  /state          ?session_id=<optional>
GET  /tasks          lists all available tasks
GET  /health         liveness check
GET  /               web dashboard (HTML)

Session management
------------------
Each session_id (UUID string) maps to an independent CRISPREnv instance,
allowing concurrent evaluators or users without state collision.
If session_id is omitted, the request uses the shared "default" session.
"""

from __future__ import annotations

import os
import sys
import threading

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel
from typing import Optional

from src.env import CRISPREnv
from src.models import CRISPRAction
from src.tasks import TASKS
# ── App setup ────────────────────────────────────────────────────────────────

app = FastAPI(
    title="CRISPR Guide RNA Optimizer — OpenEnv",
    version="2.0.0",
    description="AI Agent environment for CRISPR guide RNA design. OpenEnv-compliant.",
)

# Thread-safe session registry: session_id -> CRISPREnv
_sessions: dict[str, CRISPREnv] = {}
_sessions_lock = threading.Lock()
_DEFAULT_SESSION = "default"


def _get_env(session_id: str) -> Optional[CRISPREnv]:
    with _sessions_lock:
        return _sessions.get(session_id)


def _set_env(session_id: str, env: CRISPREnv) -> None:
    with _sessions_lock:
        _sessions[session_id] = env


# ── Request / Response schemas ────────────────────────────────────────────────

class ResetRequest(BaseModel):
    task_name: str = "single_guide_easy"
    session_id: str = _DEFAULT_SESSION


class StepRequest(BaseModel):
    guide_sequence: str
    rationale: Optional[str] = None
    session_id: str = _DEFAULT_SESSION


class StateRequest(BaseModel):
    session_id: str = _DEFAULT_SESSION


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok", "env": "crispr-guide-optimizer", "version": "2.0.0"}


@app.get("/tasks")
def list_tasks():
    return {
        "tasks": [
            {
                "name": t.name,
                "difficulty": t.difficulty,
                "max_steps": t.max_steps,
                "description": t.description,
                "success_threshold": t.success_threshold,
            }
            for t in TASKS.values()
        ]
    }


@app.post("/reset")
def reset(req: ResetRequest):
    try:
        env = CRISPREnv(task_name=req.task_name)
        obs = env.reset()
        _set_env(req.session_id, env)
        return obs.model_dump()
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Reset failed: {e}")


@app.post("/step")
def step(req: StepRequest):
    env = _get_env(req.session_id)
    if env is None:
        raise HTTPException(status_code=400, detail="Call /reset first.")
    if env._done:
        raise HTTPException(status_code=400, detail="Episode done. Call /reset.")
    try:
        action = CRISPRAction(guide_sequence=req.guide_sequence, rationale=req.rationale)
        obs, reward, done, info = env.step(action)
        return {
            "observation": obs.model_dump(),
            "reward": reward.model_dump(),
            "done": done,
            "info": info,
        }
    except Exception as e:
        raise HTTPException(status_code=422, detail=str(e))


@app.get("/state")
def state(session_id: str = _DEFAULT_SESSION):
    env = _get_env(session_id)
    if env is None:
        return {"state": None, "message": "No active episode. Call /reset first."}
    return env.state()


@app.get("/", response_class=HTMLResponse)
def dashboard():
    return HTMLResponse(content=DASHBOARD_HTML, status_code=200)


# ── Minimal dashboard HTML ────────────────────────────────────────────────────

DASHBOARD_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>CRISPR Guide RNA Optimizer — OpenEnv</title>
<style>
  body{font-family:'Segoe UI',Arial,sans-serif;background:#F8FAFC;color:#1E293B;margin:0;padding:0}
  header{background:linear-gradient(135deg,#1E3A5F,#2563EB);color:white;padding:1.5rem 2rem}
  h1{margin:0 0 .3rem;font-size:1.5rem}
  .sub{font-size:.85rem;opacity:.85}
  .container{max-width:900px;margin:2rem auto;padding:0 1rem}
  .card{background:white;border-radius:10px;padding:1.5rem;margin-bottom:1rem;
        box-shadow:0 1px 4px rgba(0,0,0,.08)}
  h2{margin:0 0 1rem;font-size:1.1rem;color:#1E3A5F}
  .grid{display:grid;grid-template-columns:1fr 1fr;gap:1rem}
  label{display:block;font-size:.8rem;font-weight:600;color:#475569;margin-bottom:.3rem}
  input,select,textarea{width:100%;padding:.5rem .75rem;border:1px solid #CBD5E1;
    border-radius:6px;font-size:.9rem;box-sizing:border-box}
  textarea{height:80px;font-family:monospace;font-size:.82rem}
  button{background:#2563EB;color:white;border:none;padding:.6rem 1.4rem;
    border-radius:6px;font-size:.9rem;cursor:pointer;font-weight:600}
  button:hover{background:#1D4ED8}
  pre{background:#1E293B;color:#F1F5F9;padding:1rem;border-radius:8px;
    font-size:.78rem;overflow-x:auto;white-space:pre-wrap;max-height:400px;overflow-y:auto}
  .tag{display:inline-block;padding:.2rem .6rem;border-radius:4px;font-size:.75rem;font-weight:600}
  .easy{background:#D1FAE5;color:#065F46}
  .medium{background:#FEF3C7;color:#92400E}
  .hard{background:#FEE2E2;color:#991B1B}
  .links{font-size:.82rem;color:#475569}
  .links a{color:#2563EB;margin-right:1rem}
</style>
</head>
<body>
<header>
  <h1>🧬 CRISPR Guide RNA Optimizer — OpenEnv Environment</h1>
  <p class="sub">Real NCBI sequences · Doench 2016 scoring · 3 tasks · Full step()/reset()/state() API</p>
</header>
<div class="container">

  <div class="card">
    <h2>Available Tasks</h2>
    <div id="taskList">Loading...</div>
  </div>

  <div class="card">
    <h2>1. Reset Episode</h2>
    <label>Task</label>
    <select id="taskSel">
      <option value="single_guide_easy">single_guide_easy (Easy)</option>
      <option value="ranked_panel_medium">ranked_panel_medium (Medium)</option>
      <option value="multi_gene_hard">multi_gene_hard (Hard)</option>
    </select>
    <br><br>
    <label>Session ID (optional — use different IDs for parallel sessions)</label>
    <input id="sessionId" type="text" value="default" placeholder="default">
    <br><br>
    <button onclick="doReset()">POST /reset</button>
    <pre id="resetOut">—</pre>
  </div>

  <div class="card">
    <h2>2. Take a Step</h2>
    <label>Guide sequence (exactly 20 nt, A/C/G/T only)</label>
    <input id="guideSeq" type="text" maxlength="20" placeholder="e.g. ATGGAGGAGCCGCAGTCAGA">
    <br><br>
    <label>Rationale (optional)</label>
    <textarea id="rationale" placeholder="Why this sequence?"></textarea>
    <br>
    <button onclick="doStep()">POST /step</button>
    <pre id="stepOut">—</pre>
  </div>

  <div class="card">
    <h2>3. State Snapshot</h2>
    <button onclick="doState()">GET /state</button>
    <pre id="stateOut">—</pre>
  </div>

  <div class="card links">
    <a href="/docs">📄 Swagger UI</a>
    <a href="/redoc">📚 ReDoc</a>
    <a href="/tasks">🗂 Task JSON</a>
    <a href="/health">❤️ Health</a>
    <a href="/state">📊 State</a>
  </div>

</div>
<script>
function sid(){return document.getElementById('sessionId').value.trim()||'default';}
async function loadTasks(){
  const r=await fetch('/tasks'); const d=await r.json();
  const diff={'easy':'easy','medium':'medium','hard':'hard'};
  document.getElementById('taskList').innerHTML=d.tasks.map(t=>`
    <p><strong>${t.name}</strong> <span class="tag ${diff[t.difficulty]}">${t.difficulty}</span>
    max_steps=${t.max_steps} · threshold=${t.success_threshold}<br>
    <small>${t.description.substring(0,120)}…</small></p>`).join('');
}
async function doReset(){
  const r=await fetch('/reset',{method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify({task_name:document.getElementById('taskSel').value,session_id:sid()})});
  document.getElementById('resetOut').textContent=JSON.stringify(await r.json(),null,2);
}
async function doStep(){
  const seq=document.getElementById('guideSeq').value.trim();
  const rat=document.getElementById('rationale').value.trim();
  const body={guide_sequence:seq,session_id:sid()};
  if(rat) body.rationale=rat;
  const r=await fetch('/step',{method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify(body)});
  document.getElementById('stepOut').textContent=JSON.stringify(await r.json(),null,2);
}
async function doState(){
  const r=await fetch('/state?session_id='+encodeURIComponent(sid()));
  document.getElementById('stateOut').textContent=JSON.stringify(await r.json(),null,2);
}
loadTasks();
</script>
</body>
</html>"""


# ── Entrypoint ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 7860))
    uvicorn.run("server:app", host="0.0.0.0", port=port, reload=False)