"""
Lightweight web dashboard — FastAPI JSON API + embedded single-page HTML.
Runs in a background thread; reads from a shared state dict updated by the sentinel.

Auth: optional. Call setup_auth(auth_db) before start() to protect all routes.
When auth is active, unauthenticated page requests are redirected to /login;
unauthenticated API requests get a 401 JSON response.
"""

import threading
import time
from typing import Any, Optional

import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse

# ---------------------------------------------------------------------------
# Shared state (written by sentinel tick, read by API)
# ---------------------------------------------------------------------------

_state: dict[str, Any] = {}
_lock = threading.Lock()
_auth = None   # AuthDB instance, set by setup_auth()


def setup_auth(auth_db) -> None:
    global _auth
    _auth = auth_db


def update_state(processes, session_log, scan_count: int, interval: float):
    """Called by the sentinel on every tick to push fresh data."""
    from .resolver import hostname

    def _short(ip, host):
        if host == ip:
            return ip
        parts = host.split(".")
        return ".".join(parts[-3:]) if len(parts) > 3 else host

    procs_out = []
    ext_out = []
    local_out = []

    for proc in processes:
        ext_count = sum(1 for c in proc.connections if c.is_external)
        procs_out.append({
            "pid": proc.pid,
            "name": proc.name,
            "status": proc.status,
            "cpu": round(proc.cpu_percent, 1),
            "mem_mb": round(proc.memory_mb),
            "ext_conns": ext_count,
            "cmd": proc.cmdline[:100],
        })

        groups: dict[tuple, int] = {}
        for conn in proc.connections:
            if conn.is_external:
                key = (conn.remote_ip, conn.remote_port, conn.protocol)
                groups[key] = groups.get(key, 0) + 1
            else:
                local_out.append({
                    "pid": proc.pid,
                    "name": proc.name,
                    "local": conn.local_addr,
                    "remote": conn.remote_addr,
                    "status": conn.status,
                })

        for (ip, port, proto), count in groups.items():
            host = hostname(ip)
            ext_out.append({
                "pid": proc.pid,
                "name": proc.name,
                "remote_ip": ip,
                "hostname": _short(ip, host),
                "port": port,
                "proto": proto.upper(),
                "count": count,
            })

    # Session history
    events = session_log.recent_events(limit=30)
    hist_out = []
    for ev in events:
        if ev.event == "seen":
            continue
        host = hostname(ev.remote_ip)
        short = _short(ev.remote_ip, host)
        still_open = ev.event == "opened" and ev.duration_s is None
        if still_open:
            dur = "still open"
            display_event = "open"
        elif ev.duration_s is not None:
            s = ev.duration_s
            dur = f"{s:.0f}s" if s < 60 else (f"{s/60:.1f}m" if s < 3600 else f"{s/3600:.1f}h")
            display_event = ev.event
        else:
            dur = "—"
            display_event = ev.event
        hist_out.append({
            "time": time.strftime("%H:%M:%S", time.localtime(ev.ts)),
            "event": display_event,
            "process": ev.process_name,
            "hostname": short,
            "ip": ev.remote_ip,
            "port": ev.port,
            "duration": dur,
        })

    # Insights
    raw = session_log.insights()
    insights_out = {}
    if raw:
        insights_out = {
            "session_age": _fmt_age(raw.get("session_age_s", 0)),
            "unique_destinations": raw.get("unique_hosts", 0),
            "total_connections": raw.get("total_connections", 0),
            "most_contacted": raw.get("most_contacted"),
            "phases": raw.get("phases", []),
        }
        if raw.get("longest"):
            ev = raw["longest"]
            host = hostname(ev.remote_ip)
            insights_out["longest"] = {
                "duration": _fmt_age(ev.duration_s or 0),
                "host": _short(ev.remote_ip, host),
                "process": ev.process_name,
            }

    alert = len(ext_out) > 0

    with _lock:
        _state.update({
            "scan": scan_count,
            "interval": interval,
            "alert": alert,
            "processes": procs_out,
            "external": ext_out,
            "local": local_out,
            "history": hist_out,
            "insights": insights_out,
            "updated_at": time.strftime("%H:%M:%S"),
            "_session_log": session_log,
        })


def _fmt_age(s: float) -> str:
    if s < 60:
        return f"{s:.0f}s"
    if s < 3600:
        return f"{s/60:.1f}m"
    return f"{s/3600:.1f}h"


# ---------------------------------------------------------------------------
# FastAPI app + auth middleware
# ---------------------------------------------------------------------------

app = FastAPI(title="LLM Sentinel")

_AUTH_EXEMPT = {"/login", "/api/login"}


@app.middleware("http")
async def auth_middleware(request: Request, call_next):
    if _auth is None or request.url.path in _AUTH_EXEMPT:
        return await call_next(request)
    token = request.cookies.get("sentinel_session", "")
    user = _auth.validate_session(token)
    if not user:
        if request.url.path.startswith("/api/"):
            return JSONResponse({"error": "unauthorized"}, status_code=401)
        return RedirectResponse("/login", status_code=302)
    return await call_next(request)


# ---------------------------------------------------------------------------
# API endpoints
# ---------------------------------------------------------------------------

@app.get("/api/state")
def get_state():
    with _lock:
        data = {k: v for k, v in _state.items() if not k.startswith("_")}
    return JSONResponse(data or {"scan": 0, "alert": False})


@app.get("/api/sessions")
def get_sessions():
    with _lock:
        sl = _state.get("_session_log")
    if sl is None:
        return JSONResponse([])
    sessions = sl.list_sessions()
    out = []
    for s in sessions:
        started = s["started_at"] or 0
        last = s["last_active"] or started
        duration_s = last - started
        out.append({
            "id": s["session_id"],
            "started": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(started)),
            "started_ts": started,
            "duration": _fmt_age(duration_s),
            "conn_count": s["conn_count"],
            "unique_ips": s["unique_ips"],
            "processes": s["processes"] or "",
            "is_current": s["is_current"],
        })
    return JSONResponse(out)


@app.get("/api/sessions/{session_id}")
def get_session_detail(session_id: str):
    with _lock:
        sl = _state.get("_session_log")
    if sl is None:
        return JSONResponse([])
    events = sl.get_session_events(session_id)
    out = []
    for ev in events:
        if ev.event == "seen":
            continue
        still_open = ev.event == "opened" and ev.duration_s is None
        dur = "still open" if still_open else (
            _fmt_age(ev.duration_s) if ev.duration_s is not None else "—"
        )
        out.append({
            "time": time.strftime("%H:%M:%S", time.localtime(ev.ts)),
            "event": "open" if still_open else ev.event,
            "process": ev.process_name,
            "hostname": ev.hostname if ev.hostname != ev.remote_ip else "",
            "ip": ev.remote_ip,
            "port": ev.port,
            "duration": dur,
        })
    return JSONResponse(out)


# ---------------------------------------------------------------------------
# Auth endpoints
# ---------------------------------------------------------------------------

@app.get("/login", response_class=HTMLResponse)
def login_page(error: str = ""):
    err_html = (
        f'<div class="err-alert">'
        f'<i class="bi bi-exclamation-triangle-fill"></i>{error}</div>'
        if error else ""
    )
    # _LOGIN_HTML is an f-string so {{ERROR}} rendered to {ERROR}
    return HTMLResponse(_LOGIN_HTML.replace("{ERROR}", err_html))


@app.post("/api/login")
async def do_login(request: Request):
    form = await request.form()
    username = str(form.get("username", "")).strip()
    password = str(form.get("password", ""))
    if _auth and _auth.verify_user(username, password):
        token = _auth.create_session(username)
        resp = RedirectResponse("/", status_code=302)
        resp.set_cookie(
            "sentinel_session", token,
            httponly=True, samesite="strict",
            max_age=8 * 3600,
        )
        return resp
    return RedirectResponse("/login?error=Invalid+credentials", status_code=302)


@app.get("/logout")
def logout(request: Request):
    token = request.cookies.get("sentinel_session", "")
    if _auth and token:
        _auth.revoke_session(token)
    resp = RedirectResponse("/login", status_code=302)
    resp.delete_cookie("sentinel_session")
    return resp


# ---------------------------------------------------------------------------
# Page routes
# ---------------------------------------------------------------------------

@app.get("/", response_class=HTMLResponse)
def index():
    return HTMLResponse(_HTML)


@app.get("/sessions", response_class=HTMLResponse)
def sessions_page():
    return HTMLResponse(_SESSIONS_HTML)


# ---------------------------------------------------------------------------
# Background server launcher
# ---------------------------------------------------------------------------

def start(host: str = "127.0.0.1", port: int = 7777):
    config = uvicorn.Config(app, host=host, port=port,
                            log_level="error", access_log=False)
    server = uvicorn.Server(config)
    t = threading.Thread(target=server.run, daemon=True)
    t.start()
    return f"http://{host}:{port}"


# ---------------------------------------------------------------------------
# Shared HTML fragments — enterprise design system
# ---------------------------------------------------------------------------

_BS = "https://cdn.jsdelivr.net/npm/bootstrap@5.3.3/dist/css/bootstrap.min.css"
_BS_JS = "https://cdn.jsdelivr.net/npm/bootstrap@5.3.3/dist/js/bootstrap.bundle.min.js"
_BI = "https://cdn.jsdelivr.net/npm/bootstrap-icons@1.11.3/font/bootstrap-icons.min.css"
_INTER = "https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap"

_THEME_SCRIPT = """
<script>
(function(){
  var t = localStorage.getItem('sentinel-theme') || 'dark';
  document.documentElement.setAttribute('data-bs-theme', t);
})();
</script>"""

_THEME_TOGGLE_JS = """
function toggleTheme() {
  var html = document.documentElement;
  var next = html.getAttribute('data-bs-theme') === 'dark' ? 'light' : 'dark';
  html.setAttribute('data-bs-theme', next);
  localStorage.setItem('sentinel-theme', next);
  var icon = document.getElementById('theme-icon');
  if (icon) icon.className = next === 'dark' ? 'bi bi-sun' : 'bi bi-moon-stars-fill';
}
"""

_SHARED_STYLE = """
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

  /* ── Design tokens ── */
  [data-bs-theme=dark] {
    --bg-base:    #0d1117;
    --bg-surface: #161b22;
    --bg-raised:  #1c2330;
    --border:     #30363d;
    --border-sub: #21262d;
    --text-1:     #e6edf3;
    --text-2:     #8b949e;
    --text-3:     #6e7681;
    --accent:     #58a6ff;
    --danger:     #f85149;
    --success:    #3fb950;
    --warning:    #e3b341;
    --danger-bg:  rgba(248,81,73,.12);
    --success-bg: rgba(63,185,80,.12);
    --warning-bg: rgba(227,179,65,.12);
  }
  [data-bs-theme=light] {
    --bg-base:    #f6f8fa;
    --bg-surface: #ffffff;
    --bg-raised:  #f0f2f5;
    --border:     #d0d7de;
    --border-sub: #eaeef2;
    --text-1:     #1f2328;
    --text-2:     #57606a;
    --text-3:     #848d97;
    --accent:     #0969da;
    --danger:     #cf222e;
    --success:    #1a7f37;
    --warning:    #9a6700;
    --danger-bg:  #ffebe9;
    --success-bg: #dafbe1;
    --warning-bg: #fff8c5;
  }

  /* ── Base ── */
  *, *::before, *::after { box-sizing: border-box; }
  html, body { height: 100%; }
  body {
    font-family: 'Inter', system-ui, -apple-system, sans-serif;
    font-size: 13px;
    line-height: 1.5;
    color: var(--text-1);
    background: var(--bg-base);
  }

  /* ── Navbar ── */
  .sn-nav {
    height: 52px;
    background: var(--bg-surface);
    border-bottom: 1px solid var(--border);
    display: flex;
    align-items: center;
    padding: 0 20px;
    gap: 16px;
    position: sticky;
    top: 0;
    z-index: 100;
  }
  .sn-brand {
    display: flex;
    align-items: center;
    gap: 8px;
    font-weight: 700;
    font-size: 14px;
    color: var(--text-1);
    text-decoration: none;
    letter-spacing: -0.2px;
  }
  .sn-brand i { color: var(--accent); font-size: 16px; }
  .sn-brand:hover { color: var(--text-1); }
  .sn-divider { width: 1px; height: 20px; background: var(--border); }
  .sn-nav-link {
    font-size: 12.5px;
    font-weight: 500;
    color: var(--text-2);
    text-decoration: none;
    padding: 4px 8px;
    border-radius: 4px;
    transition: background .15s, color .15s;
  }
  .sn-nav-link:hover { background: var(--bg-raised); color: var(--text-1); }
  .sn-nav-link.active { color: var(--text-1); background: var(--bg-raised); }
  .sn-spacer { flex: 1; }
  .sn-meta { font-size: 11.5px; color: var(--text-3); font-variant-numeric: tabular-nums; }
  .sn-btn {
    display: inline-flex;
    align-items: center;
    gap: 5px;
    font-size: 12px;
    font-weight: 500;
    padding: 4px 10px;
    border-radius: 4px;
    border: 1px solid var(--border);
    background: transparent;
    color: var(--text-2);
    cursor: pointer;
    text-decoration: none;
    transition: background .15s, color .15s;
    white-space: nowrap;
  }
  .sn-btn:hover { background: var(--bg-raised); color: var(--text-1); }
  .sn-btn-danger { border-color: transparent; color: var(--danger); }
  .sn-btn-danger:hover { background: var(--danger-bg); color: var(--danger); }

  /* ── Status pill ── */
  .status-pill {
    display: inline-flex;
    align-items: center;
    gap: 5px;
    font-size: 11px;
    font-weight: 600;
    letter-spacing: .4px;
    text-transform: uppercase;
    padding: 2px 8px;
    border-radius: 20px;
  }
  .status-dot {
    width: 6px; height: 6px;
    border-radius: 50%;
    flex-shrink: 0;
  }
  .status-clear { background: var(--success-bg); color: var(--success); }
  .status-clear .status-dot { background: var(--success); }
  .status-alert { background: var(--danger-bg); color: var(--danger); }
  .status-alert .status-dot { background: var(--danger); animation: blink 1s ease-in-out infinite; }
  @keyframes blink { 0%,100%{opacity:1} 50%{opacity:.35} }

  /* ── Page wrapper ── */
  .sn-page { padding: 20px 24px; }

  /* ── Metric tiles ── */
  .metric-row { display: grid; grid-template-columns: repeat(4, 1fr); gap: 12px; margin-bottom: 20px; }
  @media(max-width:900px) { .metric-row { grid-template-columns: repeat(2, 1fr); } }
  .metric-tile {
    background: var(--bg-surface);
    border: 1px solid var(--border);
    border-radius: 6px;
    padding: 14px 16px;
  }
  .metric-tile .mt-label {
    font-size: 11px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: .5px;
    color: var(--text-3);
    margin-bottom: 6px;
  }
  .metric-tile .mt-value {
    font-size: 24px;
    font-weight: 700;
    line-height: 1;
    color: var(--text-1);
    font-variant-numeric: tabular-nums;
  }
  .metric-tile .mt-value.danger { color: var(--danger); }
  .metric-tile .mt-value.success { color: var(--success); }
  .metric-tile .mt-sub {
    font-size: 11px;
    color: var(--text-3);
    margin-top: 4px;
  }

  /* ── Panel ── */
  .sn-panel {
    background: var(--bg-surface);
    border: 1px solid var(--border);
    border-radius: 6px;
    margin-bottom: 16px;
    overflow: hidden;
  }
  .sn-panel-header {
    display: flex;
    align-items: center;
    gap: 8px;
    padding: 10px 16px;
    border-bottom: 1px solid var(--border);
    background: var(--bg-raised);
  }
  .sn-panel-title {
    font-size: 11.5px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: .5px;
    color: var(--text-2);
  }
  .sn-panel-count {
    font-size: 11px;
    font-weight: 600;
    color: var(--text-3);
    background: var(--bg-base);
    border: 1px solid var(--border);
    padding: 1px 7px;
    border-radius: 20px;
  }
  .sn-panel-count.hot { color: var(--danger); border-color: var(--danger-bg); background: var(--danger-bg); }
  .sn-panel-actions { margin-left: auto; display: flex; gap: 6px; }
  .sn-panel-collapse-btn {
    background: none;
    border: none;
    color: var(--text-3);
    cursor: pointer;
    padding: 2px 4px;
    font-size: 11px;
  }

  /* ── Table ── */
  .sn-table { width: 100%; border-collapse: collapse; }
  .sn-table th {
    font-size: 11px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: .4px;
    color: var(--text-3);
    padding: 8px 12px;
    border-bottom: 1px solid var(--border);
    white-space: nowrap;
    background: var(--bg-raised);
  }
  .sn-table td {
    padding: 8px 12px;
    border-bottom: 1px solid var(--border-sub);
    color: var(--text-1);
    vertical-align: middle;
  }
  .sn-table tr:last-child td { border-bottom: none; }
  .sn-table tbody tr:hover td { background: var(--bg-raised); }
  .sn-table .empty td { color: var(--text-3); text-align: center; padding: 24px; font-style: italic; font-size: 12px; }
  .sn-table .mono { font-family: 'SF Mono','Fira Code',Menlo,monospace; font-size: 11.5px; }
  .sn-table .danger { color: var(--danger); }
  .sn-table .success { color: var(--success); }
  .sn-table .warning { color: var(--warning); }
  .sn-table .muted { color: var(--text-3); }

  /* ── Status badges ── */
  .tag {
    display: inline-flex;
    align-items: center;
    gap: 4px;
    font-size: 10.5px;
    font-weight: 600;
    letter-spacing: .3px;
    text-transform: uppercase;
    padding: 2px 7px;
    border-radius: 3px;
  }
  .tag-danger  { background: var(--danger-bg);  color: var(--danger); }
  .tag-success { background: var(--success-bg); color: var(--success); }
  .tag-warning { background: var(--warning-bg); color: var(--warning); }
  .tag-neutral { background: var(--bg-raised); color: var(--text-2); border: 1px solid var(--border); }
  .tag-dot { width: 5px; height: 5px; border-radius: 50%; background: currentColor; }

  /* ── Two-col layout ── */
  .sn-cols { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; margin-bottom: 16px; }
  @media(max-width:900px) { .sn-cols { grid-template-columns: 1fr; } }
  .sn-cols .sn-panel { margin-bottom: 0; }

  /* ── Insights ── */
  .insight-stats { display: grid; grid-template-columns: repeat(3, 1fr); gap: 0; border-bottom: 1px solid var(--border); }
  .insight-stat { padding: 12px 14px; border-right: 1px solid var(--border); }
  .insight-stat:last-child { border-right: none; }
  .insight-stat-val { font-size: 20px; font-weight: 700; line-height: 1; color: var(--text-1); font-variant-numeric: tabular-nums; }
  .insight-stat-lbl { font-size: 10.5px; text-transform: uppercase; letter-spacing: .4px; color: var(--text-3); margin-top: 3px; }
  .insight-body { padding: 12px 14px; }
  .insight-row { display: flex; align-items: baseline; gap: 6px; margin-bottom: 6px; font-size: 12px; }
  .insight-row .lbl { color: var(--text-3); min-width: 110px; }
  .insight-row .val { color: var(--text-1); font-weight: 500; }
  .phase-list { margin-top: 8px; }
  .phase-item { display: flex; align-items: center; gap: 8px; font-size: 11.5px; margin-bottom: 5px; color: var(--text-2); }
  .phase-time { color: var(--text-3); font-variant-numeric: tabular-nums; min-width: 60px; font-size: 11px; }
  .phase-icon { color: var(--accent); }

  /* ── Alert banner ── */
  .alert-banner {
    display: flex; align-items: center; gap: 10px;
    padding: 10px 16px;
    border-radius: 6px;
    font-size: 12.5px;
    font-weight: 500;
    margin-bottom: 16px;
  }
  .alert-banner.danger { background: var(--danger-bg); color: var(--danger); border: 1px solid rgba(248,81,73,.3); }
  .alert-banner.success { background: var(--success-bg); color: var(--success); border: 1px solid rgba(63,185,80,.3); }

  /* ── Session cards ── */
  .session-card {
    background: var(--bg-surface);
    border: 1px solid var(--border);
    border-radius: 6px;
    margin-bottom: 10px;
    overflow: hidden;
  }
  .session-card.current { border-color: var(--success); }
  .session-card-header {
    display: flex;
    align-items: center;
    gap: 10px;
    padding: 11px 16px;
    cursor: pointer;
    background: var(--bg-surface);
    transition: background .12s;
    user-select: none;
  }
  .session-card-header:hover { background: var(--bg-raised); }
  .session-card-body { border-top: 1px solid var(--border); display: none; }
  .session-card-body.open { display: block; }
  .session-chevron { color: var(--text-3); font-size: 11px; transition: transform .2s; }
  .session-chevron.open { transform: rotate(90deg); }
  .session-ts { font-family: 'SF Mono','Fira Code',monospace; font-size: 12.5px; font-weight: 600; color: var(--text-1); }
  .session-meta { display: flex; gap: 14px; margin-left: 4px; }
  .session-meta-item { display: flex; align-items: center; gap: 4px; font-size: 11.5px; color: var(--text-3); }
  .session-meta-item i { font-size: 11px; }

  /* ── Login ── */
  .login-wrap { min-height: 100vh; display: flex; align-items: center; justify-content: center; background: var(--bg-base); }
  .login-card { width: 100%; max-width: 360px; }
  .login-logo { display: flex; align-items: center; gap: 10px; justify-content: center; margin-bottom: 6px; }
  .login-logo-icon { font-size: 28px; color: var(--accent); }
  .login-logo-text { font-size: 18px; font-weight: 700; letter-spacing: -0.5px; color: var(--text-1); }
  .login-tagline { text-align: center; font-size: 12px; color: var(--text-3); margin-bottom: 24px; }
  .login-box {
    background: var(--bg-surface);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 28px 24px;
  }
  .sn-field { margin-bottom: 16px; }
  .sn-label { display: block; font-size: 12px; font-weight: 600; color: var(--text-2); margin-bottom: 5px; }
  .sn-input {
    width: 100%;
    padding: 7px 10px;
    font-size: 13px;
    font-family: inherit;
    background: var(--bg-base);
    border: 1px solid var(--border);
    border-radius: 5px;
    color: var(--text-1);
    outline: none;
    transition: border-color .15s;
  }
  .sn-input:focus { border-color: var(--accent); box-shadow: 0 0 0 2px rgba(88,166,255,.15); }
  .sn-submit {
    width: 100%;
    padding: 8px;
    font-size: 13px;
    font-weight: 600;
    font-family: inherit;
    background: var(--accent);
    border: none;
    border-radius: 5px;
    color: #fff;
    cursor: pointer;
    transition: opacity .15s;
    margin-top: 4px;
  }
  .sn-submit:hover { opacity: .9; }
  .login-footer { text-align: center; margin-top: 16px; }
  .err-alert {
    display: flex; align-items: center; gap: 7px;
    padding: 8px 12px;
    border-radius: 5px;
    font-size: 12px;
    font-weight: 500;
    background: var(--danger-bg);
    color: var(--danger);
    border: 1px solid rgba(248,81,73,.25);
    margin-bottom: 16px;
  }
</style>"""


# ---------------------------------------------------------------------------
# Login page
# ---------------------------------------------------------------------------

_LOGIN_HTML = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>LLM Sentinel — Sign In</title>
{_THEME_SCRIPT}
<link rel="stylesheet" href="{_BI}">
{_SHARED_STYLE}
</head>
<body>
<div class="login-wrap">
  <div class="login-card">
    <div class="login-logo">
      <i class="bi bi-shield-lock-fill login-logo-icon"></i>
      <span class="login-logo-text">LLM Sentinel</span>
    </div>
    <div class="login-tagline">Network monitoring — sign in to continue</div>
    <div class="login-box">
      {{ERROR}}
      <form method="post" action="/api/login">
        <div class="sn-field">
          <label class="sn-label">Username</label>
          <input type="text" name="username" class="sn-input"
                 autocomplete="username" autofocus required>
        </div>
        <div class="sn-field">
          <label class="sn-label">Password</label>
          <input type="password" name="password" class="sn-input"
                 autocomplete="current-password" required>
        </div>
        <button type="submit" class="sn-submit">Sign in</button>
      </form>
    </div>
    <div class="login-footer">
      <button class="sn-btn" onclick="toggleTheme()" style="margin:auto">
        <i id="theme-icon" class="bi bi-sun"></i> Toggle theme
      </button>
    </div>
  </div>
</div>
<script>
{_THEME_TOGGLE_JS}
(function(){{
  var icon = document.getElementById('theme-icon');
  if(icon) icon.className = document.documentElement.getAttribute('data-bs-theme')==='dark'
    ? 'bi bi-sun' : 'bi bi-moon-stars-fill';
}})();
</script>
</body>
</html>"""

# Route handler calls: _LOGIN_HTML.replace("{{ERROR}}", error_html)


# ---------------------------------------------------------------------------
# Main dashboard
# ---------------------------------------------------------------------------

_HTML = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>LLM Sentinel</title>
{_THEME_SCRIPT}
<link rel="stylesheet" href="{_BI}">
{_SHARED_STYLE}
</head>
<body>

<nav class="sn-nav">
  <span class="sn-brand">
    <i class="bi bi-shield-lock-fill"></i>LLM Sentinel
  </span>
  <div class="sn-divider"></div>
  <span class="sn-nav-link active">Dashboard</span>
  <a href="/sessions" class="sn-nav-link">Session History</a>
  <div class="sn-spacer"></div>
  <span id="status-pill" class="status-pill status-clear">
    <span class="status-dot"></span>All clear
  </span>
  <span id="meta" class="sn-meta" style="min-width:130px;text-align:right">initializing…</span>
  <button class="sn-btn" onclick="toggleTheme()" title="Toggle theme">
    <i id="theme-icon" class="bi bi-sun"></i>
  </button>
  <a href="/logout" class="sn-btn sn-btn-danger">
    <i class="bi bi-box-arrow-right"></i>Sign out
  </a>
</nav>

<div class="sn-page">

  <!-- Metric tiles -->
  <div class="metric-row">
    <div class="metric-tile">
      <div class="mt-label">Status</div>
      <div class="mt-value success" id="tile-status">Clear</div>
      <div class="mt-sub" id="tile-scan">scan —</div>
    </div>
    <div class="metric-tile">
      <div class="mt-label">LLM Processes</div>
      <div class="mt-value" id="tile-procs">—</div>
      <div class="mt-sub">running</div>
    </div>
    <div class="metric-tile">
      <div class="mt-label">External Connections</div>
      <div class="mt-value" id="tile-ext">—</div>
      <div class="mt-sub" id="tile-ext-sub">none detected</div>
    </div>
    <div class="metric-tile">
      <div class="mt-label">Session Age</div>
      <div class="mt-value" id="tile-age">—</div>
      <div class="mt-sub" id="tile-dests">— destinations</div>
    </div>
  </div>

  <!-- External connections alert -->
  <div id="ext-alert" style="display:none" class="alert-banner danger">
    <i class="bi bi-exclamation-triangle-fill"></i>
    <span id="ext-alert-text">External connections detected</span>
  </div>

  <!-- Processes -->
  <div class="sn-panel">
    <div class="sn-panel-header">
      <span class="sn-panel-title">LLM Processes</span>
      <span class="sn-panel-count" id="proc-count">0</span>
    </div>
    <div class="table-responsive">
      <table class="sn-table">
        <thead><tr>
          <th>PID</th><th>Process</th><th>Status</th>
          <th>CPU %</th><th>Mem MB</th><th>Ext Conns</th><th>Command</th>
        </tr></thead>
        <tbody id="proc-body">
          <tr class="empty"><td colspan="7">No LLM processes detected</td></tr>
        </tbody>
      </table>
    </div>
  </div>

  <!-- External connections -->
  <div class="sn-panel" id="ext-panel">
    <div class="sn-panel-header">
      <span class="sn-panel-title">External Connections</span>
      <span class="sn-panel-count" id="ext-count">0</span>
    </div>
    <div class="table-responsive">
      <table class="sn-table">
        <thead><tr>
          <th>Process</th><th>Remote IP</th><th>Hostname</th>
          <th>Port</th><th>Protocol</th><th>Conns</th>
        </tr></thead>
        <tbody id="ext-body">
          <tr class="empty"><td colspan="6">No external connections</td></tr>
        </tbody>
      </table>
    </div>
  </div>

  <!-- History + Insights -->
  <div class="sn-cols">
    <div class="sn-panel">
      <div class="sn-panel-header">
        <span class="sn-panel-title">Connection Events</span>
        <span class="sn-panel-count" id="hist-count">0</span>
      </div>
      <div class="table-responsive">
        <table class="sn-table">
          <thead><tr>
            <th>Time</th><th>Event</th><th>Process</th>
            <th>Destination</th><th>Port</th><th>Duration</th>
          </tr></thead>
          <tbody id="hist-body">
            <tr class="empty"><td colspan="6">No events yet</td></tr>
          </tbody>
        </table>
      </div>
    </div>

    <div class="sn-panel">
      <div class="sn-panel-header">
        <span class="sn-panel-title">Session Insights</span>
      </div>
      <div id="insights-stats" class="insight-stats" style="display:none">
        <div class="insight-stat">
          <div class="insight-stat-val" id="ins-age">—</div>
          <div class="insight-stat-lbl">Session age</div>
        </div>
        <div class="insight-stat">
          <div class="insight-stat-val" id="ins-dests">—</div>
          <div class="insight-stat-lbl">Destinations</div>
        </div>
        <div class="insight-stat">
          <div class="insight-stat-val" id="ins-total">—</div>
          <div class="insight-stat-lbl">Connections</div>
        </div>
      </div>
      <div class="insight-body" id="insights-body">
        <div style="color:var(--text-3);font-style:italic;font-size:12px">Waiting for data…</div>
      </div>
    </div>
  </div>

  <!-- Local connections (collapsible) -->
  <div class="sn-panel">
    <div class="sn-panel-header" style="cursor:pointer" onclick="toggleLocal()">
      <i class="bi bi-chevron-right session-chevron" id="local-chevron"></i>
      <span class="sn-panel-title">Local Connections</span>
      <span class="sn-panel-count" id="local-count">0</span>
    </div>
    <div id="local-body-wrap" style="display:none">
      <div class="table-responsive">
        <table class="sn-table">
          <thead><tr><th>PID</th><th>Process</th><th>Local Addr</th><th>Remote Addr</th><th>State</th></tr></thead>
          <tbody id="local-body">
            <tr class="empty"><td colspan="5">No local connections</td></tr>
          </tbody>
        </table>
      </div>
    </div>
  </div>

</div><!-- /page -->

<script>
{_THEME_TOGGLE_JS}

(function(){{
  var icon = document.getElementById('theme-icon');
  if(icon) icon.className = document.documentElement.getAttribute('data-bs-theme')==='dark'
    ? 'bi bi-sun' : 'bi bi-moon-stars-fill';
}})();

function esc(s) {{
  return String(s).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
}}

function evTag(ev) {{
  if (ev === 'open' || ev === 'opened')
    return '<span class="tag tag-danger"><span class="tag-dot"></span>OPEN</span>';
  if (ev === 'closed')
    return '<span class="tag tag-success"><span class="tag-dot"></span>CLOSED</span>';
  return '<span class="tag tag-neutral">' + esc(ev.toUpperCase()) + '</span>';
}}

function toggleLocal() {{
  var wrap = document.getElementById('local-body-wrap');
  var ch   = document.getElementById('local-chevron');
  var open = wrap.style.display !== 'none';
  wrap.style.display = open ? 'none' : 'block';
  ch.className = open ? 'bi bi-chevron-right session-chevron' : 'bi bi-chevron-right session-chevron open';
}}

async function refresh() {{
  let data;
  try {{
    const r = await fetch('/api/state');
    if(r.status===401){{ location='/login'; return; }}
    data = await r.json();
  }} catch(e) {{
    document.getElementById('meta').textContent = 'connection lost';
    return;
  }}

  const hasAlert = data.alert;
  document.title = (hasAlert ? '⚠ ' : '') + 'LLM Sentinel';

  const pill = document.getElementById('status-pill');
  if (hasAlert) {{
    pill.className = 'status-pill status-alert';
    pill.innerHTML = '<span class="status-dot"></span>Alert';
  }} else {{
    pill.className = 'status-pill status-clear';
    pill.innerHTML = '<span class="status-dot"></span>All clear';
  }}

  document.getElementById('meta').textContent = 'scan #' + (data.scan||0) + ' · ' + (data.updated_at||'');

  const ext = data.external || [];
  const procs = data.processes || [];

  // Tiles
  const tileSt = document.getElementById('tile-status');
  tileSt.textContent = hasAlert ? 'Alert' : 'Clear';
  tileSt.className = 'mt-value ' + (hasAlert ? 'danger' : 'success');
  document.getElementById('tile-scan').textContent = 'scan #' + (data.scan||0);
  document.getElementById('tile-procs').textContent = procs.length;
  const tileExt = document.getElementById('tile-ext');
  tileExt.textContent = ext.length;
  tileExt.className = 'mt-value ' + (ext.length > 0 ? 'danger' : '');
  document.getElementById('tile-ext-sub').textContent = ext.length > 0
    ? ext.length + ' active' : 'none detected';

  const ins = data.insights || {{}};
  document.getElementById('tile-age').textContent = ins.session_age || '—';
  document.getElementById('tile-dests').textContent = (ins.unique_destinations || 0) + ' destinations';

  // Alert banner
  const banner = document.getElementById('ext-alert');
  if (hasAlert) {{
    banner.style.display = 'flex';
    document.getElementById('ext-alert-text').textContent =
      ext.length + ' active external connection' + (ext.length!==1?'s':'') + ' detected';
  }} else {{
    banner.style.display = 'none';
  }}

  // Processes
  document.getElementById('proc-count').textContent = procs.length;
  document.getElementById('proc-body').innerHTML = procs.length === 0
    ? '<tr class="empty"><td colspan="7">No LLM processes detected</td></tr>'
    : procs.map(p =>
        '<tr>' +
        '<td class="mono muted">' + esc(p.pid) + '</td>' +
        '<td style="font-weight:600">' + esc(p.name) + '</td>' +
        '<td><span class="tag tag-success">' + esc(p.status) + '</span></td>' +
        '<td class="' + (p.cpu > 100 ? 'warning' : 'muted') + '">' + p.cpu + '%</td>' +
        '<td class="muted">' + p.mem_mb + ' MB</td>' +
        '<td>' + (p.ext_conns > 0
          ? '<span class="tag tag-danger">' + p.ext_conns + ' ext</span>'
          : '<span class="tag tag-neutral">0</span>') + '</td>' +
        '<td class="mono muted" style="max-width:260px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap" title="' + esc(p.cmd) + '">' + esc(p.cmd) + '</td>' +
        '</tr>').join('');

  // External connections
  const extCount = document.getElementById('ext-count');
  extCount.textContent = ext.length;
  extCount.className = 'sn-panel-count' + (ext.length > 0 ? ' hot' : '');
  document.getElementById('ext-body').innerHTML = ext.length === 0
    ? '<tr class="empty"><td colspan="6">No external connections</td></tr>'
    : ext.map(c =>
        '<tr>' +
        '<td style="font-weight:600">' + esc(c.name) + ' <span class="mono muted">' + esc(c.pid) + '</span></td>' +
        '<td class="mono danger">' + esc(c.remote_ip) + '</td>' +
        '<td class="warning">' + esc(c.hostname) + '</td>' +
        '<td class="mono danger" style="font-weight:600">' + esc(c.port) + '</td>' +
        '<td class="muted">' + esc(c.proto) + '</td>' +
        '<td>' + (c.count > 1
          ? '<span class="tag tag-danger">×' + c.count + '</span>'
          : '<span class="muted">×1</span>') + '</td>' +
        '</tr>').join('');

  // History
  const hist = data.history || [];
  document.getElementById('hist-count').textContent = hist.length;
  document.getElementById('hist-body').innerHTML = hist.length === 0
    ? '<tr class="empty"><td colspan="6">No events recorded yet</td></tr>'
    : hist.map(h =>
        '<tr>' +
        '<td class="mono muted">' + esc(h.time) + '</td>' +
        '<td>' + evTag(h.event) + '</td>' +
        '<td>' + esc(h.process) + '</td>' +
        '<td class="warning">' + esc(h.hostname || h.ip) + '</td>' +
        '<td class="mono muted">' + esc(h.port) + '</td>' +
        '<td class="' + (h.event==='closed' ? 'success' : 'warning') + '">' + esc(h.duration) + '</td>' +
        '</tr>').join('');

  // Insights
  if (ins.session_age) {{
    document.getElementById('insights-stats').style.display = 'grid';
    document.getElementById('ins-age').textContent   = ins.session_age;
    document.getElementById('ins-dests').textContent = ins.unique_destinations || 0;
    document.getElementById('ins-total').textContent = ins.total_connections || 0;

    const icons = {{
      startup:   '<i class="bi bi-arrow-up-right-circle phase-icon"></i>',
      download:  '<i class="bi bi-download phase-icon"></i>',
      inference: '<i class="bi bi-cpu phase-icon"></i>',
    }};
    const phases = (ins.phases || []).map(p =>
      '<div class="phase-item">' +
      '<span class="phase-time">' + (p.ts ? new Date(p.ts*1000).toLocaleTimeString() : '') + '</span>' +
      (icons[p.name] || '<span class="phase-icon">·</span>') +
      '<span>' + esc(p.label) + (p.count ? ' <span style="color:var(--text-3)">(' + p.count + ')</span>' : '') + '</span>' +
      '</div>'
    ).join('');
    const longest = ins.longest
      ? esc(ins.longest.duration) + ' → ' + esc(ins.longest.host) : '—';

    document.getElementById('insights-body').innerHTML =
      (ins.most_contacted
        ? '<div class="insight-row"><span class="lbl">Most contacted</span><span class="val warning">' + esc(ins.most_contacted) + '</span></div>' : '') +
      '<div class="insight-row"><span class="lbl">Longest connection</span><span class="val">' + longest + '</span></div>' +
      (phases ? '<div class="phase-list">' + phases + '</div>' : '') +
      (hasAlert
        ? '<div style="margin-top:10px" class="alert-banner danger" style="margin:10px 0 0"><i class="bi bi-exclamation-triangle-fill"></i> Active external connections</div>'
        : '<div style="margin-top:10px" class="alert-banner success"><i class="bi bi-check-circle-fill"></i> No external connections</div>');
  }}

  // Local
  const local = data.local || [];
  document.getElementById('local-count').textContent = local.length;
  document.getElementById('local-body').innerHTML = local.length === 0
    ? '<tr class="empty"><td colspan="5">No local connections</td></tr>'
    : local.map(c =>
        '<tr>' +
        '<td class="mono muted">' + esc(c.pid) + '</td>' +
        '<td>' + esc(c.name) + '</td>' +
        '<td class="mono muted">' + esc(c.local) + '</td>' +
        '<td class="mono muted">' + esc(c.remote) + '</td>' +
        '<td class="muted">' + esc(c.status) + '</td>' +
        '</tr>').join('');
}}

refresh();
setInterval(refresh, 3000);
</script>
</body>
</html>"""


# ---------------------------------------------------------------------------
# Sessions history page
# ---------------------------------------------------------------------------

_SESSIONS_HTML = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>LLM Sentinel — Session History</title>
{_THEME_SCRIPT}
<link rel="stylesheet" href="{_BI}">
{_SHARED_STYLE}
</head>
<body>

<nav class="sn-nav">
  <a href="/" class="sn-brand">
    <i class="bi bi-shield-lock-fill"></i>LLM Sentinel
  </a>
  <div class="sn-divider"></div>
  <a href="/" class="sn-nav-link">Dashboard</a>
  <span class="sn-nav-link active">Session History</span>
  <div class="sn-spacer"></div>
  <button class="sn-btn" onclick="toggleTheme()">
    <i id="theme-icon" class="bi bi-sun"></i>
  </button>
  <a href="/logout" class="sn-btn sn-btn-danger">
    <i class="bi bi-box-arrow-right"></i>Sign out
  </a>
</nav>

<div class="sn-page">
  <div style="display:flex;align-items:center;gap:10px;margin-bottom:18px">
    <span style="font-size:13px;font-weight:600;color:var(--text-2);text-transform:uppercase;letter-spacing:.5px">Session History</span>
    <span class="sn-panel-count" id="session-count">…</span>
  </div>
  <div id="sessions-list">
    <div style="color:var(--text-3);text-align:center;padding:48px 0;font-style:italic;font-size:13px">Loading…</div>
  </div>
</div>

<script>
{_THEME_TOGGLE_JS}

(function(){{
  var icon = document.getElementById('theme-icon');
  if(icon) icon.className = document.documentElement.getAttribute('data-bs-theme')==='dark'
    ? 'bi bi-sun' : 'bi bi-moon-stars-fill';
}})();

function esc(s) {{
  return String(s).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
}}

function evTag(ev) {{
  if (ev === 'open' || ev === 'opened')
    return '<span class="tag tag-danger"><span class="tag-dot"></span>OPEN</span>';
  if (ev === 'closed')
    return '<span class="tag tag-success"><span class="tag-dot"></span>CLOSED</span>';
  return '<span class="tag tag-neutral">' + esc(ev.toUpperCase()) + '</span>';
}}

const _loaded = {{}};

function toggle(el) {{
  const i   = el.dataset.idx;
  const sid = el.dataset.sid;
  const body = document.getElementById('sb-' + i);
  const ch   = document.getElementById('si-' + i);
  const isOpen = body.classList.contains('open');

  if (!_loaded[i]) {{
    _loaded[i] = true;
    body.innerHTML = '<div style="padding:16px;color:var(--text-3);font-style:italic;font-size:12px">Loading…</div>';
    fetch('/api/sessions/' + encodeURIComponent(sid))
      .then(r => {{ if(r.status===401){{location='/login';}} return r.json(); }})
      .then(events => {{
        if (!Array.isArray(events) || !events.length) {{
          body.innerHTML = '<div style="padding:16px;color:var(--text-3);font-style:italic;font-size:12px">No events recorded.</div>';
          return;
        }}
        const rows = events.map(e =>
          '<tr>' +
          '<td class="mono muted">' + esc(e.time) + '</td>' +
          '<td>' + evTag(e.event) + '</td>' +
          '<td>' + esc(e.process) + '</td>' +
          '<td class="warning">' + esc(e.hostname || e.ip) + '</td>' +
          '<td class="mono muted">' + esc(e.ip) + '</td>' +
          '<td class="mono muted">' + esc(e.port) + '</td>' +
          '<td class="' + (e.event==='closed' ? 'success' : 'warning') + '">' + esc(e.duration) + '</td>' +
          '</tr>'
        ).join('');
        body.innerHTML =
          '<div class="table-responsive"><table class="sn-table">' +
          '<thead><tr><th>Time</th><th>Event</th><th>Process</th><th>Destination</th><th>IP</th><th>Port</th><th>Duration</th></tr></thead>' +
          '<tbody>' + rows + '</tbody></table></div>';
      }}).catch(() => {{
        body.innerHTML = '<div style="padding:16px;color:var(--danger);font-size:12px">Failed to load events.</div>';
      }});
  }}

  if (isOpen) {{
    body.classList.remove('open');
    ch.classList.remove('open');
  }} else {{
    body.classList.add('open');
    ch.classList.add('open');
  }}
}}

async function loadSessions() {{
  let sessions;
  try {{
    const r = await fetch('/api/sessions');
    if (r.status === 401) {{ location = '/login'; return; }}
    sessions = await r.json();
  }} catch(e) {{ sessions = []; }}

  if (!Array.isArray(sessions)) sessions = [];
  document.getElementById('session-count').textContent = sessions.length;
  const el = document.getElementById('sessions-list');

  if (!sessions.length) {{
    el.innerHTML = '<div style="color:var(--text-3);text-align:center;padding:48px 0;font-style:italic;font-size:13px">No sessions recorded yet.</div>';
    return;
  }}

  el.innerHTML = sessions.map((s, i) =>
    '<div class="session-card' + (s.is_current ? ' current' : '') + '">' +
      '<div class="session-card-header" data-idx="' + i + '" data-sid="' + esc(s.id) + '" onclick="toggle(this)">' +
        '<i class="bi bi-chevron-right session-chevron" id="si-' + i + '"></i>' +
        '<span class="session-ts">' + esc(s.started) + '</span>' +
        (s.is_current ? '<span class="tag tag-success" style="margin-left:4px">LIVE</span>' : '') +
        '<div class="session-meta">' +
          '<span class="session-meta-item"><i class="bi bi-clock"></i>' + esc(s.duration) + '</span>' +
          '<span class="session-meta-item"><i class="bi bi-arrow-left-right"></i>' + s.conn_count + ' conn' + (s.conn_count!==1?'s':'') + '</span>' +
          '<span class="session-meta-item"><i class="bi bi-globe"></i>' + s.unique_ips + ' dest' + (s.unique_ips!==1?'s':'') + '</span>' +
          (s.processes ? '<span class="session-meta-item"><i class="bi bi-cpu"></i>' + esc(s.processes) + '</span>' : '') +
        '</div>' +
      '</div>' +
      '<div class="session-card-body" id="sb-' + i + '"></div>' +
    '</div>'
  ).join('');
}}

loadSessions();
</script>
</body>
</html>"""
