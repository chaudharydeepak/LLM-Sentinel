"""
Lightweight web dashboard — FastAPI JSON API + embedded single-page HTML.
Runs in a background thread; reads from a shared state dict updated by the sentinel.
"""

import threading
import time
from typing import Any

import uvicorn
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse

# ---------------------------------------------------------------------------
# Shared state (written by sentinel tick, read by API)
# ---------------------------------------------------------------------------

_state: dict[str, Any] = {}
_lock = threading.Lock()


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
        })


def _fmt_age(s: float) -> str:
    if s < 60:
        return f"{s:.0f}s"
    if s < 3600:
        return f"{s/60:.1f}m"
    return f"{s/3600:.1f}h"


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(title="LLM Sentinel")


@app.get("/api/state")
def get_state():
    with _lock:
        return JSONResponse(_state or {"scan": 0, "alert": False})


@app.get("/", response_class=HTMLResponse)
def index():
    return HTMLResponse(_HTML)


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
# Embedded HTML (single file, no build step)
# ---------------------------------------------------------------------------

_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>LLM Sentinel</title>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { background: #0d1117; color: #c9d1d9; font-family: 'SF Mono', 'Fira Code', monospace; font-size: 13px; }
  header { background: #161b22; border-bottom: 1px solid #30363d; padding: 12px 20px; display: flex; align-items: center; gap: 16px; position: sticky; top: 0; z-index: 10; }
  header h1 { font-size: 15px; font-weight: 600; color: #f0f6fc; }
  .badge { padding: 3px 10px; border-radius: 12px; font-size: 11px; font-weight: 700; letter-spacing: .5px; }
  .badge.ok  { background: #1a4731; color: #3fb950; }
  .badge.alert { background: #4a1010; color: #f85149; animation: pulse 1s ease-in-out infinite; }
  @keyframes pulse { 0%,100%{opacity:1} 50%{opacity:.6} }
  .meta { color: #8b949e; font-size: 11px; margin-left: auto; }
  main { display: grid; grid-template-columns: 1fr 1fr; gap: 12px; padding: 16px; }
  .full { grid-column: 1 / -1; }
  section { background: #161b22; border: 1px solid #30363d; border-radius: 8px; overflow: hidden; }
  section.danger { border-color: #6e2020; }
  section h2 { font-size: 12px; font-weight: 600; padding: 8px 14px; border-bottom: 1px solid #30363d; color: #8b949e; text-transform: uppercase; letter-spacing: .8px; display: flex; align-items: center; gap: 8px; }
  section h2 .count { background: #21262d; border-radius: 10px; padding: 1px 7px; font-size: 11px; }
  section h2 .count.red { background: #4a1010; color: #f85149; }
  table { width: 100%; border-collapse: collapse; }
  th { text-align: left; padding: 6px 14px; color: #8b949e; font-size: 11px; font-weight: 500; border-bottom: 1px solid #21262d; }
  td { padding: 6px 14px; border-bottom: 1px solid #1c2128; vertical-align: top; }
  tr:last-child td { border-bottom: none; }
  tr:hover td { background: #1c2128; }
  .empty { padding: 12px 14px; color: #484f58; font-style: italic; }
  .ext  { color: #f85149; font-weight: 600; }
  .loc  { color: #3fb950; }
  .dim  { color: #8b949e; }
  .yellow { color: #e3b341; }
  .badge-event { display: inline-block; padding: 1px 7px; border-radius: 4px; font-size: 10px; font-weight: 700; }
  .ev-opened { background:#4a1010; color:#f85149; }
  .ev-closed { background:#1a4731; color:#3fb950; }
  .ev-active { background:#2d2a10; color:#e3b341; }
  .insight-grid { display: grid; grid-template-columns: repeat(3, 1fr); gap: 1px; background: #21262d; }
  .insight-cell { background: #161b22; padding: 12px 16px; }
  .insight-cell .val { font-size: 22px; font-weight: 700; color: #f0f6fc; margin-bottom: 2px; }
  .insight-cell .lbl { font-size: 11px; color: #8b949e; }
  .phases { padding: 12px 14px; display: flex; flex-direction: column; gap: 6px; }
  .phase { display: flex; gap: 10px; align-items: flex-start; }
  .phase .time { color: #8b949e; min-width: 70px; }
  .phase .icon { color: #58a6ff; }
  .phase .label { color: #c9d1d9; }
  .now-banner { padding: 8px 14px; font-size: 12px; border-top: 1px solid #30363d; }
  .now-clear { color: #3fb950; }
  .now-alert { color: #f85149; }
  .cpu-high { color: #f0883e; font-weight: 600; }
  .mem { color: #58a6ff; }
</style>
</head>
<body>
<header>
  <h1>LLM Sentinel</h1>
  <span id="status-badge" class="badge ok">ALL CLEAR</span>
  <span id="meta" class="meta">loading…</span>
</header>
<main>

  <!-- Processes -->
  <section class="full" id="sec-processes">
    <h2>LLM Processes <span class="count" id="proc-count">0</span></h2>
    <table>
      <thead><tr>
        <th>PID</th><th>Name</th><th>Status</th><th>CPU%</th><th>Mem (MB)</th><th>Ext Conns</th><th>Command</th>
      </tr></thead>
      <tbody id="proc-body"><tr><td colspan="7" class="empty">No LLM processes detected</td></tr></tbody>
    </table>
  </section>

  <!-- External connections -->
  <section class="full" id="sec-external">
    <h2>External Connections <span class="count red" id="ext-count">0</span></h2>
    <table>
      <thead><tr>
        <th>Process</th><th>Remote IP</th><th>Hostname / Org</th><th>Port</th><th>Proto</th><th>Conns</th>
      </tr></thead>
      <tbody id="ext-body"><tr><td colspan="6" class="empty">None</td></tr></tbody>
    </table>
  </section>

  <!-- Session history -->
  <section style="grid-column:1/2">
    <h2>Session History <span class="count" id="hist-count">0</span></h2>
    <table>
      <thead><tr><th>Time</th><th>Event</th><th>Process</th><th>Destination</th><th>Port</th><th>Duration</th></tr></thead>
      <tbody id="hist-body"><tr><td colspan="6" class="empty">No events yet</td></tr></tbody>
    </table>
  </section>

  <!-- Insights -->
  <section style="grid-column:2/3">
    <h2>Insights</h2>
    <div id="insights-body">
      <div class="empty" style="padding:12px 14px">Waiting for data…</div>
    </div>
  </section>

  <!-- Local connections -->
  <section class="full" id="sec-local">
    <h2>Local Connections <span class="count" id="local-count">0</span></h2>
    <table>
      <thead><tr><th>PID</th><th>Process</th><th>Local</th><th>Remote</th><th>Status</th></tr></thead>
      <tbody id="local-body"><tr><td colspan="5" class="empty">No local connections</td></tr></tbody>
    </table>
  </section>

</main>
<script>
const $ = id => document.getElementById(id);

function esc(s) {
  return String(s).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
}

async function refresh() {
  let data;
  try { data = await fetch('/api/state').then(r => r.json()); }
  catch(e) { $('meta').textContent = 'connection lost…'; return; }

  const badge = $('status-badge');
  if (data.alert) {
    badge.textContent = 'ALERT'; badge.className = 'badge alert';
    document.title = '🔴 LLM Sentinel';
  } else {
    badge.textContent = 'ALL CLEAR'; badge.className = 'badge ok';
    document.title = '🟢 LLM Sentinel';
  }
  $('meta').textContent = `scan #${data.scan} · updated ${data.updated_at}`;

  // Processes
  const procs = data.processes || [];
  $('proc-count').textContent = procs.length;
  $('proc-body').innerHTML = procs.length === 0
    ? '<tr><td colspan="7" class="empty">No LLM processes detected</td></tr>'
    : procs.map(p => `<tr>
        <td class="dim">${esc(p.pid)}</td>
        <td><strong>${esc(p.name)}</strong></td>
        <td class="loc">${esc(p.status)}</td>
        <td class="${p.cpu > 100 ? 'cpu-high' : ''}">${p.cpu}</td>
        <td class="mem">${p.mem_mb}</td>
        <td class="${p.ext_conns > 0 ? 'ext' : 'loc'}">${p.ext_conns > 0 ? '⚠ ' + p.ext_conns : '✓ 0'}</td>
        <td class="dim" style="font-size:11px">${esc(p.cmd)}</td>
      </tr>`).join('');

  // External connections
  const ext = data.external || [];
  $('ext-count').textContent = ext.length;
  $('ext-count').className = 'count' + (ext.length > 0 ? ' red' : '');
  $('sec-external').className = 'full' + (ext.length > 0 ? ' danger' : '');
  $('ext-body').innerHTML = ext.length === 0
    ? '<tr><td colspan="6" class="empty">None — no external connections</td></tr>'
    : ext.map(c => `<tr>
        <td><strong>${esc(c.name)}</strong> <span class="dim">${esc(c.pid)}</span></td>
        <td class="ext">${esc(c.remote_ip)}</td>
        <td class="yellow">${esc(c.hostname)}</td>
        <td class="ext"><strong>${esc(c.port)}</strong></td>
        <td class="dim">${esc(c.proto)}</td>
        <td>${c.count > 1 ? '<strong class="ext">×' + c.count + '</strong>' : '×1'}</td>
      </tr>`).join('');

  // History
  const hist = data.history || [];
  $('hist-count').textContent = hist.length;
  $('hist-body').innerHTML = hist.length === 0
    ? '<tr><td colspan="6" class="empty">No events yet</td></tr>'
    : hist.map(h => {
        const cls = {opened:'ev-opened', open:'ev-active', closed:'ev-closed'}[h.event] || 'ev-active';
        return `<tr>
          <td class="dim">${esc(h.time)}</td>
          <td><span class="badge-event ${cls}">${esc(h.event).toUpperCase()}</span></td>
          <td>${esc(h.process)}</td>
          <td class="yellow">${esc(h.hostname)}</td>
          <td class="dim">${esc(h.port)}</td>
          <td class="${h.event==='closed'?'loc':'yellow'}">${esc(h.duration)}</td>
        </tr>`;
      }).join('');

  // Insights
  const ins = data.insights || {};
  if (!ins.session_age) {
    $('insights-body').innerHTML = '<div class="empty" style="padding:12px 14px">Waiting for data…</div>';
  } else {
    const phases = (ins.phases || []).map(p => {
      const icons = {startup:'→', download:'↓', inference:'⚙'};
      return `<div class="phase">
        <span class="time">${p.ts ? new Date(p.ts*1000).toLocaleTimeString() : ''}</span>
        <span class="icon">${icons[p.name]||'•'}</span>
        <span class="label">${esc(p.label)}${p.count ? ' <span class="dim">('+p.count+' conns)</span>' : ''}</span>
      </div>`;
    }).join('');

    const longest = ins.longest
      ? `<strong>${esc(ins.longest.duration)}</strong> → ${esc(ins.longest.host)}`
      : '—';

    let nowHtml = '';
    if (data.alert) {
      nowHtml = `<div class="now-banner now-alert">⚠ Active external connections</div>`;
    } else {
      nowHtml = `<div class="now-banner now-clear">✓ No external connections right now</div>`;
    }

    $('insights-body').innerHTML = `
      <div class="insight-grid">
        <div class="insight-cell"><div class="val">${esc(ins.session_age)}</div><div class="lbl">Session age</div></div>
        <div class="insight-cell"><div class="val">${esc(ins.unique_destinations)}</div><div class="lbl">Unique destinations</div></div>
        <div class="insight-cell"><div class="val">${esc(ins.total_connections)}</div><div class="lbl">Connections logged</div></div>
      </div>
      ${ins.most_contacted ? `<div style="padding:8px 14px;color:#8b949e;font-size:11px">Most contacted: <span class="yellow">${esc(ins.most_contacted)}</span>  &nbsp; Longest: ${longest}</div>` : ''}
      ${phases ? '<div class="phases">' + phases + '</div>' : ''}
      ${nowHtml}
    `;
  }

  // Local connections
  const local = data.local || [];
  $('local-count').textContent = local.length;
  $('local-body').innerHTML = local.length === 0
    ? '<tr><td colspan="5" class="empty">No local connections</td></tr>'
    : local.map(c => `<tr>
        <td class="dim">${esc(c.pid)}</td>
        <td>${esc(c.name)}</td>
        <td class="dim">${esc(c.local)}</td>
        <td class="dim">${esc(c.remote)}</td>
        <td class="dim">${esc(c.status)}</td>
      </tr>`).join('');
}

refresh();
setInterval(refresh, 3000);
</script>
</body>
</html>
"""
