"""
Live terminal dashboard using Rich.

Layout:
  ┌─ Header ──────────────────────────────┐
  ├─ LLM Processes ───────────────────────┤
  ├─ External Connections (active) ───────┤
  ├─ Local Connections ───────────────────┤
  ├─ Session History (log timeline) ──────┤
  └─ Insights ────────────────────────────┘
"""

import time

from rich.console import Console
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich import box

from .alerts import AlertManager
from .process_monitor import LLMProcess
from .resolver import hostname
from .session_log import SessionLog, ConnectionEvent

console = Console()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fmt_ts(ts: float) -> str:
    return time.strftime("%H:%M:%S", time.localtime(ts))

def _fmt_dur(s: float | None) -> str:
    if s is None:
        return "active"
    if s < 60:
        return f"{s:.0f}s"
    if s < 3600:
        return f"{s/60:.1f}m"
    return f"{s/3600:.1f}h"

def _short_host(ip: str, host: str) -> str:
    if host == ip:
        return ip
    parts = host.split(".")
    return ".".join(parts[-3:]) if len(parts) > 3 else host


# ---------------------------------------------------------------------------
# Process table
# ---------------------------------------------------------------------------

def _process_table(processes: list[LLMProcess]) -> Table:
    table = Table(box=box.SIMPLE_HEAVY, header_style="bold cyan", expand=True)
    table.add_column("PID", style="dim", width=7)
    table.add_column("Name", min_width=14)
    table.add_column("Status", width=10)
    table.add_column("CPU%", width=6)
    table.add_column("Mem (MB)", width=9)
    table.add_column("Ext. Conns", width=11)
    table.add_column("Path / Command", style="dim")

    if not processes:
        table.add_row("-", "[dim]No LLM processes detected[/dim]", "", "", "", "", "")
        return table

    for proc in processes:
        ext_count = sum(1 for c in proc.connections if c.is_external)
        ext_cell = (
            Text(f"  {ext_count}  ", style="bold white on red") if ext_count > 0
            else Text("  0  ", style="bold green")
        )
        status_style = "bold green" if proc.status == "running" else "dim"
        table.add_row(
            str(proc.pid),
            proc.name,
            Text(proc.status, style=status_style),
            f"{proc.cpu_percent:.1f}",
            f"{proc.memory_mb:.0f}",
            ext_cell,
            proc.cmdline[:80] + ("…" if len(proc.cmdline) > 80 else ""),
        )
    return table


# ---------------------------------------------------------------------------
# External connections (grouped, active only)
# ---------------------------------------------------------------------------

def _external_table(processes: list[LLMProcess]) -> Table:
    table = Table(box=box.SIMPLE_HEAVY, header_style="bold red", expand=True)
    table.add_column("Process", min_width=10)
    table.add_column("PID", style="dim", width=7)
    table.add_column("Proto", width=5)
    table.add_column("Remote IP", min_width=20)
    table.add_column("Hostname / Org", min_width=26)
    table.add_column("Port", width=6)
    table.add_column("Conns", width=6)
    table.add_column("Status", width=13)

    groups: dict[tuple, list] = {}
    for proc in processes:
        for conn in proc.connections:
            if not conn.is_external:
                continue
            key = (proc.name, proc.pid, conn.remote_ip, conn.remote_port, conn.protocol)
            groups.setdefault(key, []).append(conn.status)

    if not groups:
        table.add_row("[dim]None[/dim]", "", "", "", "", "", "", "")
        return table

    for (name, pid, remote_ip, port, proto), statuses in sorted(groups.items()):
        count = len(statuses)
        host = hostname(remote_ip)
        short = _short_host(remote_ip, host)
        host_cell = (
            Text(short, style="bold yellow")
            if short != remote_ip
            else Text("resolving…", style="dim italic")
        )
        status = max(set(statuses), key=statuses.count)
        table.add_row(
            Text(name, style="bold"),
            str(pid),
            proto.upper(),
            Text(remote_ip, style="red"),
            host_cell,
            Text(str(port), style="bold red"),
            Text(f"×{count}", style="bold red") if count > 1 else Text("×1"),
            status,
        )
    return table


# ---------------------------------------------------------------------------
# Local connections
# ---------------------------------------------------------------------------

def _local_table(processes: list[LLMProcess]) -> Table:
    table = Table(box=box.SIMPLE, header_style="dim", expand=True)
    table.add_column("PID", style="dim", width=7)
    table.add_column("Process", style="dim", min_width=10)
    table.add_column("Proto", style="dim", width=5)
    table.add_column("Local", style="dim", min_width=22)
    table.add_column("Remote", style="dim", min_width=22)
    table.add_column("Status", style="dim", width=13)

    rows = [(proc, conn) for proc in processes
            for conn in proc.connections if not conn.is_external]

    if not rows:
        table.add_row("", "[dim]No local connections[/dim]", "", "", "", "")
        return table

    for proc, conn in rows:
        table.add_row(str(proc.pid), proc.name, conn.protocol.upper(),
                      conn.local_addr, conn.remote_addr, conn.status)
    return table


# ---------------------------------------------------------------------------
# Session history (log timeline)
# ---------------------------------------------------------------------------

def _history_table(session_log: SessionLog) -> Table:
    table = Table(box=box.SIMPLE_HEAVY, header_style="bold blue", expand=True)
    table.add_column("Time", width=10)
    table.add_column("Event", width=8)
    table.add_column("Process", min_width=10)
    table.add_column("Destination", min_width=26)
    table.add_column("IP", min_width=18)
    table.add_column("Port", width=6)
    table.add_column("Duration", width=9)

    events = session_log.recent_events(limit=25)
    if not events:
        table.add_row("[dim]No events yet[/dim]", "", "", "", "", "", "")
        return table

    for ev in events:
        host = hostname(ev.remote_ip)
        short = _short_host(ev.remote_ip, host)

        event_cell = {
            "opened": Text("OPENED", style="bold red"),
            "closed": Text("CLOSED", style="bold green"),
            "seen":   Text("active", style="dim yellow"),
        }.get(ev.event, Text(ev.event))

        dur_cell = _fmt_dur(ev.duration_s)
        if ev.event == "opened" and ev.duration_s is None:
            dur_text = Text("active", style="bold yellow")
        elif ev.event == "closed":
            dur_text = Text(dur_cell, style="green")
        else:
            dur_text = Text(dur_cell, style="dim")

        table.add_row(
            _fmt_ts(ev.ts),
            event_cell,
            ev.process_name,
            Text(short, style="yellow" if short != ev.remote_ip else "dim"),
            Text(ev.remote_ip, style="dim"),
            str(ev.port),
            dur_text,
        )
    return table


# ---------------------------------------------------------------------------
# Insights panel
# ---------------------------------------------------------------------------

def _insights_panel(session_log: SessionLog, processes: list[LLMProcess]) -> Panel:
    data = session_log.insights()

    lines: list[Text] = []

    if not data:
        return Panel(Text("No data yet — waiting for connections…", style="dim"),
                     title="[bold blue]Insights[/bold blue]", border_style="blue")

    # Session age
    age = data.get("session_age_s", 0)
    age_str = f"{age/60:.1f}m" if age >= 60 else f"{age:.0f}s"
    lines.append(Text(f"Session: {age_str} old", style="dim"))

    # Stats row
    stats = (
        f"  Unique destinations: [bold]{data.get('unique_hosts', 0)}[/bold]"
        f"    Total connections logged: [bold]{data.get('total_connections', 0)}[/bold]"
    )
    lines.append(Text.from_markup(stats))

    # Most contacted
    if data.get("most_contacted"):
        lines.append(Text.from_markup(
            f"  Most contacted: [bold yellow]{data['most_contacted']}[/bold yellow]"
        ))

    # Longest connection
    if data.get("longest"):
        ev = data["longest"]
        host = hostname(ev.remote_ip)
        short = _short_host(ev.remote_ip, host)
        lines.append(Text.from_markup(
            f"  Longest connection: [bold]{_fmt_dur(ev.duration_s)}[/bold]"
            f" → [yellow]{short}[/yellow] ({ev.process_name})"
        ))

    # Phases
    phases = data.get("phases", [])
    if phases:
        lines.append(Text(""))
        lines.append(Text("  Detected phases:", style="bold"))
        phase_icons = {
            "startup": "→",
            "download": "↓",
            "inference": "⚙",
        }
        for phase in phases:
            icon = phase_icons.get(phase["name"], "•")
            ts_str = _fmt_ts(phase["ts"])
            count_str = f"  ({phase['count']} conns)" if phase["count"] else ""
            lines.append(Text.from_markup(
                f"    [dim]{ts_str}[/dim]  {icon}  [cyan]{phase['label']}[/cyan]{count_str}"
            ))

    # Current state hint
    lines.append(Text(""))
    runner_procs = [p for p in processes
                    if "runner" in p.cmdline.lower() or p.memory_mb > 500]
    active_ext = sum(1 for p in processes
                     for c in p.connections if c.is_external)
    if runner_procs and active_ext == 0:
        lines.append(Text.from_markup(
            "  [bold green]Now:[/bold green] Model running locally — [green]no external traffic[/green]"
        ))
    elif runner_procs and active_ext > 0:
        lines.append(Text.from_markup(
            f"  [bold yellow]Now:[/bold yellow] Model running — [red]{active_ext} external connection(s) open[/red]"
        ))
    elif active_ext > 0:
        lines.append(Text.from_markup(
            f"  [bold red]Now:[/bold red] [red]{active_ext} external connection(s) open[/red]"
        ))
    else:
        lines.append(Text.from_markup("  [dim]Now: idle[/dim]"))

    body = Text("\n").join(lines)
    return Panel(body, title="[bold blue]Insights[/bold blue]",
                 border_style="blue", padding=(0, 1))


# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------

def build_layout(
    processes: list[LLMProcess],
    alert_manager: AlertManager,
    session_log: SessionLog,
    scan_count: int,
    interval: float,
) -> Layout:
    ext_conns = [c for p in processes for c in p.connections if c.is_external]
    local_conns = [c for p in processes for c in p.connections if not c.is_external]
    n_ext = len(ext_conns)

    # Header
    if n_ext > 0:
        unique_ips = {c.remote_ip for c in ext_conns}
        short_dests = {_short_host(ip, hostname(ip)) for ip in unique_ips}
        dest_str = ", ".join(sorted(short_dests))
        status_msg = (
            f"[bold white on red]  ALERT  [/bold white on red] "
            f"[red]{n_ext} external → {dest_str}[/red]"
        )
        header_style = "red"
    else:
        status_msg = "[bold green]All clear — no external connections[/bold green]"
        header_style = "green"

    db_path = str(session_log._db_path)
    header_text = (
        f"[bold]LLM Sentinel[/bold]  "
        f"[dim]scan #{scan_count} · {interval}s · {len(processes)} process(es) · "
        f"db: {db_path}[/dim]"
        f"    {status_msg}"
    )

    # Group unique external destinations for panel height
    ext_groups: set[tuple] = set()
    for p in processes:
        for c in ext_conns:
            ext_groups.add((p.pid, c.remote_ip, c.remote_port))
    n_ext_rows = max(len(ext_groups), 1)

    history_events = session_log.recent_events(limit=25)
    # Only show 'seen' events if there's nothing else to show
    display_events = [e for e in history_events if e.event != "seen"]
    if not display_events:
        display_events = history_events[:5]

    proc_h   = max(len(processes) + 5, 6)
    ext_h    = max(n_ext_rows + 5, 5)
    local_h  = max(len(local_conns) + 5, 5) if local_conns else 4
    hist_h   = max(len(display_events) + 5, 7)
    insight_h = 12

    layout = Layout()
    layout.split_column(
        Layout(name="header",   size=3),
        Layout(name="processes", size=proc_h),
        Layout(name="external",  size=ext_h),
        Layout(name="local",     size=local_h),
        Layout(name="history",   size=hist_h),
        Layout(name="insights",  size=insight_h),
    )

    layout["header"].update(Panel(header_text, style=header_style, padding=(0, 1)))

    layout["processes"].update(
        Panel(_process_table(processes),
              title="[bold cyan]LLM Processes[/bold cyan]", border_style="cyan")
    )

    ext_title = (
        f"[bold red]External Connections — Active ({n_ext})[/bold red]"
        if n_ext > 0 else "[dim]External Connections — Active (0)[/dim]"
    )
    layout["external"].update(
        Panel(_external_table(processes), title=ext_title,
              border_style="red" if n_ext > 0 else "dim")
    )

    layout["local"].update(
        Panel(_local_table(processes),
              title=f"[dim]Local Connections ({len(local_conns)})[/dim]",
              border_style="dim")
    )

    layout["history"].update(
        Panel(_history_table(session_log),
              title="[bold blue]Session History[/bold blue]", border_style="blue")
    )

    layout["insights"].update(_insights_panel(session_log, processes))

    return layout


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def run_dashboard(sentinel_fn, interval: float = 3.0):
    with Live(console=console, refresh_per_second=1, screen=True) as live:
        while True:
            processes, alert_manager, session_log, scan_count = sentinel_fn()
            layout = build_layout(processes, alert_manager, session_log,
                                  scan_count, interval)
            live.update(layout)
            time.sleep(interval)
