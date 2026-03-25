"""
Live terminal dashboard using Rich.
Layout:
  ┌─ Header ──────────────────────────────┐
  ├─ Detected LLM Processes ──────────────┤
  ├─ External Connections (RED) ──────────┤  ← prominent, with hostname
  ├─ Local Connections (dimmed) ──────────┤
  └─ Alerts ──────────────────────────────┘
"""

import time

from rich.console import Console
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich import box

from .alerts import Alert, AlertManager
from .network_monitor import Connection
from .process_monitor import LLMProcess
from .resolver import hostname

console = Console()


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
# External connections table (prominent)
# ---------------------------------------------------------------------------

def _external_table(processes: list[LLMProcess]) -> Table:
    table = Table(box=box.SIMPLE_HEAVY, header_style="bold red", expand=True)
    table.add_column("PID", width=7)
    table.add_column("Process", min_width=10)
    table.add_column("Proto", width=5)
    table.add_column("Local Port", width=11)
    table.add_column("Remote IP", min_width=18)
    table.add_column("Hostname", min_width=28)
    table.add_column("Port", width=6)
    table.add_column("Status", width=13)

    rows = [
        (proc, conn)
        for proc in processes
        for conn in proc.connections
        if conn.is_external
    ]

    if not rows:
        table.add_row("", "[dim]None[/dim]", "", "", "", "", "", "")
        return table

    for proc, conn in rows:
        host = hostname(conn.remote_ip)
        # Shorten long hostnames: keep last 3 labels
        labels = host.split(".")
        short_host = ".".join(labels[-3:]) if len(labels) > 3 else host
        if short_host != conn.remote_ip:
            host_cell = Text(short_host, style="bold yellow")
        else:
            host_cell = Text(conn.remote_ip, style="red")  # still resolving

        local_port = conn.local_addr.rsplit(":", 1)[-1] if ":" in conn.local_addr else conn.local_addr
        table.add_row(
            str(proc.pid),
            Text(proc.name, style="bold"),
            conn.protocol.upper(),
            local_port,
            Text(conn.remote_ip, style="red"),
            host_cell,
            Text(str(conn.remote_port), style="bold red"),
            conn.status,
        )
    return table


# ---------------------------------------------------------------------------
# Local connections table (de-emphasised)
# ---------------------------------------------------------------------------

def _local_table(processes: list[LLMProcess]) -> Table:
    table = Table(box=box.SIMPLE, header_style="dim", expand=True)
    table.add_column("PID", style="dim", width=7)
    table.add_column("Process", style="dim", min_width=10)
    table.add_column("Proto", style="dim", width=5)
    table.add_column("Local", style="dim", min_width=22)
    table.add_column("Remote", style="dim", min_width=22)
    table.add_column("Status", style="dim", width=13)

    rows = [
        (proc, conn)
        for proc in processes
        for conn in proc.connections
        if not conn.is_external
    ]

    if not rows:
        table.add_row("", "[dim]No local connections[/dim]", "", "", "", "")
        return table

    for proc, conn in rows:
        table.add_row(
            str(proc.pid), proc.name, conn.protocol.upper(),
            conn.local_addr, conn.remote_addr, conn.status,
        )
    return table


# ---------------------------------------------------------------------------
# Alerts table
# ---------------------------------------------------------------------------

def _alerts_table(alert_manager: AlertManager) -> Table:
    table = Table(box=box.SIMPLE_HEAVY, header_style="bold red", expand=True)
    table.add_column("Time", width=10)
    table.add_column("PID", width=7)
    table.add_column("Process", min_width=10)
    table.add_column("Remote IP", min_width=17)
    table.add_column("Hostname", min_width=30)
    table.add_column("Port", width=6)
    table.add_column("Proto", width=5)
    table.add_column("Hits", width=5)

    alerts = alert_manager.get_alerts()
    if not alerts:
        table.add_row("[dim]No alerts yet[/dim]", "", "", "", "", "", "", "")
        return table

    for alert in alerts[:20]:
        host = hostname(alert.remote_addr)
        labels = host.split(".")
        short_host = ".".join(labels[-3:]) if len(labels) > 3 else host
        host_cell = (
            Text(short_host, style="bold yellow")
            if short_host != alert.remote_addr
            else Text("resolving…", style="dim")
        )
        table.add_row(
            alert.formatted_time(),
            str(alert.pid),
            alert.process_name,
            Text(alert.remote_addr, style="red"),
            host_cell,
            Text(str(alert.remote_port), style="bold"),
            alert.protocol.upper(),
            Text(str(alert.count), style="bold" if alert.count > 1 else ""),
        )
    return table


# ---------------------------------------------------------------------------
# Layout assembly
# ---------------------------------------------------------------------------

def build_layout(
    processes: list[LLMProcess],
    alert_manager: AlertManager,
    scan_count: int,
    interval: float,
) -> Layout:
    ext_conns = [
        conn
        for proc in processes
        for conn in proc.connections
        if conn.is_external
    ]
    local_conns = [
        conn
        for proc in processes
        for conn in proc.connections
        if not conn.is_external
    ]
    n_ext = len(ext_conns)

    # Header message
    if n_ext > 0:
        unique_ips = {c.remote_ip for c in ext_conns}
        hosts = [hostname(ip) for ip in unique_ips]
        short = [".".join(h.split(".")[-3:]) if "." in h and h != ip else ip
                 for h, ip in zip(hosts, unique_ips)]
        dest_str = ", ".join(sorted(set(short)))
        status_msg = (
            f"[bold white on red]  ALERT  [/bold white on red] "
            f"[red]{n_ext} external connection(s) → {dest_str}[/red]"
        )
        header_style = "red"
    else:
        status_msg = "[bold green]All clear — no external connections[/bold green]"
        header_style = "green"

    header_text = (
        f"[bold]LLM Sentinel[/bold]  "
        f"[dim]scan #{scan_count} · {interval}s interval · {len(processes)} process(es)[/dim]"
        f"    {status_msg}"
    )

    # Dynamic heights
    proc_h = max(len(processes) + 5, 6)
    ext_h = max(n_ext + 5, 5)
    local_h = max(len(local_conns) + 5, 5) if local_conns else 4
    alert_h = max(min(len(alert_manager.get_alerts()), 20) + 5, 5)

    layout = Layout()
    layout.split_column(
        Layout(name="header", size=3),
        Layout(name="processes", size=proc_h),
        Layout(name="external", size=ext_h),
        Layout(name="local", size=local_h),
        Layout(name="alerts", size=alert_h),
    )

    layout["header"].update(Panel(header_text, style=header_style, padding=(0, 1)))

    layout["processes"].update(
        Panel(
            _process_table(processes),
            title="[bold cyan]LLM Processes[/bold cyan]",
            border_style="cyan",
        )
    )

    ext_title = (
        f"[bold red]External Connections ({n_ext})[/bold red]"
        if n_ext > 0
        else "[dim]External Connections (0)[/dim]"
    )
    layout["external"].update(
        Panel(
            _external_table(processes),
            title=ext_title,
            border_style="red" if n_ext > 0 else "dim",
        )
    )

    layout["local"].update(
        Panel(
            _local_table(processes),
            title=f"[dim]Local Connections ({len(local_conns)})[/dim]",
            border_style="dim",
        )
    )

    n_alerts = alert_manager.total_count
    alert_title = (
        f"[bold red]Alerts — {n_alerts} unique destination(s)[/bold red]"
        if n_alerts > 0
        else "[dim]Alerts[/dim]"
    )
    layout["alerts"].update(
        Panel(
            _alerts_table(alert_manager),
            title=alert_title,
            border_style="red" if n_alerts > 0 else "dim",
        )
    )

    return layout


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def run_dashboard(sentinel_fn, interval: float = 3.0):
    with Live(console=console, refresh_per_second=1, screen=True) as live:
        while True:
            processes, alert_manager, scan_count = sentinel_fn()
            layout = build_layout(processes, alert_manager, scan_count, interval)
            live.update(layout)
            time.sleep(interval)
