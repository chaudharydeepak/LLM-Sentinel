"""
Live terminal dashboard using Rich.
Renders detected LLM processes, their connections, and any alerts.
"""

from rich.console import Console
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich import box

from .alerts import AlertManager
from .process_monitor import LLMProcess

console = Console()


def _process_table(processes: list[LLMProcess]) -> Table:
    table = Table(
        box=box.SIMPLE_HEAVY,
        show_header=True,
        header_style="bold cyan",
        expand=True,
    )
    table.add_column("PID", style="dim", width=7)
    table.add_column("Name", min_width=16)
    table.add_column("Pattern", style="dim", min_width=12)
    table.add_column("Status", width=10)
    table.add_column("CPU%", width=6)
    table.add_column("Mem (MB)", width=9)
    table.add_column("Ext. Conns", width=11)
    table.add_column("Command", style="dim")

    if not processes:
        table.add_row(
            "-", "[dim]No LLM processes detected[/dim]", "", "", "", "", "", ""
        )
        return table

    for proc in processes:
        ext_count = sum(1 for c in proc.connections if c.is_external)
        ext_cell = (
            Text(str(ext_count), style="bold red") if ext_count > 0
            else Text("0", style="green")
        )
        status_color = "green" if proc.status == "running" else "dim"
        table.add_row(
            str(proc.pid),
            proc.name,
            proc.matched_pattern,
            Text(proc.status, style=status_color),
            f"{proc.cpu_percent:.1f}",
            f"{proc.memory_mb:.0f}",
            ext_cell,
            proc.cmdline[:60] + ("..." if len(proc.cmdline) > 60 else ""),
        )

    return table


def _connection_table(processes: list[LLMProcess]) -> Table:
    table = Table(
        box=box.SIMPLE,
        show_header=True,
        header_style="bold magenta",
        expand=True,
    )
    table.add_column("PID", width=7)
    table.add_column("Process", min_width=12)
    table.add_column("Proto", width=5)
    table.add_column("Local", min_width=22)
    table.add_column("Remote", min_width=22)
    table.add_column("Status", width=12)
    table.add_column("Risk", width=10)

    any_conn = False
    for proc in processes:
        for conn in proc.connections:
            any_conn = True
            risk_text = (
                Text("EXTERNAL", style="bold red blink")
                if conn.is_external
                else Text("local", style="dim green")
            )
            table.add_row(
                str(proc.pid),
                proc.name,
                conn.protocol.upper(),
                conn.local_addr,
                conn.remote_addr,
                conn.status,
                risk_text,
            )

    if not any_conn:
        table.add_row("-", "[dim]No active connections[/dim]", "", "", "", "", "")

    return table


def _alerts_table(alert_manager: AlertManager) -> Table:
    table = Table(
        box=box.SIMPLE,
        show_header=True,
        header_style="bold red",
        expand=True,
    )
    table.add_column("Time", width=10)
    table.add_column("PID", width=7)
    table.add_column("Process", min_width=12)
    table.add_column("Remote Address", min_width=22)
    table.add_column("Port", width=6)
    table.add_column("Proto", width=5)
    table.add_column("Hits", width=5)

    alerts = alert_manager.get_alerts()
    if not alerts:
        table.add_row("[dim]No alerts[/dim]", "", "", "", "", "", "")
        return table

    for alert in alerts[:20]:  # show last 20
        table.add_row(
            alert.formatted_time(),
            str(alert.pid),
            alert.process_name,
            Text(alert.remote_addr, style="red"),
            str(alert.remote_port),
            alert.protocol.upper(),
            str(alert.count),
        )

    return table


def build_layout(
    processes: list[LLMProcess],
    alert_manager: AlertManager,
    scan_count: int,
    interval: float,
) -> Layout:
    layout = Layout()
    layout.split_column(
        Layout(name="header", size=3),
        Layout(name="processes", size=len(processes) + 6),
        Layout(name="connections", size=max(len(processes) * 2 + 5, 7)),
        Layout(name="alerts"),
    )

    ext_total = sum(
        sum(1 for c in p.connections if c.is_external) for p in processes
    )
    header_style = "bold red" if ext_total > 0 else "bold green"
    status_msg = (
        f"[bold red]ALERT: {ext_total} external connection(s) detected[/bold red]"
        if ext_total > 0
        else "[bold green]All clear — no external connections[/bold green]"
    )

    layout["header"].update(
        Panel(
            f"LLM Sentinel  |  scan #{scan_count}  |  interval {interval}s  |  "
            f"processes: {len(processes)}  |  {status_msg}",
            style=header_style,
        )
    )
    layout["processes"].update(
        Panel(_process_table(processes), title="[bold cyan]Detected LLM Processes[/bold cyan]")
    )
    layout["connections"].update(
        Panel(_connection_table(processes), title="[bold magenta]Active Connections[/bold magenta]")
    )
    layout["alerts"].update(
        Panel(
            _alerts_table(alert_manager),
            title=f"[bold red]Alerts ({alert_manager.total_count} unique)[/bold red]",
        )
    )

    return layout


def run_dashboard(sentinel_fn, interval: float = 3.0):
    """
    sentinel_fn: callable that returns (list[LLMProcess], AlertManager, int)
    Called on each refresh tick.
    """
    with Live(console=console, refresh_per_second=1, screen=True) as live:
        while True:
            processes, alert_manager, scan_count = sentinel_fn()
            layout = build_layout(processes, alert_manager, scan_count, interval)
            live.update(layout)
            import time
            time.sleep(interval)
