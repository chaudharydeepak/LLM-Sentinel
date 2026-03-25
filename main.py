#!/usr/bin/env python3
"""
LLM Sentinel — monitors local LLM processes for unexpected external network connections.
"""

import argparse
import sys
import time

from llm_sentinel.alerts import AlertManager
from llm_sentinel.network_monitor import get_all_llm_connections
from llm_sentinel.process_monitor import get_llm_processes


def build_sentinel(alert_manager: AlertManager):
    scan_count = 0

    def tick():
        nonlocal scan_count
        scan_count += 1

        processes = get_llm_processes()
        pids = [p.pid for p in processes]
        connections_by_pid = get_all_llm_connections(pids)

        for proc in processes:
            conns = connections_by_pid.get(proc.pid, [])
            proc.connections = conns
            for conn in conns:
                if conn.is_external:
                    alert_manager.record(
                        pid=proc.pid,
                        process_name=proc.name,
                        remote_addr=conn.remote_ip,
                        remote_port=conn.remote_port,
                        protocol=conn.protocol,
                    )

        return processes, alert_manager, scan_count

    return tick


def run_dashboard_mode(interval: float, log_file: str | None):
    from llm_sentinel.dashboard import run_dashboard

    alert_manager = AlertManager(log_to_file=log_file)
    sentinel_fn = build_sentinel(alert_manager)
    run_dashboard(sentinel_fn, interval=interval)


def run_cli_mode(interval: float, log_file: str | None, count: int | None):
    """Non-interactive mode — prints findings to stdout."""
    from rich.console import Console
    from rich.table import Table

    console = Console()
    alert_manager = AlertManager(log_to_file=log_file)
    sentinel_fn = build_sentinel(alert_manager)
    scans = 0

    try:
        while True:
            processes, am, scan_num = sentinel_fn()
            scans += 1

            table = Table(title=f"Scan #{scan_num}", show_lines=True)
            table.add_column("PID")
            table.add_column("Name")
            table.add_column("Ext Connections")

            for proc in processes:
                ext = [c for c in proc.connections if c.is_external]
                ext_str = ", ".join(f"{c.remote_addr}" for c in ext) or "none"
                table.add_row(str(proc.pid), proc.name, ext_str)

            if not processes:
                console.print(f"[dim]Scan #{scan_num}: no LLM processes found[/dim]")
            else:
                console.print(table)

            if count and scans >= count:
                break

            time.sleep(interval)
    except KeyboardInterrupt:
        console.print("\n[dim]Stopped.[/dim]")


def main():
    parser = argparse.ArgumentParser(
        description="LLM Sentinel — monitor local LLM processes for external network leaks"
    )
    parser.add_argument(
        "--interval", "-i",
        type=float,
        default=3.0,
        help="Scan interval in seconds (default: 3)",
    )
    parser.add_argument(
        "--no-dashboard",
        action="store_true",
        help="Print to stdout instead of live dashboard",
    )
    parser.add_argument(
        "--log",
        metavar="FILE",
        help="Log external connection alerts to a file",
    )
    parser.add_argument(
        "--count", "-n",
        type=int,
        default=None,
        help="Run N scans then exit (useful for scripting)",
    )
    args = parser.parse_args()

    try:
        if args.no_dashboard:
            run_cli_mode(interval=args.interval, log_file=args.log, count=args.count)
        else:
            run_dashboard_mode(interval=args.interval, log_file=args.log)
    except KeyboardInterrupt:
        print("\nStopped.")
        sys.exit(0)


if __name__ == "__main__":
    main()
