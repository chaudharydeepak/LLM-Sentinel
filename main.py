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
from llm_sentinel.resolver import hostname
from llm_sentinel.session_log import SessionLog


def build_sentinel(alert_manager: AlertManager, session_log: SessionLog,
                   web_enabled: bool = False):
    scan_count = 0
    # Tracks currently-open external connections: key -> log row_id
    # key = (pid, remote_ip, remote_port)
    open_conns: dict[tuple, int] = {}

    def tick():
        nonlocal scan_count
        scan_count += 1

        processes = get_llm_processes()
        pids = [p.pid for p in processes]
        connections_by_pid = get_all_llm_connections(pids)

        # Build set of currently active external connection keys
        active_keys: set[tuple] = set()

        for proc in processes:
            conns = connections_by_pid.get(proc.pid, [])
            proc.connections = conns
            for conn in conns:
                if not conn.is_external:
                    continue

                key = (proc.pid, conn.remote_ip, conn.remote_port)
                active_keys.add(key)

                if key not in open_conns:
                    # New connection — log it and fire an alert
                    host = hostname(conn.remote_ip)
                    row_id = session_log.record_opened(
                        pid=proc.pid,
                        process_name=proc.name,
                        remote_ip=conn.remote_ip,
                        hostname=host,
                        port=conn.remote_port,
                        protocol=conn.protocol,
                    )
                    open_conns[key] = row_id
                    alert_manager.record(
                        pid=proc.pid,
                        process_name=proc.name,
                        remote_addr=conn.remote_ip,
                        remote_port=conn.remote_port,
                        protocol=conn.protocol,
                    )
                else:
                    # Still active — update heartbeat and refresh hostname if resolved
                    row_id = open_conns[key]
                    session_log.record_seen(row_id)
                    host = hostname(conn.remote_ip)
                    if host != conn.remote_ip:
                        session_log.update_hostname(row_id, host)

        # Detect closed connections (were open last scan, not active now)
        closed_keys = set(open_conns.keys()) - active_keys
        for key in closed_keys:
            session_log.record_closed(open_conns.pop(key))

        if web_enabled:
            from llm_sentinel.web import update_state
            update_state(processes, session_log, scan_count, interval=3.0)

        return processes, alert_manager, session_log, scan_count

    return tick


def run_dashboard_mode(interval: float, log_file: str | None, web: bool = False,
                       web_port: int = 7777, web_password: str | None = None):
    from llm_sentinel.dashboard import run_dashboard

    alert_manager = AlertManager(log_to_file=log_file)
    session_log = SessionLog()
    sentinel_fn = build_sentinel(alert_manager, session_log, web_enabled=web)

    if web:
        import secrets as _secrets
        from llm_sentinel.auth import AuthDB
        from llm_sentinel.web import start as start_web, setup_auth

        auth = AuthDB(session_log._conn)

        if web_password:
            auth.create_user("admin", web_password)
            print(f"  Web auth: password set for admin")
        elif not auth.has_users():
            generated = _secrets.token_urlsafe(12)
            auth.create_user("admin", generated)
            print(f"  Web auth: no users found — created admin account")
            print(f"  Web password: {generated}")
            print(f"  (change with --web-password YOUR_PASSWORD)")

        setup_auth(auth)
        url = start_web(port=web_port)
        import time; time.sleep(0.5)
        print(f"  Web dashboard → {url}")
        time.sleep(1.5)

    run_dashboard(sentinel_fn, interval=interval)


def run_cli_mode(interval: float, log_file: str | None, count: int | None):
    from rich.console import Console
    from rich.table import Table

    console = Console()
    alert_manager = AlertManager(log_to_file=log_file)
    session_log = SessionLog()
    sentinel_fn = build_sentinel(alert_manager, session_log)
    scans = 0

    try:
        while True:
            processes, am, sl, scan_num = sentinel_fn()
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
    parser.add_argument(
        "--web",
        action="store_true",
        help="Also serve web dashboard at http://localhost:7777",
    )
    parser.add_argument(
        "--web-port",
        type=int,
        default=7777,
        metavar="PORT",
        help="Port for web dashboard (default: 7777)",
    )
    parser.add_argument(
        "--web-password",
        metavar="PASSWORD",
        help="Set the admin password for the web dashboard",
    )
    args = parser.parse_args()

    try:
        if args.no_dashboard:
            run_cli_mode(interval=args.interval, log_file=args.log, count=args.count)
        else:
            run_dashboard_mode(interval=args.interval, log_file=args.log,
                               web=args.web, web_port=args.web_port,
                               web_password=args.web_password)
    except KeyboardInterrupt:
        print("\nStopped.")
        sys.exit(0)


if __name__ == "__main__":
    main()
