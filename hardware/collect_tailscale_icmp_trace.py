"""Collect an auditable ICMP RTT trace to a remote Tailscale peer.

This collector is intended for a real wide-area trace whose remote endpoint
does not expose the HTTP verifier protocol. It records the RTT reported by
``ping`` separately from process wall time and preserves failed probes.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import platform
import re
import socket
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path


FIELDS = (
    "sequence",
    "utc_start",
    "scheduled_monotonic_s",
    "monotonic_start_s",
    "schedule_lag_ms",
    "rtt_ms",
    "probe_wall_ms",
    "success",
    "error",
)
PING_RTT = re.compile(
    r"\btime(?P<relation>[=<])(?P<rtt>[0-9]+(?:\.[0-9]+)?)\s*ms\b"
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def command_output(command: list[str], timeout_s: float = 15.0) -> str | None:
    try:
        return subprocess.check_output(
            command, stderr=subprocess.STDOUT, text=True, timeout=timeout_s
        ).strip()
    except (OSError, subprocess.SubprocessError):
        return None


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_ping_rtt(output: str) -> float | None:
    """Return the ICMP RTT in milliseconds from iputils ``ping`` output."""

    match = PING_RTT.search(output)
    if match is None:
        return None
    value = float(match.group("rtt"))
    if not math.isfinite(value) or value < 0:
        return None
    return value


def probe_once(target: str, timeout_s: float) -> tuple[float | None, float, str]:
    started = time.monotonic()
    timeout_arg = str(max(1, int(math.ceil(timeout_s))))
    try:
        result = subprocess.run(
            ["ping", "-n", "-c", "1", "-W", timeout_arg, target],
            capture_output=True,
            text=True,
            timeout=timeout_s + 1.0,
            check=False,
        )
        wall_ms = (time.monotonic() - started) * 1000.0
    except (OSError, subprocess.SubprocessError) as exc:
        wall_ms = (time.monotonic() - started) * 1000.0
        return None, wall_ms, f"{type(exc).__name__}: {exc}"

    combined = "\n".join(part for part in (result.stdout, result.stderr) if part)
    rtt_ms = parse_ping_rtt(combined)
    if result.returncode == 0 and rtt_ms is not None:
        return rtt_ms, wall_ms, ""
    final_line = next(
        (line.strip() for line in reversed(combined.splitlines()) if line.strip()),
        "no ping output",
    )
    return None, wall_ms, f"ping exit {result.returncode}: {final_line}"


def collect(args: argparse.Namespace) -> None:
    output = Path(args.output).expanduser().resolve()
    if output.exists() or output.with_suffix(".manifest.json").exists():
        raise FileExistsError(f"Refusing to overwrite trace output for {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    manifest_path = output.with_suffix(".manifest.json")

    entrypoint = Path(__file__).resolve()
    repo_root = entrypoint.parent.parent
    success_count = 0
    row_count = 0
    interrupted = False
    route_before = command_output(["ip", "route", "get", args.target])
    tailscale_ping_before = command_output(
        ["tailscale", "ping", "-c", "3", args.target], timeout_s=30.0
    )
    started_utc = utc_now()
    started_monotonic = time.monotonic()
    deadline = started_monotonic + args.duration_s

    try:
        with output.open("x", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=FIELDS)
            writer.writeheader()
            handle.flush()

            sequence = 0
            next_scheduled = started_monotonic
            while time.monotonic() < deadline:
                scheduled = next_scheduled
                wait_s = scheduled - time.monotonic()
                if wait_s > 0:
                    time.sleep(wait_s)
                if time.monotonic() >= deadline:
                    break

                utc_start = utc_now()
                mono_start = time.monotonic()
                rtt_ms, wall_ms, error = probe_once(args.target, args.timeout_s)
                success = rtt_ms is not None
                writer.writerow(
                    {
                        "sequence": sequence,
                        "utc_start": utc_start,
                        "scheduled_monotonic_s": f"{scheduled:.9f}",
                        "monotonic_start_s": f"{mono_start:.9f}",
                        "schedule_lag_ms": (
                            f"{max(0.0, mono_start - scheduled) * 1000.0:.6f}"
                        ),
                        "rtt_ms": "" if rtt_ms is None else f"{rtt_ms:.6f}",
                        "probe_wall_ms": f"{wall_ms:.6f}",
                        "success": int(success),
                        "error": error,
                    }
                )
                handle.flush()
                row_count += 1
                success_count += int(success)
                sequence += 1
                next_scheduled = scheduled + args.interval_s
                if next_scheduled < time.monotonic():
                    next_scheduled = time.monotonic() + args.interval_s
    except KeyboardInterrupt:
        interrupted = True
    finally:
        manifest = {
            "schema_version": 1,
            "trace_kind": "icmp_echo_rtt",
            "transport": (
                "ICMP echo over Tailscale over the edge device's active access path"
            ),
            "network_label": args.network_label,
            "target": args.target,
            "target_resolution": command_output(["getent", "hosts", args.target]),
            "client_hostname": socket.gethostname(),
            "client_tailscale_ipv4": command_output(["tailscale", "ip", "-4"]),
            "platform": platform.platform(),
            "python": platform.python_version(),
            "git_commit": command_output(
                ["git", "-C", str(repo_root), "rev-parse", "HEAD"]
            ),
            "entrypoint": str(entrypoint),
            "entrypoint_sha256": sha256_file(entrypoint),
            "started_utc": started_utc,
            "finished_utc": utc_now(),
            "requested_duration_s": args.duration_s,
            "actual_elapsed_s": time.monotonic() - started_monotonic,
            "sampling_interval_s": args.interval_s,
            "scheduler": "fixed cadence without catch-up after overruns",
            "probe_timeout_s": args.timeout_s,
            "rows": row_count,
            "successful_rows": success_count,
            "failed_rows": row_count - success_count,
            "interrupted": interrupted,
            "raw_csv": output.name,
            "raw_csv_sha256": sha256_file(output) if output.exists() else None,
            "route_before": route_before,
            "route_after": command_output(["ip", "route", "get", args.target]),
            "tailscale_ping_before": tailscale_ping_before,
            "tailscale_ping_after": command_output(
                ["tailscale", "ping", "-c", "3", args.target], timeout_s=30.0
            ),
            "wifi_interface": command_output(["iw", "dev"]),
            "network_manager_devices": command_output(
                ["nmcli", "-t", "-f", "DEVICE,TYPE,STATE,CONNECTION", "device"]
            ),
            "interpretation_boundary": (
                "RTT is the ICMP echo time to a remote Tailscale peer. It is a "
                "measured wide-area delay trace for replay, not an HTTP verifier "
                "RTT, one-way propagation trace, cellular trace, or independent "
                "verification that the operator-assigned peer name identifies its "
                "physical location."
            ),
        }
        manifest_path.write_text(
            json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
        )
        print(json.dumps(manifest, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", required=True, help="Tailscale peer IP or name")
    parser.add_argument("--output", required=True, help="Raw CSV output path")
    parser.add_argument("--duration-s", type=float, default=1800.0)
    parser.add_argument("--interval-s", type=float, default=0.5)
    parser.add_argument("--timeout-s", type=float, default=3.0)
    parser.add_argument("--network-label", default="unspecified")
    args = parser.parse_args()
    if args.duration_s <= 0 or args.interval_s <= 0 or args.timeout_s <= 0:
        parser.error("duration, interval, and timeout must be positive")
    return args


if __name__ == "__main__":
    collect(parse_args())
