#!/usr/bin/env python3
"""
markov_netem.py — Background daemon: Markov-switched tc netem delay.

Run on the machine where tc netem controls the network interface (3090 server
or Jetson, depending on where you apply netem). This process runs indefinitely,
switching between good/bad states according to a two-state Markov chain and
updating tc netem accordingly.

Usage:
  sudo python markov_netem.py --iface eth0 --d-good 5 --d-bad 80 \\
      --p-g2b 0.1 --p-b2g 0.1 --interval 0.1

Press Ctrl+C to stop and remove netem rules.

The current state is written to /tmp/markov_netem_state.txt every interval
so that the client can read it for contextual UCB experiments.
"""

from __future__ import annotations

import argparse
import random
import signal
import subprocess
import sys
import time
from pathlib import Path

STATE_FILE = Path("/tmp/markov_netem_state.txt")


def set_netem(iface: str, delay_ms: int, jitter_ms: int = 0) -> None:
    if jitter_ms > 0:
        cmd = ["tc", "qdisc", "change", "dev", iface, "root",
               "netem", "delay", f"{delay_ms}ms", f"{jitter_ms}ms"]
    else:
        cmd = ["tc", "qdisc", "change", "dev", iface, "root",
               "netem", "delay", f"{delay_ms}ms"]
    subprocess.run(cmd, check=True)


def init_netem(iface: str, delay_ms: int) -> None:
    subprocess.run(
        ["tc", "qdisc", "add", "dev", iface, "root",
         "netem", "delay", f"{delay_ms}ms"],
        check=True,
    )


def del_netem(iface: str) -> None:
    subprocess.run(
        ["tc", "qdisc", "del", "dev", iface, "root"],
        check=False,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--iface", default="eth0")
    parser.add_argument("--d-good", type=int, default=5,
                        help="Delay in good state (ms)")
    parser.add_argument("--d-bad", type=int, default=80,
                        help="Delay in bad state (ms)")
    parser.add_argument("--jitter-good", type=int, default=0,
                        help="Jitter in good state (ms)")
    parser.add_argument("--jitter-bad", type=int, default=0,
                        help="Jitter in bad state (ms)")
    parser.add_argument("--p-g2b", type=float, default=0.1,
                        help="Transition prob good->bad per interval")
    parser.add_argument("--p-b2g", type=float, default=0.1,
                        help="Transition prob bad->good per interval")
    parser.add_argument("--interval", type=float, default=0.1,
                        help="State-check interval (seconds)")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    random.seed(args.seed)
    state = "good"

    # Cleanup on exit
    def _cleanup(sig, frame):
        print(f"\n[netem] Removing netem on {args.iface}")
        del_netem(args.iface)
        STATE_FILE.unlink(missing_ok=True)
        sys.exit(0)

    signal.signal(signal.SIGINT, _cleanup)
    signal.signal(signal.SIGTERM, _cleanup)

    print(f"[netem] Initializing on {args.iface}, state=good ({args.d_good}ms)")
    init_netem(args.iface, args.d_good)
    STATE_FILE.write_text(f"{state} {args.d_good}")

    prev_state = state
    t = 0
    while True:
        time.sleep(args.interval)
        t += 1

        # Markov transition
        if state == "good":
            if random.random() < args.p_g2b:
                state = "bad"
        else:
            if random.random() < args.p_b2g:
                state = "good"

        # Apply netem if state changed
        if state != prev_state:
            delay = args.d_good if state == "good" else args.d_bad
            jitter = args.jitter_good if state == "good" else args.jitter_bad
            set_netem(args.iface, delay, jitter)
            STATE_FILE.write_text(f"{state} {delay}")
            print(f"[netem] t={t}  {prev_state} -> {state} ({delay}ms)")
            prev_state = state


if __name__ == "__main__":
    main()
