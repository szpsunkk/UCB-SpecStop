#!/usr/bin/env bash
# Run on the 3090 server to manage network delay emulation (tc netem).
# Must be run with sudo.
#
# Usage:
#   bash setup_netem.sh add   ETH_IFACE  DELAY_MS [JITTER_MS] [DIST]
#   bash setup_netem.sh change ETH_IFACE DELAY_MS [JITTER_MS] [DIST]
#   bash setup_netem.sh del   ETH_IFACE
#
# DIST: normal | pareto | paretonormal (default: none = deterministic)
# Example (deterministic 50ms):
#   sudo bash setup_netem.sh add eth0 50
# Example (exponential jitter, mean=50ms, std=30ms):
#   sudo bash setup_netem.sh add eth0 50 30 normal

set -euo pipefail

ACTION=${1:-add}
IFACE=${2:-eth0}
DELAY_MS=${3:-0}
JITTER_MS=${4:-0}
DIST=${5:-}

case "$ACTION" in
  add)
    if [ -n "$DIST" ] && [ "$JITTER_MS" -gt 0 ]; then
      tc qdisc add dev "$IFACE" root netem delay "${DELAY_MS}ms" "${JITTER_MS}ms" distribution "$DIST"
    elif [ "$JITTER_MS" -gt 0 ]; then
      tc qdisc add dev "$IFACE" root netem delay "${DELAY_MS}ms" "${JITTER_MS}ms"
    else
      tc qdisc add dev "$IFACE" root netem delay "${DELAY_MS}ms"
    fi
    echo "Added netem: delay=${DELAY_MS}ms jitter=${JITTER_MS}ms dist=${DIST:-deterministic}"
    ;;
  change)
    if [ -n "$DIST" ] && [ "$JITTER_MS" -gt 0 ]; then
      tc qdisc change dev "$IFACE" root netem delay "${DELAY_MS}ms" "${JITTER_MS}ms" distribution "$DIST"
    elif [ "$JITTER_MS" -gt 0 ]; then
      tc qdisc change dev "$IFACE" root netem delay "${DELAY_MS}ms" "${JITTER_MS}ms"
    else
      tc qdisc change dev "$IFACE" root netem delay "${DELAY_MS}ms"
    fi
    echo "Changed netem: delay=${DELAY_MS}ms jitter=${JITTER_MS}ms"
    ;;
  del)
    tc qdisc del dev "$IFACE" root 2>/dev/null && echo "Removed netem on $IFACE" || echo "No netem to remove"
    ;;
  *)
    echo "Unknown action: $ACTION. Use add|change|del"
    exit 1
    ;;
esac
