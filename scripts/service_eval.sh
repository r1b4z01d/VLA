#!/usr/bin/env bash
# supervisor unit vla_eval: scripts/eval_hw.py with its arguments from config/eval.env.
# `exec`s the eval so supervisor's SIGTERM reaches python directly (the eval turns it into EXIT: run
# closed + uploaded). The eval's own preflight stops the recording panel and starts the inference server.
set -euo pipefail
cd "$(dirname "$(readlink -f "$0")")/.."
ENV_FILE=config/eval.env
[ -f "$ENV_FILE" ] || { echo "[service] missing $ENV_FILE — run scripts/install_services.sh rd (copies services/rd/eval.env.example)"; exit 1; }
set -a; . "$ENV_FILE"; set +a
: "${TASK:?set TASK in $ENV_FILE}"
REMOTE="${REMOTE:-127.0.0.1:8777}"
# The eval does its own preflight at launch (ur5e_lerobot/services.py): it stops vla_teleop and the ROS
# UR driver if they run (they own the UR control script, the deck and port 8090) and brings
# vla_infer_server up for a loopback REMOTE (AUTO_INFER=0 only verifies it).
# Take over from a stale robot app holding the page port — typically a panel started from a terminal
# and never quit (with --ui web its window is hidden; one wedged for 95 min on 2026-10-26). Starting
# this unit is the operator's intent to run THIS instance, so a stale panel/eval is asked to quit
# (SIGTERM = its clean shutdown), then killed if it ignores that. Anything else holding the port is
# left alone (the app keeps retrying the port by itself).
take_over_port() {
  local port="$1" pid kids
  # `|| true`: with nothing on the port grep exits 1, and under set -e/pipefail that killed the whole
  # wrapper before the launch (supervisor saw exit 1 x4 -> FATAL, 2026-10-26 01:34).
  pid=$(ss -ltnpH "sport = :$port" 2>/dev/null | grep -oE 'pid=[0-9]+' | head -1 | cut -d= -f2 || true)
  [ -n "$pid" ] && [ "$pid" != "$$" ] || return 0
  if tr '\0' ' ' < "/proc/$pid/cmdline" 2>/dev/null | grep -qE 'manual_panel|eval_hw'; then
    kids=$(pgrep -P "$pid" 2>/dev/null || true)
    echo "[service] port $port is held by a stale robot app (pid $pid) — asking it to quit"
    kill -TERM "$pid" 2>/dev/null || true
    for _ in $(seq 1 40); do kill -0 "$pid" 2>/dev/null || break; sleep 0.5; done
    if kill -0 "$pid" 2>/dev/null; then
      echo "[service] pid $pid ignored SIGTERM for 20 s — killing it"
      kill -KILL "$pid" 2>/dev/null || true
    fi
    for c in $kids; do kill -KILL "$c" 2>/dev/null || true; done   # forked writer children also hold the port
    sleep 1
  else
    echo "[service] port $port is held by pid $pid (not a robot app) — leaving it; the page will retry"
  fi
}
take_over_port "${WEB_PORT:-8090}"

args=(scripts/eval_hw.py --remote "$REMOTE" --cameras udp --ui web --task "$TASK" --fps "${FPS:-12}")
[ "${VIDEO:-1}" = 1 ] && args+=(--video)
[ "${AUTO_INFER:-1}" = 1 ] || args+=(--no-auto-infer)
[ -n "${SCENE_SQUARE:-}" ] && args+=(--scene-square "$SCENE_SQUARE")
[ -n "${SIDE_SQUARE:-}" ] && args+=(--side-square "$SIDE_SQUARE")
# shellcheck disable=SC2206
[ -n "${EXTRA_ARGS:-}" ] && args+=(${EXTRA_ARGS})
echo "[service] $(date +%T) vla_eval: ${args[*]}"
exec ./run.sh "${args[@]}"
