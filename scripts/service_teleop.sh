#!/usr/bin/env bash
# supervisor unit vla_teleop: the teleop/recording panel with its arguments from config/teleop.env.
# `exec` so supervisor's SIGTERM reaches python directly (the panel turns it into a clean Quit).
set -euo pipefail
cd "$(dirname "$(readlink -f "$0")")/.."
ENV_FILE=config/teleop.env
[ -f "$ENV_FILE" ] || { echo "[service] missing $ENV_FILE — run scripts/install_services.sh rd (copies services/rd/teleop.env.example)"; exit 1; }
set -a; . "$ENV_FILE"; set +a
: "${ROOT:?set ROOT in $ENV_FILE}"; : "${TASK:?set TASK in $ENV_FILE}"
REPO_ID="${REPO_ID:-local/$(basename "$ROOT")}"
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
# The eval and the ROS UR driver own the UR control script, the deck and port 8090: stop them first
# (ur5e_lerobot/services.py; `supervisorctl stop` is their clean shutdown, so this can take a minute).
./run.sh -m ur5e_lerobot.services teleop-preflight || { echo "[service] teleop preflight failed"; exit 1; }

args=(-m ur5e_lerobot.teleop.manual_panel --engine hardware --cameras udp --ui web
      --root "$ROOT" --repo_id "$REPO_ID" --task "$TASK")
[ "${RESUME:-1}" = 1 ] && args+=(--resume)
[ -n "${SCENE_SQUARE:-}" ] && args+=(--scene-square "$SCENE_SQUARE")
[ -n "${SIDE_SQUARE:-}" ] && args+=(--side-square "$SIDE_SQUARE")
[ -n "${OPERATOR:-}" ] && args+=(--operator "$OPERATOR")
# shellcheck disable=SC2206
[ -n "${EXTRA_ARGS:-}" ] && args+=(${EXTRA_ARGS})
echo "[service] $(date +%T) vla_teleop: ${args[*]}"
exec ./run.sh "${args[@]}"
