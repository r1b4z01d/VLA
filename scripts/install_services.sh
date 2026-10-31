#!/usr/bin/env bash
# Install the VLA supervisor units (mirrors rd_ws/Scripts/install_services.sh). Units are per-host
# because the user/paths differ (bryan@GPU vs rd@robot PC):
#
#   ./scripts/install_services.sh          # GPU box:   services/*.conf
#   ./scripts/install_services.sh rd       # robot PC:  services/rd/*.conf
#
# GPU   vla_webui        autostart=true  -> the datasets/models/evals dashboard, port 8081
#       vla_infer_server autostart=false -> FALLBACK only — the robot's Jetson serves itself (see `rd`);
#                                           start it only to drive from the GPU box (it pins GPU memory):
#                                           sudo supervisorctl start vla_infer_server
# RD    vla_teleop             autostart=false -> the recording panel, args from config/teleop.env (service_teleop.sh)
#       vla_eval               autostart=false -> the policy eval, args from config/eval.env (service_eval.sh)
#       vla_cameras            autostart=true -> owns the USB cameras, multicasts frames (camera_server.py)
#       vla_camera_ros_bridge  autostart=true -> the same frames as ROS 2 CompressedImage topics
#       vla_infer_server       autostart=false -> THE production inference server since the robot PC became a
#                                                 Jetson Orin (rd_ws/provision/jetson/): start it before driving, eval --remote 127.0.0.1:8777
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONF_D=/etc/supervisor/conf.d
HOST_DIR="${1:-}"                 # "" = GPU units at services/, "rd" = services/rd/
SRC="$REPO/services${HOST_DIR:+/$HOST_DIR}"

[ -d "$CONF_D" ] || { echo "no $CONF_D — is supervisor installed?" >&2; exit 1; }
ls "$SRC"/*.conf >/dev/null 2>&1 || { echo "no units in $SRC" >&2; exit 1; }

mkdir -p "$REPO/log"   # the units log here; supervisor won't create it and fails to start without it

# The served checkpoint is an indirection so the unit file never has to change. Point it at something
# on first install if it isn't set up yet.
if [ ! -e "$REPO/outputs/serve_ckpt" ]; then
  echo "note: $REPO/outputs/serve_ckpt does not exist — vla_infer_server will fail until you set it:"
  echo "  ln -sfn $REPO/outputs/train/<run>/checkpoints/last/pretrained_model $REPO/outputs/serve_ckpt"
fi

if [ "$HOST_DIR" = "rd" ]; then
  # The camera service multicasts 30-120 KB JPEG frames as ~1.4 KB UDP chunks; three cameras can
  # land ~360 KB in a receiver at once, and the kernel default socket receive buffer is 208 KB
  # (net.core.rmem_max — a larger SO_RCVBUF request is silently clamped to it). 16 MB lets every
  # subscriber, and DDS which has the same shape of problem, absorb a burst. Prudent, not a measured
  # fix: the frame drops seen on this box were the ROS bridge's own doing (camera_ros_bridge.py),
  # and the box's system-wide UDP error counter runs at ~38k/s from other stacks regardless.
  # Persisted so a reboot keeps it; applied now so this install takes effect without one.
  SYSCTL=/etc/sysctl.d/99-vla-camera-udp.conf
  printf 'net.core.rmem_max = 16777216\nnet.core.rmem_default = 1048576\n' | sudo tee "$SYSCTL" >/dev/null
  sudo sysctl -q -p "$SYSCTL"
  echo "udp receive buffer: rmem_max=$(sysctl -n net.core.rmem_max) (persisted in $SYSCTL)"
fi

if [ "$HOST_DIR" = "rd" ]; then
  # Per-robot arguments for the manual-start apps. Examples are the repo's; the live copies are the
  # operator's and are never overwritten.
  mkdir -p "$REPO/config"
  for n in teleop eval; do
    if [ ! -f "$REPO/config/$n.env" ]; then
      cp "$SRC/$n.env.example" "$REPO/config/$n.env"
      echo "config/$n.env created from the example — edit it before: sudo supervisorctl start vla_$n"
    fi
  done
  chmod +x "$REPO"/scripts/service_teleop.sh "$REPO"/scripts/service_eval.sh
fi

sudo cp "$SRC"/*.conf "$CONF_D"/
sudo supervisorctl reread
sudo supervisorctl update
sudo supervisorctl status | grep -E 'vla_' || true
if [ "$HOST_DIR" = "rd" ]; then
  echo "manual-start apps: sudo supervisorctl start vla_teleop | vla_eval  (or the ▶ buttons on the manager's RobotDisco Teleop tab)"
fi
