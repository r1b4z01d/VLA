#!/usr/bin/env bash
# Install the VLA supervisor units on the GPU box (mirrors rd_ws/Scripts/install_services.sh).
#
#   ./scripts/install_services.sh
#
# vla_webui       autostart=true  -> the datasets/models/evals dashboard, port 8081
# vla_infer_server autostart=false -> start it only when driving the robot (it pins GPU memory):
#                                     sudo supervisorctl start vla_infer_server
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONF_D=/etc/supervisor/conf.d

[ -d "$CONF_D" ] || { echo "no $CONF_D — is supervisor installed?" >&2; exit 1; }

mkdir -p "$REPO/log"   # the units log here; supervisor won't create it and fails to start without it

# The served checkpoint is an indirection so the unit file never has to change. Point it at something
# on first install if it isn't set up yet.
if [ ! -e "$REPO/outputs/serve_ckpt" ]; then
  echo "note: $REPO/outputs/serve_ckpt does not exist — vla_infer_server will fail until you set it:"
  echo "  ln -sfn $REPO/outputs/train/<run>/checkpoints/last/pretrained_model $REPO/outputs/serve_ckpt"
fi

sudo cp "$REPO"/services/*.conf "$CONF_D"/
sudo supervisorctl reread
sudo supervisorctl update
sudo supervisorctl status | grep -E 'vla_' || true
