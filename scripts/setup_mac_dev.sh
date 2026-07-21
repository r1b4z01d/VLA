#!/usr/bin/env bash
# Phase 0 dev environment on macOS (Apple Silicon) — learn LeRobot, no hardware.
# Uses uv to manage a Python 3.10 venv. Heavy install (PyTorch etc.); run deliberately.
set -euo pipefail

cd "$(dirname "$0")/.."

if ! command -v uv >/dev/null 2>&1; then
  echo "uv not found. Install: https://docs.astral.sh/uv/getting-started/installation/"
  exit 1
fi

# LeRobot needs Python >=3.10. The system python is 3.9, so pin a fresh one via uv.
uv python install 3.10
uv venv --python 3.10 .venv
# shellcheck disable=SC1091
source .venv/bin/activate

echo ">>> Installing LeRobot (this pulls PyTorch — several GB)…"
uv pip install lerobot
# numpy pin noted by recent LeRobot install docs:
uv pip install "numpy==1.26.0"

# Optional extras (uncomment what you need; check current extras in LeRobot docs):
# uv pip install "lerobot[smolvla]"   # SmolVLA policy
# uv pip install "lerobot[pi0]"       # pi0 / pi0-fast

cat <<'EOF'

✅ Mac dev env ready. Activate with:  source .venv/bin/activate

Phase 0 smoke test (toy sim, runs on CPU/MPS):
  - Browse a dataset:   https://huggingface.co/docs/lerobot/getting_started
  - Visualize / run a pretrained PushT or ALOHA policy in a gym env.
  - Then do one record -> train -> eval loop on a small task.

Note: VLA *training* (pi0/SmolVLA) belongs on the NVIDIA workstation, not here.
EOF
