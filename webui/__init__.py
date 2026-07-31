"""webui — a browser console to manage the VLA project's LeRobot datasets.

Dataset explorer (list / merge / delete / add / folder-monitor), episode manager
(trim / delete / rating / notes / operator / rename-task / move), and an episode
player that embeds lerobot's rerun web viewer.

Runs on the GPU box; launched as:
    HF_HUB_OFFLINE=1 PYTHONPATH=. .venv/bin/python -m webui.server --host 0.0.0.0 --port 8080

The heavy lifting is delegated to lerobot.datasets.dataset_tools (delete/merge/rename/to-video)
and lerobot.scripts.lerobot_dataset_viz (the player); we never hand-roll dataset surgery except
sub-episode trim, which has no library equivalent (see ops.trim_episode).

Note: intentionally NO `from __future__ import annotations` here — in a package __init__ it binds the
name `annotations` to a __future__._Feature, which then shadows the `webui.annotations` SUBMODULE for
`from . import annotations` in sibling modules. The submodules keep their own future-import safely.
"""
import os

os.environ.setdefault("HF_HUB_OFFLINE", "1")  # local-only datasets — never hit the HF hub
