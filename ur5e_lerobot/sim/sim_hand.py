"""SimHand — an AmazingHandClient-compatible backend backed by the MuJoCo sim.

Implements the subset of the hand-client interface that URAmazingHand uses
(`connect` / `close` / `send_curls`) so the adapter can run with the hand in sim and no
hardware. Also exposes `render()` to provide the camera observation.
"""
from __future__ import annotations

from .amazing_hand_mujoco import AmazingHandMujoco


class SimHand:
    def __init__(self, sim: AmazingHandMujoco | None = None, settle_steps: int = 15):
        # settle_steps controls how far the hand moves toward the target per command
        # (small => smooth motion across recorded frames).
        self.sim = sim or AmazingHandMujoco()
        self.settle_steps = settle_steps
        self._connected = False
        self.curls = [0.0, 0.0, 0.0, 0.0]

    def connect(self) -> "SimHand":
        self.sim.reset()
        self._connected = True
        return self

    def close(self) -> None:
        self._connected = False

    @property
    def is_connected(self) -> bool:
        return self._connected

    def send_curls(self, curls, speed=None) -> None:
        self.curls = list(curls)
        self.sim.set_curls(curls)
        self.sim.settle(self.settle_steps)

    def render(self, **kw):
        return self.sim.render(**kw)
