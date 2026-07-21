"""Search for an EE approach offset (relative to the block) that lifts the cube.

The AmazingHand's grasp aperture is offset from the flange axis, so a straight top-down
descent misses. This sweeps dx,dy offsets, runs approach->descend->close->lift, and
reports which offset actually raises the block. Run headless; renders the best attempt.
"""
import mujoco
import numpy as np
from PIL import Image

from ur5e_lerobot.sim import MujocoCell

cell = MujocoCell(settle_steps=8)
bid = mujoco.mj_name2id(cell.model, mujoco.mjtObj.mjOBJ_BODY, "block")
BX, BY = 0.0, 0.45
ori = cell.get_ee_pose()[3:]


def hold(x, y, z, curl, n):
    for _ in range(n):
        cell.set_ee_pose([x, y, z, *ori])
        cell.set_curls([curl] * 4)
        cell.step(8)


def attempt(dx, dy, zg):
    cell.reset()
    z0 = float(cell.data.xpos[bid][2])
    hold(BX + dx, BY + dy, 0.22, 0.0, 20)   # above, open
    hold(BX + dx, BY + dy, zg, 0.0, 30)     # descend, open
    hold(BX + dx, BY + dy, zg, 1.0, 45)     # close
    hold(BX + dx, BY + dy, 0.30, 1.0, 45)   # lift
    return float(cell.data.xpos[bid][2]) - z0


results = []
for dx in (-0.05, -0.025, 0.0, 0.025, 0.05):
    for dy in (-0.05, -0.025, 0.0, 0.025, 0.05):
        lift = attempt(dx, dy, 0.11)
        results.append((lift, dx, dy))
        print(f"dx={dx:+.3f} dy={dy:+.3f}  lift={lift:+.3f}")

results.sort(reverse=True)
best_lift, bdx, bdy = results[0]
print(f"\nBEST: dx={bdx:+.3f} dy={bdy:+.3f}  lift={best_lift:+.3f}")
attempt(bdx, bdy, 0.11)
Image.fromarray(cell.render(640, 540)).save("outputs/sim/grasp_best.png")
print("LIFTED" if best_lift > 0.05 else "STILL FAILING")
