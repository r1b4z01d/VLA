# Recording Sim Demos (teleop)

Drive the UR5e + AmazingHand in sim (MuJoCo) and record LeRobot datasets. **Prefer the workstation** — it
records a *video* dataset where you train (image datasets load far slower). Run on the box's own desktop;
`MUJOCO_GL=egl` renders headless on the 4090, `--use-videos` needs system FFmpeg (present on the box).
```bash
cd ~/VLA && .venv/bin/python -m ur5e_lerobot.teleop.spacemouse         # confirm the SpaceMouse reads
cd ~/VLA && MUJOCO_GL=egl .venv/bin/python -m ur5e_lerobot.teleop.manual_panel \
  --input spacemouse --use-videos --root outputs/datasets/teleop
```
Drop `--use-videos` for images; `--input sliders` for keyboard/mouse; `--resume` appends. On the Mac,
images only, and quit the 3Dconnexion 3DxWare driver first so the raw SpaceMouse is readable.

## Grasping & scene
The AmazingHand can't form a stable *physics* grasp in sim (4-finger geometry won't oppose), so a
**grasp aid welds the block to the hand on close and releases on open** — pick-place "just works" for
demo collection (`build_combined_model(grasp_aid=False)` to disable). Block = 4×4×9 cm on the floor →
green pad. Panel sim controls (hidden on `--engine hardware`): **Randomize Box/Goal** + a **ROYGBIV
box-color** dropdown. Collect *position variety*; a fixed position overfits (see
[Training & Eval](Training-and-Eval)).
