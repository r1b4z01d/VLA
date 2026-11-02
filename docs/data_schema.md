# Dataset schema (action / observation)

The contract shared by the teleop panel, the `URAmazingHand` LeRobot adapter, the simulator, the
inference server and every recorded dataset. Code: [`ur5e_lerobot/schema.py`](../ur5e_lerobot/schema.py).
Two layouts: **V1** (live, every dataset so far) and **V2** (the 8-DOF hand, opt-in).

## V1 (live)
**Action — 10-D**
| idx | name | unit | frame |
|---|---|---|---|
| 0–2 | `ee_x, ee_y, ee_z` | m | UR **base frame** |
| 3–5 | `ee_rx, ee_ry, ee_rz` | rad | rotation vector (axis-angle) of the TCP in the base frame |
| 6–9 | `curl_index, curl_middle, curl_ring, curl_thumb` | — | [0, 1], 0 = open, 1 = closed |

**State — 16-D**: `ee_pose[6] + arm_joints q1..q6 [6] + hand_curls[4]`.

- **Absolute targets**, not deltas: the teleop integrates SpaceMouse / gamepad deltas into an absolute
  EE target, the policy predicts absolute targets as chunks. (`scripts/make_delta_dataset.py` converts a
  dataset to deltas for experiments; `scripts/train_nostate.py` drops the state.)
- Rotation is stored as axis-angle; the policy input may use a 6-D rotation instead (a transform, not
  a schema change).
- **Arm execution**: `robot/rtde_arm.py` — the target goes to a 50 Hz **servoJ** streaming thread with
  one controller IK per hand-over (or DLS IK, `eval_hw.py --ik-mode`), with the no-go / max-step /
  z-floor clamps in the command path. No ROS, no MoveIt.
- **Hand execution**: 4 curls → `AmazingHandClient.send_curls` → 8 servo offsets → one `J:` line per
  step over **UDP to 192.168.11.117:8766** (TCP :8765 is the legacy transport).
- **Curl channel ↔ finger**: on this hand channel 0 moves the middle finger, 1 the ring/pinky, 2 the
  index, 3 the thumb (measured 2026-10-26). The column names are left as they are (they are the
  dataset's action names); the Stream Deck keys and the manager's Hand tab remap number → channel so a
  key does what its label says.

## V2 (8-DOF hand, opt-in)
Enabled once the hand reports per-finger flex **and** abduction (`docs/amazinghand_esp32_buttons.md`).
**Action — 14-D** `ActionV2`: `ee_pose[6] + flex[4] in [0,1] + abduct[4] in [-1,1]`
(`ACTION_NAMES_V2`). **State — 20-D**: `ee_pose[6] + arm_joints[6] + flex[4] + abduct[4]`
(`STATE_NAMES_V2`, `STATE_DIM_V2`). Kept separate from V1 so existing datasets and checkpoints stay valid.

## Observation
Three image streams from the camera service (`camera_setup.md`):
- `observation.images.scene` — 540×540, third-person, square crop of the portrait frame
- `observation.images.side` — 540×540, second third-person
- `observation.images.wrist` — 540×960, eye-in-hand, uncropped

No depth is recorded. Training for SmolVLA renames them to `camera1/2/3` (`--rename_map`), the keys
`smolvla_base` was trained on.

## LeRobot features (as recorded)
```python
features = {
    "observation.images.scene": {"dtype": "video", "shape": (540, 540, 3), "names": ["height", "width", "channel"]},
    "observation.images.side":  {"dtype": "video", "shape": (540, 540, 3), "names": ["height", "width", "channel"]},
    "observation.images.wrist": {"dtype": "video", "shape": (540, 960, 3), "names": ["height", "width", "channel"]},
    "observation.state": {"dtype": "float32", "shape": (16,), "names": STATE_NAMES},
    "action":            {"dtype": "float32", "shape": (10,), "names": ACTION_NAMES},
}
```
Storage is **video** (MP4) by default since 2026-10-26 (`--png` for frames; ~25× larger). Rate: **12 fps**
for recording and eval (the 50 Hz servo stream interpolates between steps).

## Sidecars (outside the LeRobot schema)
- `meta/capture.json` (`sensors/capture_meta.py`, version 1): `squares {scene, side, wrist}`, `rotate`,
  `capture_box [960, 540]`, `shapes` per camera, `recorded {first, last}`, `host`. Written when the
  panel creates a dataset, re-applied on resume, copied into the training run, reported by the server,
  applied by the eval. A merged dataset does not get one — `scripts/capture_backfill.py` adds it.
- `meta/annotations.json` (`webui/annotations.py`): per episode `rating` (1-5), `notes`, `operator`,
  `grasps` (re-grasp marks), `regrasp_ranges` (frame ranges the trainer oversamples), `exposure`
  (`{camera: value}` locked at launch, 100 µs units). Remapped through the new episode indices on
  every rebuild (delete / trim / merge / move); a sub-episode trim does not yet shift `regrasp_ranges`.
- Eval runs: `outputs/evals/<run>/eval.json` (checkpoint, server flags, task, landmark, steps, timing,
  rating, notes, operator) + `log.csv` + `rollout.mp4`.

## Deferred
- Force/torque logging (the UR has a wrist F/T estimate).
- Logging the 8 raw servo offsets next to the 4 curls.
