# Dataset schema (action / observation)

Resolves decision D6. This is the contract shared by the teleop layer, the LeRobot
`Robot` adapter, the simulator, and the recorded dataset. Code form:
[`ur5e_lerobot/schema.py`](../ur5e_lerobot/schema.py).

## Action — 10-D
| idx | name | unit | range / frame |
|---|---|---|---|
| 0–2 | `ee_x, ee_y, ee_z` | m | UR **base frame** |
| 3–5 | `ee_rx, ee_ry, ee_rz` | rad | rotation vector (axis-angle) of TCP in base frame |
| 6–9 | `curl_index, curl_middle, curl_ring, curl_thumb` | — | [0, 1], 0=open 1=closed |

- **Absolute targets**, not deltas. Teleop integrates SpaceMouse deltas into an absolute
  EE target; the policy predicts absolute targets (or action chunks). (Delta-EE is the
  fallback if absolute proves hard to learn.)
- **Rotation:** stored as axis-angle (compact, human-readable). Convert to a 6D rotation
  representation at the *model input* for continuity (better for learning) — a transform,
  not a schema change.
- Arm execution: EE target → IK via **MoveIt Servo** / Cartesian controller (robot side).
- Hand execution: 4 curls → `AmazingHandClient.send_curls` → 8 servo offsets → TCP.

## Observation
**Images** (keys per [`camera_setup.md`](camera_setup.md); scene+wrist minimum):
- `observation.images.scene` — RealSense RGB, third-person.
- `observation.images.wrist` — eye-in-hand RGB.
- `observation.images.side`  — (recommended tier) second third-person RGB.
- *(optional)* `observation.images.scene_depth` — RealSense depth.

**Proprioceptive state** — `observation.state`, 16-D:
`ee_pose[6] + arm_joints q1..q6 [6] + hand_curls[4]`.

## LeRobot `features` (sketch — finalize against the pinned LeRobot version)
```python
features = {
    "observation.images.scene": {"dtype": "video", "shape": (H, W, 3),
                                  "names": ["height", "width", "channel"]},
    "observation.images.wrist": {"dtype": "video", "shape": (H, W, 3),
                                  "names": ["height", "width", "channel"]},
    "observation.images.side":  {"dtype": "video", "shape": (H, W, 3),
                                  "names": ["height", "width", "channel"]},
    "observation.state": {"dtype": "float32", "shape": (16,), "names": STATE_NAMES},
    "action":            {"dtype": "float32", "shape": (10,), "names": ACTION_NAMES},
}
```

## Rates & sync
- Control / record rate: **20–30 Hz** (hand demo streams at 20 Hz; arm servo can go
  higher). All cameras at the same FPS, timestamp-aligned (see camera_setup.md).

## Deferred
- Force/torque logging (UR has a wrist F/T estimate) — add later for contact-rich tasks.
- Whether to also log raw 8 servo offsets alongside the 4 curls (cheap; may help debug).
