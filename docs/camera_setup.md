# Cameras on the robot

Three identical wide-FOV USB webcams make up the policy's observation; the RealSense D435 on the base is
used by ROS (AprilTag relocalization, the landmark photos, the map pages' feed), never by the policy.
The planning-era tiers (RealSense scene cam, top-down cam, hand-eye calibration) were dropped: a pure
image → action policy needs a *stable* viewpoint, not extrinsics.

## Placement
| key | mount | rotation | stored frame |
|---|---|---|---|
| `scene` | 3rd-person, on the robot's right ("left eye"), sideways | 270° | 540×540 square crop of the portrait frame |
| `side` | 2nd 3rd-person, on the robot's left ("right eye"), sideways | 90° | 540×540 square crop |
| `wrist` | eye-in-hand on the UR5e flange, behind the AmazingHand, fingers at the top of the frame | 0° | 540×960 landscape, uncropped |

Fixed cameras must not move within or between episodes of a dataset: the policy keys off the exact
viewpoint. The crop is part of that geometry (below).

## Addressing: USB port, not device index
Same model, no serial, so `sensors/cameras.py` opens them by **USB-port `by-path`**. On the Jetson Orin
(final cabling, 2026-10-30): `scene` on port **1.1** (the dock on USB-C F), `wrist` on **2** (the other
USB-C, alone), `side` on **4.2** (a USB-A jack). One USB 2 webcam per bus: all four USB-A jacks share
one bus, so the three cameras need the three separate buses or they starve each other. Photos and the
kernel map are in `rd_ws/provision/jetson/`. Re-plugging into another port changes the path; the
constants `_PORTS` in `cameras.py` are the record.

## Capture and crop
Capture is **1280×720 MJPG**. The scene and side cameras are mounted sideways, so after rotation they are
720×1280 portrait and a **square crop** picks the 720×720 window: `SCENE_SQUARE` / `SIDE_SQUARE` =
fraction along the long axis (0.40 keeps the shelf, 1.0 the floor). Frames are then fit into the
960×540 capture box, which is how the stored 540×540 / 540×960 shapes come about. The crop is recorded
with the dataset in `meta/capture.json` (squares, rotations, capture box, shapes, recording dates, host),
carried into `outputs/train/<run>/capture.json` by training, reported by the inference server and
applied by the eval before its first observation, so a checkpoint always runs on the framing it trained
on. `scripts/camera_viewer.py --host 192.168.11.80 --squares` reads the live crop, `--square scene 0.65`
moves it (not persisted); the teleop panel's Crop card does the same until the dataset exists.

## The camera service (`scripts/camera_server.py`, unit `vla_cameras`)
One process owns the three cameras (`--fps 15 --quality 90 --iface-ip 192.168.11.80`) and publishes
JPEG frames over **multicast 239.42.11.1:5600** (1400-byte chunks, a `VCAM` header); a **control port
:5601** (crop, exposure, restart) and an **MJPEG server :5602** (`http://192.168.11.80:5602/`:
`/<cam>.mjpg`, `/<cam>.jpg`, `/<cam>.raw.jpg` = the full upright frame with the crop square drawn).
The teleop panel, the eval, the ROS bridge and any browser subscribe at once (`--cameras udp`), nothing
else opens the devices. Recording and eval run at **12 fps** (the panel's default). A camera that drops
off USB is re-opened by the service; a client that sees no frames restarts the service once and retries.
`scripts/camera_ros_bridge.py` (unit `vla_camera_ros_bridge`, system python with rclpy) republishes the
same frames as `/vla/cameras/<scene|side|wrist>/image/compressed` for the map pages and Foxglove.

## Exposure
Auto-exposure meters the whole sensor frame, which the crop mostly discards, so highlights in the
workspace clip. The panel and the eval **meter at launch and lock** (auto → lock; `--expose-manual`
hands the Stream Deck's first row to + / − / SET for a trim); the locked values go into each episode's
annotation (`expo` on the manager). `scripts/tune_exposure.py` sweeps exposure values and reports clip
% / mean / p1 / p99 on the frame as recorded, verifies the camera honours the control, and writes a
PNG per value; the winners go into `cameras.py` (`SCENE_AUTO_EXPOSURE, SCENE_EXPOSURE = 1.0, <value>`).
Gain is auto (`*_GAIN = None`).

## Lighting
Consistent, diffuse; keep background and lighting stable within a dataset. Vary the object, not the rig
(see `high_variety_recording.md`).
