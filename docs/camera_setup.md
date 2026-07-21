# Camera placement plan

For VLA / imitation learning on the UR5e + AmazingHand. Camera views ARE part of the
policy's observation space — placement, count, and stability directly bound how well a
policy can learn and generalize. Treat this as a schema decision (see D6/D7).

## Principles
1. **Scene (third-person) view** gives task context: full workspace, objects, robot.
2. **Wrist (eye-in-hand) view** gives occlusion-robust close-ups of the grasp — the
   single biggest accuracy win for manipulation policies. Essential for a dexterous
   hand where fingers + contact matter.
3. **A second third-person view** (side/oblique) disambiguates depth and covers
   occlusions the front camera misses.
4. **Fixed cameras must not move** within or between episodes of a dataset — the policy
   keys off the exact viewpoint. Rigid mounts; mark/tape tripod positions to restore.

## Recommended rig (tiered — scale to what you own)

| Tier | Cameras | When |
|---|---|---|
| **Minimum** | C1 scene + C2 wrist | smallest viable VLA dataset |
| **Recommended** | C1 scene + C2 wrist + C3 side | default; good depth/occlusion coverage |
| **Ideal** | + C4 top-down | adds planar pick-place clarity |

### C1 — Scene / front camera  → **Intel RealSense (RGB-D)**
- Static, on a tripod/frame ~0.6–1.0 m from workspace center.
- Elevated ~0.4–0.6 m above the table, pitched **30–45° down**, front or front-3/4.
- Must frame the **entire reachable workspace + margin** and see the hand through the
  whole trajectory. Use a RealSense here so depth aids scene 3D understanding.
- If you have a D435/D455 → best as this scene cam (mid-range depth).

### C2 — Wrist / eye-in-hand camera  → **webcam, or RealSense D405 if you have one**
- Mounted on the UR5e wrist-3 link / tool flange via a bracket, **offset above & behind
  the AmazingHand**, looking forward-down along the approach axis so it sees the
  **fingertips and grasp point** without the hand fully occluding the view (~10–20 cm).
- D405 is purpose-built for close-range eye-in-hand → ideal here if available; otherwise
  a wide-FOV webcam is fine (RGB-only is acceptable for the wrist).
- Route the cable along the arm with strain relief; don't let the moving arm yank it.

### C3 — Side / second third-person camera  → **webcam**
- ~90° offset from C1, similar elevation, oblique angle. Pure RGB is fine.

### C4 — Top-down camera (optional)  → **webcam or 2nd RealSense**
- Directly overhead looking straight down. Strong for tabletop pick-and-place (clean XY).
- Note: two RealSense units can have **IR-emitter interference** — angle them apart or
  disable one emitter.

## Sim ↔ real parity (critical)
- Define **C2 (wrist) as a fixed camera link/joint in the UR5e URDF/xacro** with the
  measured extrinsic, so Gazebo/Isaac render the **identical** viewpoint to the real cam.
- Place fixed cameras (C1/C3/C4) in the sim scene at their measured base-frame poses.
- Keep resolution/FOV/aspect matched between sim and real to minimize the visual gap.

## Calibration
- **Wrist cam (eye-in-hand):** hand-eye calibration → camera→flange transform. Use
  `easy_handeye2` (ROS 2) with a Charuco/ArUco board.
- **Fixed cams (eye-to-hand):** board mounted on the gripper → camera→base transform.
- Strictly required only if you log poses in a common frame, fuse depth, or want sim
  parity / 3D — a pure image→action policy can train without extrinsics, but you'll
  want them for sim-to-real.

## Recording specs
- Capture native (e.g. 640×480 or 1280×720); policies usually downsample to 224–256.
- **Same FPS across all cameras** (e.g. 30 Hz), timestamp-aligned. LeRobot stores
  per-camera streams in the dataset.
- **USB bandwidth:** multiple RealSense + webcams can saturate a single USB3 controller
  (esp. with depth). Put each RealSense on its own USB3 bus/host; prefer MJPEG for
  webcams; use powered hubs. Watch for dropped-frame / "no frames" errors.

## Lighting & consistency
- Consistent, diffuse lighting; avoid backlighting the RealSense IR.
- Keep backgrounds/lighting stable **within** a dataset. Domain randomization
  (varied light/background/minor cam jitter) is a later step for generalization, not
  for the first datasets.

## Diagrams

Plan (top-down):
```
                         front / operator
   ┌──────────────────────────────────────────────────┐
   │   (C3) side webcam                                 │
   │        ◐ ─────┐                                    │
   │                ▼                                   │
   │            ┌────────────┐                          │
   │            │  workspace │ ◀────── ◐ (C1) front     │
   │            │  ▢ objects │         scene RealSense   │
   │            │   ⬤ TCP+   │         (~0.8 m, 40° down)│
   │            │     hand   │                          │
   │            └────────────┘                          │
   │                 ▣ UR5e base                        │
   └──────────────────────────────────────────────────┘
```

Side (elevation):
```
   (C1) scene cam ◐ 40° down, ~0.5 m high
        \
         \            (C2) wrist cam ◐  (on wrist-3, looks down at fingers)
          \              \
           \             [UR5e arm]
            \              \
             \           (AmazingHand)
   ═══════════╪══════════════⬤════════════════  table
             base          workspace
```

## Open questions
- Exact inventory: how many RealSense units (and which models — D405 vs D435/D455?) and
  how many webcams? This sets the tier and the C1/C2 assignment.
- Workspace footprint + where the UR5e is mounted (table edge vs pedestal) — drives C1
  distance/height.
