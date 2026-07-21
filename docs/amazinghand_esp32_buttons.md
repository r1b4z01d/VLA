# AmazingHand — per-finger buttons (PCF8574) + flex/abduct control + recording

Spec for adding **8 finger buttons + 1 mode switch** to the palm-back and recording the resulting
**8-DOF** hand motion (per-finger flex **and** abduct). Host side is already implemented; this
documents the **ESP32 firmware** + the **`:8765` protocol** it must speak.

Controller: **Waveshare "Servo Driver with ESP32"** (ESP32-WROOM-32E). Pins already used:
`GPIO18/19` Feetech bus (`Serial1` @1M), `GPIO21/22` I²C (OLED), `GPIO23` RGB×2, `GPIO1/3` USB log,
`GPIO16/17` GM60 scanner (`Serial2`), WiFi internal.

## 1. Hardware

**PCF8574** (8-bit I²C expander) on the existing I²C bus — 8 finger buttons:
- `VCC → 3V3`, `GND → GND`, `SDA → GPIO21`, `SCL → GPIO22`
- `A0/A1/A2 → GND` ⇒ address **0x20** (no clash with the OLED at 0x3C)
- `P0..P7 → button`, each button's other leg → `GND`  (pressed = logic 0)
- optional `/INT → a spare GPIO` for interrupt-driven reads; polling at the servo rate is fine
- short wires ⇒ the chip's weak quasi-pull-ups suffice; add 10 kΩ/line for margin if desired

**Mode switch** (9th input — the PCF8574 is full): global **flex ↔ abduct** toggle.
- Recommended: **momentary button on `GPIO0` (BOOT)** → firmware toggles the mode flag, shown on
  OLED/RGB. Zero extra wiring. ⚠️ momentary only — a *latching* switch held low at power-on puts
  the ESP32 into flash-download mode.
- Or a latching toggle on a non-strapping pin (`GPIO32`/`GPIO33`, internal pull-up) if you can reach a pad.

**Button map** (`P0..P7`), 2 per finger = `[+, −]`:
| pin | finger | action |
|---|---|---|
| P0 / P1 | index  | +  /  − |
| P2 / P3 | middle | +  /  − |
| P4 / P5 | ring   | +  /  − |
| P6 / P7 | thumb  | +  /  − |
`+`/`−` act on the **active DOF** (flex when mode=FLEX, abduct when mode=ABDUCT).

## 2. Firmware state machine

Maintain per-finger targets:
```
float flex[4]   in [0,1]   // 0 = open, 1 = closed
float abduct[4] in [-1,1]  // 0 = neutral
enum  mode = FLEX | ABDUCT // toggled by the mode button
```
Loop (~50–100 Hz):
1. Read PCF8574 (1 byte; bit=0 ⇒ pressed) + debounce (~15 ms stable).
2. For each finger `i`, while its `+`/`−` held: ramp the **active** DOF
   `flex[i] += RATE*dt` (or `abduct[i] += RATE*dt`), clamped to range. Suggested `RATE`: flex ~1.5/s,
   abduct ~1.0/s.
3. Mode button press → flip `mode`; reflect on OLED (`FLEX`/`ABD`) + RGB color.
4. Map to servos and drive the bus (see §3).
5. Stream the `S:` report (§4).

Host `F:`/`J:` commands write the **same** `flex[]/abduct[]` targets (last writer wins), so the hand
can be driven by buttons **or** by the host, and `S:` always reflects the truth.

## 3. Servo mapping (per finger: servo A = even index, B = odd)

```
flex_angle = lerp(JOINT_OPEN[A], JOINT_CLOSE[A], flex[i])   // A sweeps its open→close range
abd        = abduct[i] * 20.0                               // ±20° mechanism limit
A = flex_angle + abd
B = -flex_angle + abd
```
`JOINT_OPEN  = (-35, 35, -35, 35, -35, 35, -35, 35)`
`JOINT_CLOSE = ( 90,-90,  90,-90,  90,-90,  70,-70)`  (thumb closes to ±70)
This matches the host's `flex_abduct_to_offsets()`; with `abduct=0` it is exactly today's curl motion
(`flex=(A−B)/2`, `abduct=(A+B)/2`).

## 4. `:8765` TCP protocol

Newline-terminated ASCII. Existing command kept; two additions:

| dir | line | meaning |
|---|---|---|
| host→esp | `J:a0,…,a7,speed\n` | set raw 8 servo offsets (existing) |
| host→esp | `F:f0,f1,f2,f3,a0,a1,a2,a3,speed\n` | set targets: flex[0,1]×4, abduct[-1,1]×4 (**new**) |
| host→esp | `Q\n` | request one `S:` now (optional; else just stream) |
| esp→host | `S:f0,f1,f2,f3,a0,a1,a2,a3\n` | current targets, streamed ~20–50 Hz (**new**) |

`S:` is what makes button motion **recordable** — the host logs it as the hand action.

## 5. Host side (already implemented)

`ur5e_lerobot/hand/amazing_hand_client.py`:
- `send_flex_abduct(flex, abduct)` → emits `F:` (or `send_offsets(flex_abduct_to_offsets(...))` for `J:`-only firmware)
- `read_state()` → parses the latest `S:` → `(flex[4], abduct[4])`, non-blocking

`ur5e_lerobot/schema.py`: 14-D `ActionV2` = `[ee_pose(6), flex(4), abduct(4)]`, 20-D `STATE_V2`
(kept separate from the live 10-D v1).

**Recording model** (mirrors arm freedrive): in the panel's upcoming *hand-buttons* source, the hand
is driven **locally by the buttons**; the host `read_state()`s each tick and records `(flex, abduct)`
as the hand action, while the arm is teleoped (SpaceMouse/gamepad) as usual — no `F:`/`J:` sent for
the hand in that mode.

## 6. Bring-up checklist
1. `i2cdetect`-style scan → PCF8574 at 0x20, OLED still at 0x3C.
2. Buttons: print the raw byte; confirm each finger's ± and the mode toggle.
3. Servos: verify `abduct=0` reproduces the current curl open/close, then ±abduct spreads fingers ~±20°.
4. `S:` stream: `nc <hand-ip> 8765` → watch values track the buttons.
5. Host: `AmazingHandClient.read_state()` returns live `(flex, abduct)`; then wire the panel source.
