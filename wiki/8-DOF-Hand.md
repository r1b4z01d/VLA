# 8-DOF Hand (per-finger flex + abduct) — in progress

Adding palm-back **buttons (PCF8574 on the ESP32 I²C bus)** for per-finger **flex *and* abduct**, recorded
as a **14-D** action. Groundwork done: `schema.ActionV2` (14-D) + `STATE_NAMES_V2` / `STATE_DIM_V2`
(20-D), and `amazing_hand_client` `send_flex_abduct` / `read_state` (the `abduct=0` path is bit-identical
to today's curls). The live transport is UDP `:8766` (TCP `:8765` is the legacy path); the firmware +
protocol (`F:` command, `S:` report) spec for the 8-DOF extension:
**[docs/amazinghand_esp32_buttons.md](https://github.com/r1b4z01d/VLA/blob/main/docs/amazinghand_esp32_buttons.md)**.
The manager's **Hand** tab already drives flex + abduct per finger over UDP (sliders, presets, the
firmware's gesture characters, a raw-line box), see [Web Manager](Web-Manager#hand). A panel
"hand-buttons" recording source lands when the board is wired + flashed. The live pipeline stays
**10-D**; v2 is opt-in.
