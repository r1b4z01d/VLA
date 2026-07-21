"""Elgato Stream Deck (15-key) control surface for the teleop panel.

Maps deck keys to panel actions (Record / Save / Discard / Reconnect / Quit) and per-finger
open/close. Needs the `streamdeck` pip package + Pillow, and local-seat HID access (a udev rule like
the SpaceMouse). Linux / robot PC. Key callbacks fire on the lib's own thread, so the panel drains
them from a thread-safe queue inside its Tk loop (never touch Tk from the callback thread).

Test:  ~/VLA/run.sh -m ur5e_lerobot.teleop.streamdeck   (prints key up/down events)
"""
from __future__ import annotations


class StreamDeckPad:
    """Thin wrapper: open the deck, render text labels, and report key up/down via a callback."""

    def __init__(self, flip: bool = False):
        self.deck = None
        self.key_count = 0
        self.flip = flip  # deck mounted upside down -> rotate key images 180° + reverse the indices

    def _phys(self, key: int) -> int:
        """Map logical<->physical key. A 180° flip reverses the whole grid (its own inverse)."""
        return (self.key_count - 1 - key) if self.flip else key

    def open(self) -> "StreamDeckPad":
        from StreamDeck.DeviceManager import DeviceManager

        decks = DeviceManager().enumerate()
        if not decks:
            raise RuntimeError("no Stream Deck found (connected? udev/seat access? Linux only)")
        self.deck = decks[0]
        self.deck.open()
        self.deck.reset()
        self.deck.set_brightness(60)
        self.key_count = self.deck.key_count()
        return self

    def set_label(self, key: int, text: str, bg=(25, 25, 25), fg=(240, 240, 240)) -> None:
        if self.deck is None or not (0 <= key < self.key_count):
            return
        from PIL import Image, ImageDraw, ImageFont
        from StreamDeck.ImageHelpers import PILHelper

        img = PILHelper.create_image(self.deck, background=bg)
        draw = ImageDraw.Draw(img)
        try:
            font = ImageFont.truetype("DejaVuSans.ttf", 13)
        except Exception:  # noqa: BLE001
            font = ImageFont.load_default()
        lines = text.split("\n")
        w, h = img.size
        # vertically center the block of lines
        line_h = (font.getbbox("Ag")[3] - font.getbbox("Ag")[1]) + 3
        y = (h - line_h * len(lines)) // 2
        for ln in lines:
            bb = draw.textbbox((0, 0), ln, font=font)
            draw.text(((w - (bb[2] - bb[0])) // 2, y), ln, font=font, fill=fg)
            y += line_h
        if self.flip:
            img = img.transpose(Image.ROTATE_180)  # so labels read right-way-up on the flipped deck
        self.deck.set_key_image(self._phys(key), PILHelper.to_native_format(self.deck, img))

    def on_key(self, callback) -> None:
        """callback(key: int, down: bool), key in LOGICAL coords — fired on every press/release
        (on the lib thread; flip-remapped so the press matches the label the operator sees)."""
        self.deck.set_key_callback(lambda _deck, key, state: callback(self._phys(key), bool(state)))

    def close(self) -> None:
        if self.deck is not None:
            try:
                self.deck.reset()
                self.deck.close()
            except Exception:  # noqa: BLE001
                pass
            self.deck = None


def _main() -> None:
    import time

    pad = StreamDeckPad().open()
    print(f"opened Stream Deck with {pad.key_count} keys. Press keys (Ctrl-C to stop).")
    for k in range(pad.key_count):
        pad.set_label(k, str(k))
    pad.on_key(lambda key, down: print(f"key {key} {'DOWN' if down else 'up'}"))
    try:
        while True:
            time.sleep(0.1)
    except KeyboardInterrupt:
        pass
    finally:
        pad.close()


if __name__ == "__main__":
    _main()
