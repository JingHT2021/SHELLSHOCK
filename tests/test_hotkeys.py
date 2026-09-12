from shellshock.interaction.hotkeys import register_primary_digit_hotkey


class RecordingKeyboard:
    def __init__(self):
        self.calls = []

    def add_hotkey(self, hotkey, callback):
        self.calls.append((hotkey, callback))


def test_digit_mode_hotkeys_use_primary_keyboard_scan_codes():
    keyboard = RecordingKeyboard()
    callback = object()

    register_primary_digit_hotkey(keyboard, "3", callback)

    assert keyboard.calls == [(4, callback)]
