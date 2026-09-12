"""Keyboard registrations that avoid Windows scan-code aliases."""

PRIMARY_DIGIT_SCAN_CODES = {
    "1": 2,
    "2": 3,
    "3": 4,
}


def register_primary_digit_hotkey(keyboard, digit, callback):
    """Register a mode digit using only its main-keyboard scan code.

    ``keyboard.key_to_scan_codes('3')`` also includes numpad-3 (81), which
    is the same scan code used by PageDown on Windows.
    """
    try:
        scan_code = PRIMARY_DIGIT_SCAN_CODES[digit]
    except KeyError as error:
        raise ValueError(f"unsupported primary digit: {digit}") from error
    return keyboard.add_hotkey(scan_code, callback)
