# [NEW - 2026-09-12]
# Reason: Unit test for hotkey definitions and key checking logic.
# Content: Verifies ord('N') == 0x4E, GetAsyncKeyState key detection, and key code mapping.

import ctypes

def is_key_pressed(vk_code: int) -> bool:
    """Check if a virtual key is currently pressed using GetAsyncKeyState."""
    return bool(ctypes.windll.user32.GetAsyncKeyState(vk_code) & 0x8000)

VK_F1 = 0x70
VK_KEY_N = ord('N')  # 0x4E = 78
VK_F2 = 0x71
VK_F3 = 0x72
VK_ESCAPE = 0x1B

def test_hotkey_constants():
    assert VK_KEY_N == 0x4E, f"VK_KEY_N must be 0x4E, got {hex(VK_KEY_N)}"
    assert ord('N') == 78
    res = is_key_pressed(VK_KEY_N)
    assert isinstance(res, bool)
    print(f"[PASS] VK_KEY_N = {hex(VK_KEY_N)} ({VK_KEY_N}) is valid. is_key_pressed returned: {res}")

if __name__ == "__main__":
    test_hotkey_constants()
