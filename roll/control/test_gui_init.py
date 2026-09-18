# [NEW - 2026-09-12]
# Reason: Headless initialization and lifecycle test for AimControllerGUI with auto-click controls.
import os
import sys
import tkinter as tk

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
ROLL_ROOT = os.path.dirname(CURRENT_DIR)
if ROLL_ROOT not in sys.path:
    sys.path.insert(0, ROLL_ROOT)

from control.aim_gui import AimControllerGUI

def test_gui():
    print("[*] 正在测试 AimControllerGUI 初始化与控件装配...")
    root = tk.Tk()
    root.withdraw()
    app = AimControllerGUI(root)
    assert app.var_auto_click.get() is True, "var_auto_click default should be True"
    assert app.val_click_interval.get() == 0.15, "val_click_interval default should be 0.15"
    assert app.var_click_locked_only.get() is False, "var_click_locked_only default should be False"
    print("[+] GUI 实例与自动连点控件装配成功！")
    app.on_closing()
    print("[+] GUI 优雅销毁通过 (PASS)")

if __name__ == "__main__":
    test_gui()
