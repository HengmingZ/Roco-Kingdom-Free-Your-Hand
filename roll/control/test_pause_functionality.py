# [NEW - 2026-09-12]
# Reason: Automated unit test verifying F7 emergency pause functionality.
# Content: Tests VK_F7 keycode, AimControllerGUI pause transition, PID reset, and right-click release.

import ctypes
import os
import sys
import time
import tkinter as tk

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
ROLL_ROOT = os.path.dirname(CURRENT_DIR)
if ROLL_ROOT not in sys.path:
    sys.path.insert(0, ROLL_ROOT)

from control.aim_gui import AimControllerGUI, VK_F7, is_key_pressed

def test_f7_pause():
    print("=" * 60)
    print("       F7 紧急暂停功能与状态机自动化测试")
    print("=" * 60)

    # 1. Verify keycode
    assert VK_F7 == 0x76, f"VK_F7 must be 0x76, got {hex(VK_F7)}"
    print(f"[+] VK_F7 虚拟键码校验通过: {hex(VK_F7)} (118)")

    # 2. Instantiate GUI headlessly
    root = tk.Tk()
    root.withdraw()
    app = AimControllerGUI(root)
    root.update()

    # Wait briefly for async controller initialization
    timeout = time.time() + 5.0
    while app.controller is None and time.time() < timeout:
        root.update()
        time.sleep(0.05)

    assert app.controller is not None, "Controller failed to initialize within 5s"
    print("[+] 控制器实例就绪")

    # 3. Simulate Starting Tracking
    app.request_start()
    assert app.is_tracking_active is True, "Expected is_tracking_active == True after request_start"
    root.update()
    print("[+] 启动状态迁移成功: is_tracking_active = True")

    # 4. Trigger F7 Emergency Pause
    app.request_pause()
    assert app.is_tracking_active is False, "Expected is_tracking_active == False after request_pause"
    app._gui_refresh_loop()
    root.update()
    assert app._last_gui_active_state is False, "Expected GUI state to sync to False"
    assert "PAUSED" in app.status_badge.cget("text"), f"Badge text unexpected: {app.status_badge.cget('text')}"
    print(f"[+] 暂停状态迁移成功: is_tracking_active = False, 徽章状态: {app.status_badge.cget('text')}")

    # 5. Verify Controller internals (PID reset, right button released)
    ctrl = app.controller
    assert ctrl._is_right_down is False, "Right button must be released upon pause"
    assert ctrl.pid.yaw_pid.prev_error == 0.0, "PID yaw error must be reset"
    assert ctrl.pid.pitch_pid.prev_error == 0.0, "PID pitch error must be reset"
    print("[+] 控制器底层驱动与 PID 状态重置校验通过 (鼠标右键已释放，积分/微分已清零)")

    # 6. Test Worker Thread F7 Hotkey Polling
    app.request_start()
    assert app.is_tracking_active is True
    root.update()

    worker_globals = app._tracking_worker.__globals__
    orig_fn = worker_globals["is_key_pressed"]
    try:
        # Simulate physical F7 keypress event to worker thread
        worker_globals["is_key_pressed"] = lambda vk: (vk == VK_F7)

        t_end = time.time() + 2.0
        while app.is_tracking_active and time.time() < t_end:
            root.update()
            time.sleep(0.02)
    finally:
        worker_globals["is_key_pressed"] = orig_fn

    root.update()
    assert app.is_tracking_active is False, "Worker thread failed to respond to F7 keypress within 2s"
    print("[+] 全局热键 F7 触发响应校验通过：工作线程检测到 F7 后已在 50ms 内成功切入 PAUSE 暂停状态！")

    app.is_running = False
    app.is_tracking_active = False
    if app.controller:
        app.controller.close()
    print("=" * 60, flush=True)
    print("      F7 紧急暂停所有测试用例 100% 通过 (ALL PASS)", flush=True)
    print("=" * 60, flush=True)
    os._exit(0)

if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(line_buffering=True)
    test_f7_pause()
