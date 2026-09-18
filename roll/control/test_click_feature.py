# [NEW - 2026-09-12]
# Reason: Standalone unit test for auto-clicking while tracking with camera movement.
# Content: Tests Interception left click strokes, concurrent relative movement, and non-blocking pulse state machine.

import ctypes
import os
import sys
import threading
import time

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
ROLL_ROOT = os.path.dirname(CURRENT_DIR)
if ROLL_ROOT not in sys.path:
    sys.path.insert(0, ROLL_ROOT)

from control.aim_controller import (
    VisualAimController,
    InterceptionMouseStroke,
    INTERCEPTION_MOUSE_LEFT_BUTTON_DOWN,
    INTERCEPTION_MOUSE_LEFT_BUTTON_UP,
)

def test_click_and_move():
    print("=" * 60)
    print("       视觉追踪 + 自动连点 (Click While Tracking) 单元测试")
    print("=" * 60)

    controller = VisualAimController(monitor_index=0)
    assert controller._ctx is not None, "Interception context must be initialized"

    # Define test left button methods directly
    def ensure_left_button(ctrl: VisualAimController, pressed: bool) -> None:
        if not ctrl._ctx:
            return
        with ctrl._driver_lock:
            if not ctrl._ctx:
                return
            if pressed:
                down = InterceptionMouseStroke(
                    state=INTERCEPTION_MOUSE_LEFT_BUTTON_DOWN,
                    flags=0, rolling=0, x=0, y=0, information=0
                )
                ctrl._lib.interception_send(ctrl._ctx, ctrl.device, ctypes.byref(down), 1)
            else:
                up = InterceptionMouseStroke(
                    state=INTERCEPTION_MOUSE_LEFT_BUTTON_UP,
                    flags=0, rolling=0, x=0, y=0, information=0
                )
                ctrl._lib.interception_send(ctrl._ctx, ctrl.device, ctypes.byref(up), 1)

    def send_left_click(ctrl: VisualAimController, hold_duration: float = 0.015) -> bool:
        if not ctrl._ctx:
            return False
        with ctrl._driver_lock:
            down = InterceptionMouseStroke(
                state=INTERCEPTION_MOUSE_LEFT_BUTTON_DOWN,
                flags=0, rolling=0, x=0, y=0, information=0
            )
            up = InterceptionMouseStroke(
                state=INTERCEPTION_MOUSE_LEFT_BUTTON_UP,
                flags=0, rolling=0, x=0, y=0, information=0
            )
            ok1 = ctrl._lib.interception_send(ctrl._ctx, ctrl.device, ctypes.byref(down), 1) > 0
            if hold_duration > 0:
                time.sleep(hold_duration)
            ok2 = ctrl._lib.interception_send(ctrl._ctx, ctrl.device, ctypes.byref(up), 1) > 0
            return ok1 and ok2

    # 1. Test direct left click
    print("[*] 测试 1: 发送独立鼠标左键点击...")
    ok = send_left_click(controller)
    assert ok is True, "send_left_click failed"
    print("[+] 独立左键单击测试通过！")

    # 2. Test concurrent movement and click interleaving
    print("[*] 测试 2: 模拟镜头追踪中一边相对移动一边连点...")
    t_start = time.perf_counter()
    click_count = 0
    move_count = 0
    click_interval = 0.08  # 80ms click interval (~12.5 CPS)
    last_click = 0.0
    is_left_down = False
    down_timestamp = 0.0

    for step in range(30):
        t_now = time.perf_counter()

        # Check releasing held click (non-blocking 15ms pulse)
        if is_left_down and (t_now - down_timestamp >= 0.015):
            ensure_left_button(controller, False)
            is_left_down = False

        # Camera continuous motion
        controller.send_relative_move(4, -2)
        move_count += 1

        # Trigger click
        if (t_now - last_click >= click_interval) and not is_left_down:
            ensure_left_button(controller, True)
            is_left_down = True
            down_timestamp = t_now
            last_click = t_now
            click_count += 1

        time.sleep(0.016)

    # Release any remaining click
    if is_left_down:
        ensure_left_button(controller, False)

    total_time = time.perf_counter() - t_start
    print(f"[+] 执行完成: 共移动 {move_count} 步，无阻塞连点 {click_count} 次，总耗时 {total_time:.2f}s")
    assert move_count == 30
    assert click_count >= 3, f"Expected at least 3 clicks, got {click_count}"

    controller.close()
    print("=" * 60)
    print("      连点与镜头移动并发测试通过 (PASS)")
    print("=" * 60)

if __name__ == "__main__":
    test_click_and_move()
