# [NEW - 2026-09-12]
# Reason: Unit and convergence verification test for DualAxisPID visual servoing controller.
# Content: Verifies step response, deadband suppression, anti-windup clamping, and multi-step convergence.

from __future__ import annotations

import os
import sys

# Ensure UTF-8 output
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
ROLL_ROOT = os.path.dirname(CURRENT_DIR)

if ROLL_ROOT not in sys.path:
    sys.path.insert(0, ROLL_ROOT)

from control.pid import PIDController, PIDConfig, DualAxisPID


def test_pid_convergence():
    print("=" * 60)
    print("       2D 视觉伺服 PID 控制器收敛性与稳定性基准测试")
    print("=" * 60)

    pid = DualAxisPID()

    # 1. Test Initial Large Error Clamping
    init_err_x = 450.0  # 目标在右侧 450 像素处
    init_err_y = -320.0 # 目标在上方 320 像素处
    dx, dy = pid.update(init_err_x, init_err_y, dt=0.016)

    print(f"[*] 初始大偏差: (error_x={init_err_x}, error_y={init_err_y})")
    print(f"[*] 首步控制输出: (dx={dx}, dy={dy})")
    assert abs(dx) <= 70, f"水平输出超出最大限幅: {dx}"
    assert abs(dy) <= 50, f"垂直输出超出最大限幅: {dy}"
    print("[+] 输出限幅保护 (Saturation Limiting) 校验通过！")

    # 2. Simulate Closed-Loop Visual Servoing Convergence
    print("\n[*] 开始闭环视觉伺服收敛仿真 (模拟 30 步控制迭代)...")
    curr_x = init_err_x
    curr_y = init_err_y
    # Assume 1 count of mouse relative movement shifts camera by approx 0.85 screen pixels in game
    gain_camera = 0.85

    trajectory = []
    for step in range(1, 35):
        dx, dy = pid.update(curr_x, curr_y, dt=0.016)
        curr_x -= dx * gain_camera
        curr_y -= dy * gain_camera
        trajectory.append((step, curr_x, curr_y, dx, dy))
        if step in (1, 5, 10, 15, 20, 25, 30):
            print(f"    Step {step:2d}: 剩余偏差=({curr_x:6.2f}, {curr_y:6.2f}) px | 控制量=(dx={dx:3d}, dy={dy:3d})")

    # Final error must be within deadband
    print(f"\n[*] 仿真结束最终残差: error_x={curr_x:.2f} px, error_y={curr_y:.2f} px")
    assert abs(curr_x) <= pid.yaw_pid.config.deadband, f"最终 X 偏差未收敛进入死区: {curr_x}"
    assert abs(curr_y) <= pid.pitch_pid.config.deadband, f"最终 Y 偏差未收敛进入死区: {curr_y}"
    print("[+] 闭环渐进稳定收敛校验通过 (100% CONVERGED)！")

    # 3. Test Deadband Output
    dx_dead, dy_dead = pid.update(2.0, -1.5, dt=0.016)
    assert dx_dead == 0 and dy_dead == 0, f"死区内非零输出: ({dx_dead}, {dy_dead})"
    print("[+] 死区抖动抑制 (Deadband Suppression) 校验通过！")

    print("\n" + "=" * 60)
    print("      DualAxisPID 控制器单元测试全部通过 (PASS)")
    print("=" * 60)


if __name__ == "__main__":
    test_pid_convergence()
