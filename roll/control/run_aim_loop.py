# [NEW - 2026-09-12]
# Reason: Real-time visual servoing PID aiming runner for RocoClicker roll module.
# Content: Continuous visual tracking loop, keyboard hotkey toggles (F1/F2/ESC),
#          console telemetry dashboard, and safety shutdown handling.

from __future__ import annotations

import argparse
import ctypes
from ctypes import wintypes
import os
import sys
import time

# UTF-8 console output
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
ROLL_ROOT = os.path.dirname(CURRENT_DIR)
PROJECT_ROOT = os.path.dirname(ROLL_ROOT)

if ROLL_ROOT not in sys.path:
    sys.path.insert(0, ROLL_ROOT)

from control.aim_controller import VisualAimController
from control.pid import PIDConfig


# [UPDATE - 2026-09-19]
# Reason: F7 emergency pause could be missed during quick taps due to missing 0x0001 bit check and undefined ctypes restype.
# Modification: Configured user32.GetAsyncKeyState argtypes and c_short restype, checking both 0x8000 and 0x0001.
_user32 = ctypes.windll.user32
_user32.GetAsyncKeyState.argtypes = [ctypes.c_int]
_user32.GetAsyncKeyState.restype = ctypes.c_short

VK_F1 = 0x70
VK_F7 = 0x76  # 118 (F7 key)
VK_KEY_N = ord("N")  # 0x4E (78)
VK_F2 = 0x71
VK_F3 = 0x72
VK_ESCAPE = 0x1B


def is_key_pressed(vk_code: int) -> bool:
    """Check if a virtual key is pressed globally via GetAsyncKeyState (held down or latched tap)."""
    try:
        val = _user32.GetAsyncKeyState(vk_code)
        return bool((val & 0x8000) or (val & 0x0001))
    except Exception:
        return False


def run_aim_runner(
    weights_path: str | None = None,
    monitor_index: int = 0,
    kp_x: float = 0.45,
    kp_y: float = 0.38,
    kd_x: float = 0.08,
    kd_y: float = 0.06,
    deadband: float = 4.0,
    hold_right: bool = True,
    conf_thresh: float = 0.35,
    auto_click: bool = True,
    click_interval: float = 0.15,
    click_when_locked: bool = False,
) -> None:
    print("=" * 65)
    print("   RocoClicker - 视觉伺服 PID 目标居中自动对准系统 (含连续锁定)")
    print("=" * 65)
    print(f"[*] 监控屏幕编号 : 屏幕 {monitor_index + 1}")
    print(f"[*] 模型权重文件 : {weights_path or 'roll/weights/best.pt'}")
    print(f"[*] PID 增益参数 : X(Kp={kp_x}, Kd={kd_x}) | Y(Kp={kp_y}, Kd={kd_y}) | 死区={deadband}px")
    print(f"[*] 转视角模式   : {'按住右键拖拽 (自由光标/UI模式)' if hold_right else '纯相对位移 (准星探索模式)'}")
    print(f"[*] 自动连点状态 : {'已开启 (间隔 ' + str(click_interval) + 's)' if auto_click else '已关闭'}{' [仅锁定死区触发]' if click_when_locked else ' [边转镜头边连点]'}")
    print("=" * 65)
    print("[控制快捷键说明]")
    print("  • 按 [F1] 键   : 启动 / 开启 PID 自动居中跟瞄")
    print("  • 按 [F7] 键   : 紧急暂停 / 挂起跟瞄 (释放鼠标，停止跟瞄)")
    print("  • 按 [F3] 键   : 切换目标 (强制重新锁定当前最靠近准星的目标)")
    print("  • 按 [ESC] 键  : 安全退出程序")
    print("=" * 65)

    cfg_x = PIDConfig(kp=kp_x, kd=kd_x, deadband=deadband)
    cfg_y = PIDConfig(kp=kp_y, kd=kd_y, deadband=deadband)

    controller = VisualAimController(
        weights_path=weights_path,
        monitor_index=monitor_index,
        pid_config_x=cfg_x,
        pid_config_y=cfg_y,
        hold_right_button=hold_right,
        conf_threshold=conf_thresh,
        auto_click=auto_click,
        click_interval=click_interval,
        click_only_when_locked=click_when_locked,
    )

    is_active = True
    force_reacquire = False
    print("\n[*] 控制器已就绪！当前状态: [RUNNING 跟瞄运行中]")
    print("[*] 请将游戏窗口切至前台对准目标场景...\n")

    frame_count = 0
    t_prev = time.perf_counter()
    fps = 0.0

    try:
        while True:
            # Hotkey polling
            if is_key_pressed(VK_F1):
                if not is_active:
                    is_active = True
                    controller.reset_target()
                    print("\n[+] [F1 触发] PID 自动对准已开启！           ")
                    time.sleep(0.2)
            # [UPDATE - 2026-09-12]
            # Reason: Emergency pause bound to 'F7' key (with N/F2 as fallbacks).
            # Modification: Check VK_F7, VK_KEY_N or VK_F2 for immediate disengagement.
            elif is_key_pressed(VK_F7) or is_key_pressed(VK_KEY_N) or is_key_pressed(VK_F2):
                if is_active:
                    is_active = False
                    controller.ensure_right_button(False)
                    controller.ensure_left_button(False)
                    controller.pid.reset()
                    print("\n[!] [F7 键触发] 紧急暂停：PID 自动对准已挂起！       ")
                    time.sleep(0.2)
            elif is_key_pressed(VK_F3):
                force_reacquire = True
                print("\n[↻] [F3 触发] 强制切换：重新捕获准星最近目标！")
                time.sleep(0.2)
            elif is_key_pressed(VK_ESCAPE):
                print("\n[!] [ESC 触发] 收到退出指令，正在终止...")
                break

            # Calculate FPS
            t_curr = time.perf_counter()
            frame_count += 1
            if t_curr - t_prev >= 0.5:
                fps = frame_count / (t_curr - t_prev)
                frame_count = 0
                t_prev = t_curr

            if is_active:
                step_res = controller.step(force_reacquire=force_reacquire)
                force_reacquire = False

                stat = step_res.status
                tid_str = f"ID:#{step_res.track_id}" if step_res.track_id else "ID:None"
                if stat == "LOCKED":
                    status_str = f"\033[92m[LOCKED  {tid_str}]\033[0m"
                elif stat == "TRACKING":
                    status_str = f"\033[93m[TRACK   {tid_str}]\033[0m"
                else:
                    status_str = "\033[90m[SEARCH  ID:None]\033[0m"

                # Telemetry dashboard on single line
                err_str = f"偏差:({step_res.error_x:+6.1f}, {step_res.error_y:+6.1f})px"
                ctrl_str = f"输出:(dx={step_res.dx:+3d}, dy={step_res.dy:+3d})"
                click_str = "\033[91m[CLICK!]\033[0m" if step_res.clicked else "\033[90m[------]\033[0m"
                time_str = f"耗时:{step_res.inference_time_ms:3.0f}+{step_res.control_time_ms:2.0f}ms"
                fps_str = f"FPS:{fps:4.1f}"

                print(
                    f"\r{status_str} {click_str} {err_str} | {ctrl_str} | {time_str} | {fps_str}   ",
                    end="",
                    flush=True,
                )
            else:
                print(f"\r[PAUSED 已暂停] 按 F1 开启，按 F3 换目标，按 ESC 退出...             ", end="", flush=True)
                time.sleep(0.05)

    except KeyboardInterrupt:
        print("\n[*] 用户中断 (Ctrl+C)，正在安全退出...")
    finally:
        controller.close()
        print("[+] Interception 驱动与鼠标按键已全部安全复位释放。程序退出。")


def main():
    parser = argparse.ArgumentParser(description="视觉伺服 PID 目标居中瞄准控制器")
    parser.add_argument("--weights", type=str, default=None, help="模型权重路径 (默认使用 roll/weights/best.pt)")
    parser.add_argument("--monitor", type=int, default=0, help="目标屏幕索引 (默认 0: 主屏幕)")
    parser.add_argument("--kp-x", type=float, default=0.45, help="水平轴 Proportional 增益")
    parser.add_argument("--kp-y", type=float, default=0.38, help="垂直轴 Proportional 增益")
    parser.add_argument("--kd-x", type=float, default=0.08, help="水平轴 Derivative 增益")
    parser.add_argument("--kd-y", type=float, default=0.06, help="垂直轴 Derivative 增益")
    parser.add_argument("--deadband", type=float, default=4.0, help="死区像素阈值 (默认 4.0 px)")
    parser.add_argument(
        "--no-right-click",
        action="store_true",
        help="禁用按住右键 (如果游戏处于无光标准星模式下请开启此选项)",
    )
    parser.add_argument("--conf", type=float, default=0.35, help="目标检测置信度阈值")
    parser.add_argument(
        "--no-auto-click",
        action="store_true",
        help="禁用自动连点 (仅进行镜头移动追踪)",
    )
    parser.add_argument(
        "--click-interval",
        type=float,
        default=0.15,
        help="自动连点触发间隔秒数 (默认 0.15s ≈ 6.7 CPS)",
    )
    parser.add_argument(
        "--click-when-locked",
        action="store_true",
        help="仅在准星锁定 (LOCKED) 在死区内时才触发连点 (默认: 边追踪边连点)",
    )
    args = parser.parse_args()

    run_aim_runner(
        weights_path=args.weights,
        monitor_index=args.monitor,
        kp_x=args.kp_x,
        kp_y=args.kp_y,
        kd_x=args.kd_x,
        kd_y=args.kd_y,
        deadband=args.deadband,
        hold_right=not args.no_right_click,
        conf_thresh=args.conf,
        auto_click=not args.no_auto_click,
        click_interval=args.click_interval,
        click_when_locked=args.click_when_locked,
    )


if __name__ == "__main__":
    main()
