# [NEW - 2026-09-12]
# Reason: User requested a camera roll test in `roll/` folder to verify if simulated mouse
#         movements can rotate the 3D game camera in 《洛克王国：世界》.
# Content: Moves mouse to primary screen center, left-clicks once, then tests both:
#          1) Pure relative circular movement (for locked reticle / action camera mode).
#          2) Right-button drag circular movement (for free cursor / UI mode).

from __future__ import annotations

import argparse
import ctypes
from ctypes import wintypes
import math
import os
import sys
import time

# Ensure stdout and stderr handle UTF-8 properly on Windows consoles
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

class InterceptionMouseStroke(ctypes.Structure):
    _fields_ = [
        ("state", ctypes.c_ushort),
        ("flags", ctypes.c_ushort),
        ("rolling", ctypes.c_short),
        ("x", ctypes.c_int),
        ("y", ctypes.c_int),
        ("information", ctypes.c_uint),
    ]

# Interception Mouse States
INTERCEPTION_MOUSE_LEFT_BUTTON_DOWN   = 0x001
INTERCEPTION_MOUSE_LEFT_BUTTON_UP     = 0x002
INTERCEPTION_MOUSE_RIGHT_BUTTON_DOWN  = 0x004
INTERCEPTION_MOUSE_RIGHT_BUTTON_UP    = 0x008
INTERCEPTION_MOUSE_MIDDLE_BUTTON_DOWN = 0x010
INTERCEPTION_MOUSE_MIDDLE_BUTTON_UP   = 0x020

# Interception Flags
INTERCEPTION_MOUSE_MOVE_RELATIVE = 0x000
INTERCEPTION_MOUSE_MOVE_ABSOLUTE = 0x001

class InterceptionController:
    """Encapsulates kernel-level mouse injection via interception.dll."""

    def __init__(self, dll_path: str):
        if not os.path.exists(dll_path):
            raise FileNotFoundError(f"未找到 interception.dll: {dll_path}")

        self._lib = ctypes.CDLL(dll_path)
        self._lib.interception_create_context.restype = ctypes.c_void_p
        self._lib.interception_destroy_context.argtypes = [ctypes.c_void_p]
        self._lib.interception_send.argtypes = [
            ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p, ctypes.c_uint
        ]
        self._lib.interception_send.restype = ctypes.c_int
        self._lib.interception_is_mouse.argtypes = [ctypes.c_int]
        self._lib.interception_is_mouse.restype = ctypes.c_int

        self._ctx = self._lib.interception_create_context()
        if not self._ctx:
            raise RuntimeError("无法创建 Interception 上下文，请确认驱动已安装且以管理员权限运行！")

        # 扫描可用鼠标设备 (INTERCEPTION_MOUSE 范围 11~20)
        self.mouse_devices = [
            d for d in range(11, 21) if self._lib.interception_is_mouse(d)
        ]
        # 默认优先使用 device 11 (主物理鼠标)
        self.target_device = self.mouse_devices[0] if self.mouse_devices else 11

    def send_stroke(self, stroke: InterceptionMouseStroke) -> bool:
        if not self._ctx:
            return False
        res = self._lib.interception_send(self._ctx, self.target_device, ctypes.byref(stroke), 1)
        return res > 0

    def close(self):
        if self._ctx:
            self._lib.interception_destroy_context(self._ctx)
            self._ctx = None

def attach_desktop() -> None:
    """绑定当前线程到默认交互桌面，确保后台或脚本能正常获取与设置光标。"""
    u32 = ctypes.windll.user32
    hd = u32.OpenDesktopW("default", 0, False, 0x01FF)
    if hd:
        u32.SetThreadDesktop(hd)

def get_primary_screen_size() -> tuple[int, int]:
    """获取主显示器分辨率 (width, height)。"""
    u32 = ctypes.windll.user32
    return u32.GetSystemMetrics(0), u32.GetSystemMetrics(1)

def move_to_center(ctrl: InterceptionController) -> tuple[int, int]:
    """移动鼠标到主屏幕中心。"""
    u32 = ctypes.windll.user32
    attach_desktop()

    w, h = get_primary_screen_size()
    cx, cy = w // 2, h // 2
    print(f"[*] 主显示器分辨率: {w}x{h}，计算中心坐标: ({cx}, {cy})")

    # 1. 设置系统光标绝对位置
    u32.SetCursorPos(cx, cy)

    # 2. 注入驱动级绝对移动
    abs_x = int(cx * 65535 / max(w - 1, 1))
    abs_y = int(cy * 65535 / max(h - 1, 1))
    stroke = InterceptionMouseStroke(
        state=0,
        flags=INTERCEPTION_MOUSE_MOVE_ABSOLUTE,
        rolling=0,
        x=abs_x,
        y=abs_y,
        information=0,
    )
    ctrl.send_stroke(stroke)
    time.sleep(0.05)
    return cx, cy

def click_left(ctrl: InterceptionController, hold_sec: float = 0.08) -> None:
    """在当前位置执行左键单击以激活游戏窗口。"""
    u32 = ctypes.windll.user32
    print("[*] 正在执行鼠标左键单击 (激活前台窗口)...")

    # 驱动级按下
    down = InterceptionMouseStroke(
        state=INTERCEPTION_MOUSE_LEFT_BUTTON_DOWN,
        flags=0,
        rolling=0,
        x=0,
        y=0,
        information=0,
    )
    ctrl.send_stroke(down)
    u32.mouse_event(0x0002, 0, 0, 0, 0)  # MOUSEEVENTF_LEFTDOWN 兜底
    time.sleep(hold_sec)

    # 驱动级抬起
    up = InterceptionMouseStroke(
        state=INTERCEPTION_MOUSE_LEFT_BUTTON_UP,
        flags=0,
        rolling=0,
        x=0,
        y=0,
        information=0,
    )
    ctrl.send_stroke(up)
    u32.mouse_event(0x0004, 0, 0, 0, 0)  # MOUSEEVENTF_LEFTUP 兜底
    time.sleep(0.1)

def draw_circle(
    ctrl: InterceptionController,
    radius: float = 180.0,
    circles: int = 2,
    steps_per_circle: int = 120,
    step_interval: float = 0.012,
    hold_right: bool = False,
    description: str = "",
) -> None:
    """
    通过注入相对位移量 (flags=0, dx, dy) 进行平滑画圆。
    :param ctrl: InterceptionController
    :param radius: 圆半径（单位：相对增量像素当量）
    :param circles: 画几圈
    :param steps_per_circle: 每圈拆分成多少个离散步
    :param step_interval: 每步之间的时间间隔（秒）
    :param hold_right: 是否按住鼠标右键（自由光标模式下需按住右键旋转镜头）
    :param description: 测试描述
    """
    u32 = ctypes.windll.user32
    print(f"\n>>> 开始画圆测试 [{description}] <<<")
    print(f"    参数: 半径={radius}, 圈数={circles}, 每圈采样步数={steps_per_circle}, 按住右键={hold_right}")

    if hold_right:
        print("    [!] 正在按住鼠标右键...")
        down_right = InterceptionMouseStroke(
            state=INTERCEPTION_MOUSE_RIGHT_BUTTON_DOWN,
            flags=0,
            rolling=0,
            x=0,
            y=0,
            information=0,
        )
        ctrl.send_stroke(down_right)
        u32.mouse_event(0x0008, 0, 0, 0, 0)
        time.sleep(0.05)

    try:
        total_steps = circles * steps_per_circle
        curr_offset_x = 0.0
        curr_offset_y = 0.0

        for i in range(1, total_steps + 1):
            theta = 2.0 * math.pi * (i % steps_per_circle) / steps_per_circle
            # 目标瞬时偏移 (以圆心为基准)
            target_x = radius * math.sin(theta)
            target_y = radius * (1.0 - math.cos(theta))

            # 本步需要的相对增量 dx, dy
            dx = int(round(target_x - curr_offset_x))
            dy = int(round(target_y - curr_offset_y))

            curr_offset_x += dx
            curr_offset_y += dy

            stroke = InterceptionMouseStroke(
                state=0,
                flags=INTERCEPTION_MOUSE_MOVE_RELATIVE,
                rolling=0,
                x=dx,
                y=dy,
                information=0,
            )
            ctrl.send_stroke(stroke)
            time.sleep(step_interval)

            if i % steps_per_circle == 0:
                print(f"    -> 完成第 {i // steps_per_circle}/{circles} 圈")

    finally:
        if hold_right:
            print("    [!] 释放鼠标右键...")
            up_right = InterceptionMouseStroke(
                state=INTERCEPTION_MOUSE_RIGHT_BUTTON_UP,
                flags=0,
                rolling=0,
                x=0,
                y=0,
                information=0,
            )
            ctrl.send_stroke(up_right)
            u32.mouse_event(0x0010, 0, 0, 0, 0)
            time.sleep(0.05)

    print(f">>> [{description}] 画圆完成 <<<\n")

def run():
    parser = argparse.ArgumentParser(description="洛克王国：世界 镜头移动测试脚本")
    parser.add_argument(
        "--mode",
        choices=["both", "pure", "right_drag"],
        default="both",
        help="测试模式: pure (纯相对移动, 准星模式), right_drag (按住右键拖动, 自由指针模式), both (两者依次测试)",
    )
    parser.add_argument("--circles", type=int, default=2, help="每次画圆圈数 (默认 2)")
    parser.add_argument("--radius", type=float, default=200.0, help="画圆半径 (默认 200)")
    parser.add_argument("--countdown", type=int, default=3, help="启动倒计时秒数 (默认 3)")
    args = parser.parse_args()

    # 确定 interception.dll 路径
    script_dir = os.path.dirname(os.path.abspath(__file__))
    candidates = [
        os.path.join(script_dir, "..", "RocoKingdom_Clicker", "interception.dll"),
        os.path.join(script_dir, "interception.dll"),
        r"d:\sProject\GameTraining\RocoClicker\RocoKingdom_Clicker\interception.dll",
    ]
    dll_path = next((p for p in candidates if os.path.exists(p)), None)
    if not dll_path:
        print("[ERROR] 未能在预期目录找到 interception.dll！", file=sys.stderr)
        sys.exit(1)

    print("=" * 60)
    print("  《洛克王国：世界》鼠标画圆与镜头旋转测试 (Interception)")
    print("=" * 60)
    print(f"[*] 加载驱动模块: {dll_path}")

    try:
        ctrl = InterceptionController(dll_path)
    except Exception as e:
        print(f"[ERROR] 初始化 Interception 失败: {e}", file=sys.stderr)
        sys.exit(1)

    try:
        print(f"[*] 发现可用鼠标设备: {ctrl.mouse_devices}，默认绑定: Device {ctrl.target_device}")
        print("\n[提示] 请确保游戏窗口处于前台或可见状态！")
        for c in range(args.countdown, 0, -1):
            print(f"[*] 测试将在 {c} 秒后启动...", end="\r", flush=True)
            time.sleep(1.0)
        print("[*] 正在启动测试...                                   ")

        # 1. 移动到主屏幕中心
        cx, cy = move_to_center(ctrl)

        # 2. 点击一次以聚焦游戏
        click_left(ctrl)
        time.sleep(0.4)

        # 3. 开始画圆测试
        if args.mode in ("pure", "both"):
            draw_circle(
                ctrl,
                radius=args.radius,
                circles=args.circles,
                steps_per_circle=120,
                step_interval=0.01,
                hold_right=False,
                description="模式 1: 纯相对移动画圆 (适用于准星锁定 / 战斗模式)",
            )
            if args.mode == "both":
                time.sleep(1.0)

        if args.mode in ("right_drag", "both"):
            draw_circle(
                ctrl,
                radius=args.radius,
                circles=args.circles,
                steps_per_circle=120,
                step_interval=0.01,
                hold_right=True,
                description="模式 2: 按住右键拖拽画圆 (适用于自由光标 / UI 模式)",
            )

        print("[*] 全部测试已顺利执行完毕！")

    finally:
        ctrl.close()
        print("[*] Interception 上下文已正常释放并退出。")

if __name__ == "__main__":
    run()
