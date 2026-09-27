# [UPDATE - 2026-09-12]
# Reason: Fix sudden freezing/crashing caused by:
#         1. Tkinter cross-thread widget calls from background worker thread (deadlock/crash).
#         2. Heavy 1920x1080 cv2.resize and PhotoImage recreation on GUI thread starving event loop.
#         3. High-frequency slider drag lock contention.
#         4. Concurrent Interception driver calls.
# Modification: Full asynchronous architecture separation:
#         - All UI mutations strictly confined to main thread.
#         - Preview thumbnail pre-rendered in background at throttled 10 FPS.
#         - Canvas image item reused with itemconfig.
#         - Debounced parameter syncing.

from __future__ import annotations

import ctypes
import os
import sys
import threading
import time
import traceback
from typing import Optional

import cv2
import numpy as np
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import ttk, messagebox

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

from capture.screen_grabber import ScreenGrabber
from control.aim_controller import VisualAimController, AimStepResult
from control.pid import PIDConfig


# [UPDATE - 2026-09-19]
# Reason: F7 emergency pause was missed during quick taps due to missing 0x0001 bit check and undefined ctypes restype.
# Modification: Configured user32.GetAsyncKeyState argtypes and c_short restype, and check both 0x8000 (key down)
#               and 0x0001 (key pressed since last call) to eliminate missed keypresses.
VK_F1 = 0x70
VK_F7 = 0x76  # 118 (F7 key)
VK_KEY_N = ord("N")  # 0x4E (78)
VK_F2 = 0x71
VK_F3 = 0x72
VK_ESCAPE = 0x1B

_user32 = ctypes.windll.user32
_user32.GetAsyncKeyState.argtypes = [ctypes.c_int]
_user32.GetAsyncKeyState.restype = ctypes.c_short


def is_key_pressed(vk_code: int) -> bool:
    """Check if a virtual key is pressed globally via GetAsyncKeyState (held down or latched tap)."""
    try:
        val = _user32.GetAsyncKeyState(vk_code)
        return bool((val & 0x8000) or (val & 0x0001))
    except Exception:
        return False


class AimControllerGUI:
    """
    High-stability interactive GUI for Visual Servoing Aim Controller.
    Thread-safe architecture: UI mutations strictly on main thread,
    offloaded preview downscaling, and debounced parameter syncing.
    """

    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("RocoClicker - 视觉伺服 PID 自动瞄准与实时调参控制台")
        self.root.geometry("880x750")
        self.root.minsize(820, 700)
        self.root.configure(bg="#1a1a24")

        # Telemetry & Controller state
        self.controller: Optional[VisualAimController] = None
        self.is_tracking_active = False
        self._last_gui_active_state = False
        self.force_reacquire = False
        self.is_running = True
        self.preview_enabled = tk.BooleanVar(value=True)
        self.is_preview_enabled: bool = True
        self.current_deadband: float = 4.0
        self.params_dirty = False

        # Auto-Click configuration variables
        self.var_auto_click = tk.BooleanVar(value=True)
        self.var_click_locked_only = tk.BooleanVar(value=False)
        self.val_click_interval = tk.DoubleVar(value=0.15)

        self.latest_result: Optional[AimStepResult] = None
        self.latest_thumbnail_rgb: Optional[np.ndarray] = None
        self.latest_fps: float = 0.0
        self._controller_init_done = False
        self._controller_init_error: Optional[str] = None
        self._controller_ready_handled = False
        self.data_lock = threading.Lock()

        # Canvas sizing
        self.canvas_w = 400
        self.canvas_h = 225
        self.canvas_img_id: Optional[int] = None
        self._photo_ref: Optional[ImageTk.PhotoImage] = None

        # Grab monitor info
        self.grabber = ScreenGrabber()
        self.monitors = self.grabber.get_monitors()

        # Init UI components
        self._setup_styles()
        self._build_header_card()
        self._build_main_layout()
        self._build_footer()

        # Initialize controller asynchronously
        self._init_controller_async()

        # Start GUI telemetry refresh timer (30 Hz)
        self.root.after(35, self._gui_refresh_loop)

        # Start parameter debounce sync timer (10 Hz)
        self.root.after(100, self._param_sync_loop)

        # Bind Tkinter window hotkeys for zero-latency in-window response
        self.root.bind_all("<F7>", lambda e: self.request_pause())
        self.root.bind_all("<F1>", lambda e: self.request_start())
        self.root.bind_all("<F3>", lambda e: self.request_reacquire())
        self.root.bind_all("<n>", lambda e: self.request_pause())
        self.root.bind_all("<N>", lambda e: self.request_pause())
        self.root.bind_all("<Escape>", lambda e: self.request_pause())

        # Start dedicated 100Hz global hotkey listener (independent of vision inference)
        self.hotkey_thread = threading.Thread(target=self._hotkey_worker, daemon=True)
        self.hotkey_thread.start()

        # Start tracking worker thread
        self.worker_thread = threading.Thread(target=self._tracking_worker, daemon=True)
        self.worker_thread.start()

        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

    # [UPDATE - 2026-09-12]
    # Reason: Fixed TypeError: _setup_styles() takes 0 positional arguments but 1 was given.
    # Modification: Added self to method signature.
    def _setup_styles(self) -> None:
        style = ttk.Style()
        style.theme_use("clam")

        bg = "#1a1a24"
        card_bg = "#232334"
        fg = "#f1f2f6"

        style.configure(".", background=bg, foreground=fg, font=("Segoe UI", 9))
        style.configure("Card.TFrame", background=card_bg, relief="flat")
        style.configure("CardHeader.TLabel", background=card_bg, foreground="#00d2d3", font=("Segoe UI", 11, "bold"))
        style.configure("ParamLabel.TLabel", background=card_bg, foreground="#dfe4ea", font=("Segoe UI", 9))
        style.configure("ValueLabel.TLabel", background=card_bg, foreground="#00d2d3", font=("Consolas", 10, "bold"))
        style.configure("StatKey.TLabel", background=card_bg, foreground="#a4b0be", font=("Segoe UI", 9))
        style.configure("StatVal.TLabel", background=card_bg, foreground="#ffffff", font=("Consolas", 10, "bold"))

        style.configure("TCheckbutton", background=card_bg, foreground=fg)
        style.configure("TCombobox", fieldbackground="#2f3542", background="#2f3542", foreground="#ffffff")

    def _build_header_card(self) -> None:
        header = ttk.Frame(self.root, style="Card.TFrame", padding=(15, 12))
        header.pack(fill=tk.X, padx=14, pady=(12, 6))

        # Status badge
        self.status_badge = tk.Label(
            header,
            text="● 正在初始化模型与底层驱动...",
            font=("Segoe UI", 11, "bold"),
            bg="#232334",
            fg="#ffa502",
        )
        self.status_badge.pack(side=tk.LEFT)

        # Action buttons
        btn_frame = ttk.Frame(header, style="Card.TFrame")
        btn_frame.pack(side=tk.RIGHT)

        self.btn_start = tk.Button(
            btn_frame,
            text="▶ 启动跟瞄 (F1)",
            font=("Segoe UI", 10, "bold"),
            bg="#10ac84",
            fg="#ffffff",
            activebackground="#1dd1a1",
            activeforeground="#ffffff",
            relief=tk.FLAT,
            padx=12,
            pady=4,
            cursor="hand2",
            command=self.request_start,
        )
        self.btn_start.pack(side=tk.LEFT, padx=5)

        self.btn_pause = tk.Button(
            btn_frame,
            text="⏸ 紧急暂停 (F7 / N / ESC)",
            font=("Segoe UI", 10, "bold"),
            bg="#ee5253",
            fg="#ffffff",
            activebackground="#ff6b6b",
            activeforeground="#ffffff",
            relief=tk.FLAT,
            padx=12,
            pady=4,
            cursor="hand2",
            command=self.request_pause,
        )
        self.btn_pause.pack(side=tk.LEFT, padx=5)

        self.btn_reacquire = tk.Button(
            btn_frame,
            text="↻ 切换目标 (F3)",
            font=("Segoe UI", 9, "bold"),
            bg="#2e86de",
            fg="#ffffff",
            activebackground="#54a0ff",
            activeforeground="#ffffff",
            relief=tk.FLAT,
            padx=10,
            pady=4,
            cursor="hand2",
            command=self.request_reacquire,
        )
        self.btn_reacquire.pack(side=tk.LEFT, padx=5)

    def _build_main_layout(self) -> None:
        body = ttk.Frame(self.root, padding=0)
        body.pack(fill=tk.BOTH, expand=True, padx=14, pady=6)

        # Left Column: Real-time Live Tuning Sliders
        left_col = ttk.Frame(body, style="Card.TFrame", padding=(15, 12))
        left_col.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 6))

        ttk.Label(left_col, text="⚙ 实时控制参数调节 (滑动即刻生效)", style="CardHeader.TLabel").pack(anchor=tk.W, pady=(0, 10))

        # Hardware & Mode row
        env_frame = ttk.Frame(left_col, style="Card.TFrame")
        env_frame.pack(fill=tk.X, pady=(0, 10))

        ttk.Label(env_frame, text="目标屏幕:", style="ParamLabel.TLabel").pack(side=tk.LEFT, padx=(0, 6))
        mon_names = [f"屏幕 {m.index + 1} ({m.width}x{m.height}){' [主屏]' if m.is_primary else ''}" for m in self.monitors]
        self.mon_combo = ttk.Combobox(env_frame, values=mon_names, state="readonly", width=22)
        if mon_names:
            self.mon_combo.current(0)
        self.mon_combo.pack(side=tk.LEFT, padx=(0, 12))
        self.mon_combo.bind("<<ComboboxSelected>>", self._on_env_changed)

        self.var_hold_right = tk.BooleanVar(value=True)
        self.chk_hold_right = ttk.Checkbutton(
            env_frame,
            text="按住右键转动视角 (自由光标模式)",
            variable=self.var_hold_right,
            command=self._on_env_changed,
        )
        self.chk_hold_right.pack(side=tk.LEFT)

        # Sliders Container
        slider_box = ttk.Frame(left_col, style="Card.TFrame")
        slider_box.pack(fill=tk.BOTH, expand=True)

        # 1. Kp-X
        self.val_kp_x = tk.DoubleVar(value=0.45)
        self._create_slider(
            slider_box,
            title="水平响应比例 Kp (Yaw):",
            var=self.val_kp_x,
            from_=0.05,
            to=1.50,
            resolution=0.01,
            fmt="{:.2f}",
            desc="增大加快水平转向追赶速度；过大易造成准星左右微抖",
        )

        # 2. Kd-X
        self.val_kd_x = tk.DoubleVar(value=0.08)
        self._create_slider(
            slider_box,
            title="水平阻尼系数 Kd (Yaw):",
            var=self.val_kd_x,
            from_=0.00,
            to=0.40,
            resolution=0.005,
            fmt="{:.3f}",
            desc="对水平角速度进行反向制动，抑制超调与甩头",
        )

        # 3. Kp-Y
        self.val_kp_y = tk.DoubleVar(value=0.38)
        self._create_slider(
            slider_box,
            title="垂直响应比例 Kp (Pitch):",
            var=self.val_kp_y,
            from_=0.05,
            to=1.50,
            resolution=0.01,
            fmt="{:.2f}",
            desc="垂直俯仰响应比例，通常略小于水平比例以防反冲",
        )

        # 4. Kd-Y
        self.val_kd_y = tk.DoubleVar(value=0.06)
        self._create_slider(
            slider_box,
            title="垂直阻尼系数 Kd (Pitch):",
            var=self.val_kd_y,
            from_=0.00,
            to=0.40,
            resolution=0.005,
            fmt="{:.3f}",
            desc="垂直方向平滑缓冲阻尼",
        )

        # 5. Deadband
        self.val_deadband = tk.DoubleVar(value=4.0)
        self._create_slider(
            slider_box,
            title="锁定死区半径 Deadband (像素):",
            var=self.val_deadband,
            from_=1.0,
            to=20.0,
            resolution=0.5,
            fmt="{:.1f} px",
            desc="进入此半径时视为居中锁定，消除准星微动",
        )

        # 6. Confidence Threshold
        self.val_conf = tk.DoubleVar(value=0.35)
        self._create_slider(
            slider_box,
            title="目标置信度阈值 (Confidence):",
            var=self.val_conf,
            from_=0.10,
            to=0.90,
            resolution=0.02,
            fmt="{:.2f}",
            desc="过滤误识别背景；远距离微小目标可适当调低",
        )

        # 7. Auto-Click options & interval slider
        click_opts_frame = ttk.Frame(slider_box, style="Card.TFrame")
        click_opts_frame.pack(fill=tk.X, pady=(6, 2))

        self.chk_auto_click = ttk.Checkbutton(
            click_opts_frame,
            text="🎯 自动连点 (追踪中自动点击)",
            variable=self.var_auto_click,
            command=self._on_env_changed,
        )
        self.chk_auto_click.pack(side=tk.LEFT)

        self.chk_click_locked_only = ttk.Checkbutton(
            click_opts_frame,
            text="仅在锁定 (LOCKED) 时连点",
            variable=self.var_click_locked_only,
            command=self._on_env_changed,
        )
        self.chk_click_locked_only.pack(side=tk.LEFT, padx=(12, 0))

        self._create_slider(
            slider_box,
            title="连点触发间隔 (秒):",
            var=self.val_click_interval,
            from_=0.03,
            to=0.50,
            resolution=0.01,
            fmt="{:.2f} s",
            desc="鼠标左键触发周期 (默认 0.15s ≈ 6.7 CPS；调小增频)",
        )

        # Reset button
        btn_reset = tk.Button(
            slider_box,
            text="↺ 恢复基准默认参数",
            font=("Segoe UI", 9),
            bg="#2f3542",
            fg="#dfe4ea",
            relief=tk.FLAT,
            padx=10,
            pady=3,
            cursor="hand2",
            command=self._reset_defaults,
        )
        btn_reset.pack(anchor=tk.E, pady=(8, 0))

        # Right Column: Telemetry & Live View Canvas
        right_col = ttk.Frame(body, style="Card.TFrame", padding=(15, 12))
        right_col.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(6, 0))

        ttk.Label(right_col, text="📊 运行状态与视觉反馈", style="CardHeader.TLabel").pack(anchor=tk.W, pady=(0, 10))

        # Telemetry Card
        stat_card = tk.Frame(right_col, bg="#1a1a24", bd=1, relief=tk.SOLID)
        stat_card.pack(fill=tk.X, pady=(0, 10))

        r1 = tk.Frame(stat_card, bg="#1a1a24")
        r1.pack(fill=tk.X, padx=10, pady=(6, 2))
        ttk.Label(r1, text="跟瞄状态:", style="StatKey.TLabel").pack(side=tk.LEFT)
        self.lbl_stat_status = tk.Label(r1, text="SEARCHING", font=("Consolas", 10, "bold"), bg="#1a1a24", fg="#747d8c")
        self.lbl_stat_status.pack(side=tk.LEFT, padx=(5, 12))

        ttk.Label(r1, text="锁定 ID:", style="StatKey.TLabel").pack(side=tk.LEFT)
        self.lbl_stat_id = tk.Label(r1, text="None", font=("Consolas", 10, "bold"), bg="#1a1a24", fg="#00d2d3")
        self.lbl_stat_id.pack(side=tk.LEFT, padx=(5, 12))

        ttk.Label(r1, text="连点状态:", style="StatKey.TLabel").pack(side=tk.LEFT)
        self.lbl_stat_click = tk.Label(r1, text="READY", font=("Consolas", 10, "bold"), bg="#1a1a24", fg="#2ed573")
        self.lbl_stat_click.pack(side=tk.LEFT, padx=(5, 0))

        r2 = tk.Frame(stat_card, bg="#1a1a24")
        r2.pack(fill=tk.X, padx=10, pady=2)
        ttk.Label(r2, text="准星偏差:", style="StatKey.TLabel").pack(side=tk.LEFT)
        self.lbl_stat_err = tk.Label(r2, text="dx:  +0.0 px | dy:  +0.0 px", font=("Consolas", 10), bg="#1a1a24", fg="#ffffff")
        self.lbl_stat_err.pack(side=tk.LEFT, padx=(5, 0))

        r3 = tk.Frame(stat_card, bg="#1a1a24")
        r3.pack(fill=tk.X, padx=10, pady=2)
        ttk.Label(r3, text="驱动输出:", style="StatKey.TLabel").pack(side=tk.LEFT)
        self.lbl_stat_out = tk.Label(r3, text="dx:  +0 | dy:  +0", font=("Consolas", 10), bg="#1a1a24", fg="#ffffff")
        self.lbl_stat_out.pack(side=tk.LEFT, padx=(5, 0))

        r4 = tk.Frame(stat_card, bg="#1a1a24")
        r4.pack(fill=tk.X, padx=10, pady=(2, 6))
        ttk.Label(r4, text="帧率耗时:", style="StatKey.TLabel").pack(side=tk.LEFT)
        self.lbl_stat_perf = tk.Label(r4, text="0.0 FPS | 推理: 0ms | 驱动: 0ms", font=("Consolas", 10), bg="#1a1a24", fg="#2ed573")
        self.lbl_stat_perf.pack(side=tk.LEFT, padx=(5, 0))

        # Preview Toggle & Canvas
        p_row = ttk.Frame(right_col, style="Card.TFrame")
        p_row.pack(fill=tk.X, pady=(4, 6))

        ttk.Checkbutton(
            p_row,
            text="开启画面实时缩略视窗 (后台轻量化降采样)",
            variable=self.preview_enabled,
            command=self._on_preview_toggled,
        ).pack(side=tk.LEFT)

        self.canvas = tk.Canvas(
            right_col,
            width=self.canvas_w,
            height=self.canvas_h,
            bg="#0f0f17",
            highlightthickness=1,
            highlightbackground="#2f3542",
        )
        self.canvas.pack(fill=tk.BOTH, expand=True)
        self.canvas_img_id = self.canvas.create_image(0, 0, anchor=tk.NW)

    def _create_slider(
        self,
        parent: ttk.Frame,
        title: str,
        var: tk.DoubleVar,
        from_: float,
        to: float,
        resolution: float,
        fmt: str,
        desc: str,
    ) -> None:
        frame = ttk.Frame(parent, style="Card.TFrame")
        frame.pack(fill=tk.X, pady=4)

        header = ttk.Frame(frame, style="Card.TFrame")
        header.pack(fill=tk.X)
        ttk.Label(header, text=title, style="ParamLabel.TLabel").pack(side=tk.LEFT)

        val_label = ttk.Label(header, text=fmt.format(var.get()), style="ValueLabel.TLabel")
        val_label.pack(side=tk.RIGHT)

        def on_slide(val_str: str) -> None:
            v = float(val_str)
            val_label.configure(text=fmt.format(v))
            self.params_dirty = True

        scale = tk.Scale(
            frame,
            from_=from_,
            to=to,
            resolution=resolution,
            variable=var,
            orient=tk.HORIZONTAL,
            showvalue=False,
            bg="#232334",
            troughcolor="#1a1a24",
            activebackground="#00d2d3",
            highlightthickness=0,
            command=on_slide,
        )
        scale.pack(fill=tk.X, pady=(2, 0))
        ttk.Label(frame, text=desc, font=("Segoe UI", 8), background="#232334", foreground="#747d8c").pack(anchor=tk.W)

    def _build_footer(self) -> None:
        footer = tk.Frame(self.root, bg="#11111a", pady=6)
        footer.pack(fill=tk.X, side=tk.BOTTOM)

        tip = "全局热键：[F1] 启动/恢复跟瞄  |  [F7 / N / ESC] 紧急暂停 (多键冗余保障)  |  [F3] 切换目标"
        tk.Label(footer, text=tip, font=("Segoe UI", 9), bg="#11111a", fg="#a4b0be").pack()

    def _init_controller_async(self) -> None:
        mon_idx = self.mon_combo.current() if self.mon_combo.current() >= 0 else 0
        cfg_x = PIDConfig(kp=self.val_kp_x.get(), kd=self.val_kd_x.get(), deadband=self.val_deadband.get())
        cfg_y = PIDConfig(kp=self.val_kp_y.get(), kd=self.val_kd_y.get(), deadband=self.val_deadband.get())
        hold_right = self.var_hold_right.get()
        conf = self.val_conf.get()
        auto_clk = self.var_auto_click.get()
        clk_interval = self.val_click_interval.get()
        clk_locked_only = self.var_click_locked_only.get()

        def _loader(m_idx, c_x, c_y, h_right, c_thresh, a_clk, c_intv, c_lock):
            try:
                ctrl = VisualAimController(
                    monitor_index=m_idx,
                    pid_config_x=c_x,
                    pid_config_y=c_y,
                    hold_right_button=h_right,
                    conf_threshold=c_thresh,
                    auto_click=a_clk,
                    click_interval=c_intv,
                    click_only_when_locked=c_lock,
                )
                with self.data_lock:
                    self.controller = ctrl
                    self._controller_init_done = True
            except Exception as e:
                with self.data_lock:
                    self._controller_init_error = str(e)

        threading.Thread(
            target=_loader,
            args=(mon_idx, cfg_x, cfg_y, hold_right, conf, auto_clk, clk_interval, clk_locked_only),
            daemon=True,
        ).start()

    def _on_controller_ready(self) -> None:
        self.status_badge.configure(text="● 控制器已就绪 (暂停中)", fg="#fbc531")

    def _on_controller_error(self, err_msg: str) -> None:
        self.status_badge.configure(text="● 初始化失败", fg="#e84118")
        messagebox.showerror("控制器初始化异常", f"无法加载控制器:\n{err_msg}")

    def _on_env_changed(self, event=None) -> None:
        self.params_dirty = True

    def _on_preview_toggled(self) -> None:
        self.is_preview_enabled = self.preview_enabled.get()
        if not self.is_preview_enabled:
            self.canvas.itemconfig(self.canvas_img_id, image="")
            self._photo_ref = None

    def _param_sync_loop(self) -> None:
        """Periodic debounced sync of parameters from UI to controller."""
        if not self.is_running:
            return

        if self.params_dirty:
            self.params_dirty = False
            self.current_deadband = self.val_deadband.get()
            mon_idx = self.mon_combo.current() if self.mon_combo.current() >= 0 else 0
            with self.data_lock:
                if self.controller:
                    self.controller.update_params(
                        kp_x=self.val_kp_x.get(),
                        kd_x=self.val_kd_x.get(),
                        kp_y=self.val_kp_y.get(),
                        kd_y=self.val_kd_y.get(),
                        deadband=self.current_deadband,
                        conf_threshold=self.val_conf.get(),
                        hold_right_button=self.var_hold_right.get(),
                        monitor_index=mon_idx,
                        auto_click=self.var_auto_click.get(),
                        click_interval=self.val_click_interval.get(),
                        click_only_when_locked=self.var_click_locked_only.get(),
                    )

        self.root.after(80, self._param_sync_loop)

    def _reset_defaults(self) -> None:
        self.val_kp_x.set(0.45)
        self.val_kd_x.set(0.08)
        self.val_kp_y.set(0.38)
        self.val_kd_y.set(0.06)
        self.val_deadband.set(4.0)
        self.val_conf.set(0.35)
        self.var_auto_click.set(True)
        self.val_click_interval.set(0.15)
        self.var_click_locked_only.set(False)
        self.params_dirty = True

    def request_start(self) -> None:
        if not self.controller:
            return
        self.is_tracking_active = True
        with self.data_lock:
            if self.controller:
                self.controller.reset_target()

    # [UPDATE - 2026-09-19]
    # Reason: F7 emergency pause should immediately zero FPS, release all buttons and provide immediate feedback.
    # Modification: Reset latest_fps, release buttons under data_lock, and print console feedback.
    def request_pause(self) -> None:
        self.is_tracking_active = False
        with self.data_lock:
            if self.controller:
                self.controller.ensure_right_button(False)
                self.controller.ensure_left_button(False)
                self.controller.pid.reset()
            self.latest_fps = 0.0
        print("\n[!] [紧急暂停已触发 (F7/N/ESC)] 视角转动与自动连点已停止，底层驱动按键已全部安全释放。", flush=True)

    def request_reacquire(self) -> None:
        self.force_reacquire = True

    # [UPDATE - 2026-09-19]
    # Reason: Decouple hotkey polling from heavy tracking loop so F7/N/ESC is never blocked by YOLO inference.
    # Modification: Added dedicated 100Hz (10ms) background hotkey listener daemon thread.
    def _hotkey_worker(self) -> None:
        """Dedicated high-frequency (100Hz) global hotkey listener. Decoupled from vision inference."""
        while self.is_running:
            try:
                if is_key_pressed(VK_F1):
                    if not self.is_tracking_active:
                        self.request_start()
                        time.sleep(0.15)
                elif (
                    is_key_pressed(VK_F7)
                    or is_key_pressed(VK_KEY_N)
                    or is_key_pressed(VK_F2)
                    or is_key_pressed(VK_ESCAPE)
                ):
                    if self.is_tracking_active:
                        self.request_pause()
                        time.sleep(0.15)
                elif is_key_pressed(VK_F3):
                    self.request_reacquire()
                    time.sleep(0.15)
            except Exception:
                pass
            time.sleep(0.01)

    def _tracking_worker(self) -> None:
        """Continuous tracking worker loop with background thumbnail rendering."""
        frame_count = 0
        t_prev = time.perf_counter()
        t_last_preview = 0.0

        while self.is_running:
            try:
                # Hotkey fallback check (primary polling handled by 100Hz _hotkey_worker)
                if (
                    is_key_pressed(VK_F7)
                    or is_key_pressed(VK_KEY_N)
                    or is_key_pressed(VK_F2)
                    or is_key_pressed(VK_ESCAPE)
                ):
                    if self.is_tracking_active:
                        self.request_pause()
                        time.sleep(0.15)

                # Step execution
                if self.is_tracking_active and self.controller:
                    reacquire = self.force_reacquire
                    self.force_reacquire = False

                    step_res = self.controller.step(force_reacquire=reacquire)

                    # Update FPS
                    t_curr = time.perf_counter()
                    frame_count += 1
                    if t_curr - t_prev >= 0.5:
                        fps = frame_count / (t_curr - t_prev)
                        frame_count = 0
                        t_prev = t_curr
                    else:
                        fps = self.latest_fps

                    # Background thumbnail generation (Throttled to 10 FPS to preserve CPU/GPU)
                    thumbnail_rgb = None
                    if self.is_preview_enabled and (t_curr - t_last_preview >= 0.10):
                        t_last_preview = t_curr
                        raw_frame = self.controller.last_frame
                        if raw_frame is not None:
                            thumbnail_rgb = self._generate_thumbnail_rgb(raw_frame, step_res)

                    with self.data_lock:
                        self.latest_result = step_res
                        self.latest_fps = fps
                        if thumbnail_rgb is not None:
                            self.latest_thumbnail_rgb = thumbnail_rgb
                else:
                    time.sleep(0.015)

            except Exception as e:
                # Prevent worker crash on unexpected transient error
                sys.stderr.write(f"[AimWorker Error]: {e}\n")
                traceback.print_exc()
                time.sleep(0.05)

    def _generate_thumbnail_rgb(self, frame_bgr: np.ndarray, res: AimStepResult) -> np.ndarray:
        """Render annotations directly into a small thumbnail in the background thread."""
        try:
            h, w = frame_bgr.shape[:2]
            vis = frame_bgr.copy()

            # Center crosshair
            cx, cy = int(w // 2), int(h // 2)
            cv2.drawMarker(vis, (cx, cy), (0, 0, 255), cv2.MARKER_CROSS, 24, 2)

            # Deadband circle
            db = int(self.current_deadband)
            cv2.circle(vis, (cx, cy), db, (0, 255, 255), 1)

            # Draw target bounding box and error vector
            if res and res.target:
                t = res.target
                x1, y1, x2, y2 = int(t.x1), int(t.y1), int(t.x2), int(t.y2)
                tx, ty = int(t.center_x), int(t.center_y)

                box_color = (0, 255, 0) if res.status == "LOCKED" else (0, 215, 255)
                cv2.rectangle(vis, (x1, y1), (x2, y2), box_color, 2)
                cv2.line(vis, (cx, cy), (tx, ty), (0, 0, 255), 2)

                label = f"ID:#{res.track_id} {t.confidence:.2f}"
                cv2.putText(vis, label, (x1, max(20, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, box_color, 2)

            # Downsample to target canvas size
            small = cv2.resize(vis, (self.canvas_w, self.canvas_h), interpolation=cv2.INTER_LINEAR)
            return cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
        except Exception:
            return None

    def _gui_refresh_loop(self) -> None:
        """Main thread GUI refresh: safely update widgets and canvas image."""
        if not self.is_running:
            return

        # Check async controller init on main thread
        if not self._controller_ready_handled:
            with self.data_lock:
                done = self._controller_init_done
                err = self._controller_init_error
            if done:
                self._controller_ready_handled = True
                self._on_controller_ready()
            elif err:
                self._controller_ready_handled = True
                self._on_controller_error(err)

        # 1. Synchronize tracking state to UI badge strictly on main thread
        if self.is_tracking_active != self._last_gui_active_state:
            self._last_gui_active_state = self.is_tracking_active
            if self.is_tracking_active:
                self.status_badge.configure(text="● 跟瞄运行中 (RUNNING)", fg="#4cd137")
            else:
                self.status_badge.configure(text="● 紧急暂停 (PAUSED) [按 F1 恢复]", fg="#ee5253")

        # 2. Update telemetry labels
        with self.data_lock:
            res = self.latest_result
            fps = self.latest_fps
            thumb = self.latest_thumbnail_rgb
            is_active = self.is_tracking_active

        # [UPDATE - 2026-09-19]
        # Reason: When paused, telemetry labels previously retained the last tracking result, giving the false illusion that aiming was still running.
        # Modification: Added explicit PAUSED state rendering to zero out status, click, output, and FPS indicators.
        if not is_active:
            self.lbl_stat_status.configure(text="PAUSED", fg="#ee5253")
            self.lbl_stat_click.configure(text="PAUSED", fg="#ee5253")
            self.lbl_stat_out.configure(text="dx:   +0 | dy:   +0")
            self.lbl_stat_perf.configure(text="0.0 FPS | 已挂起暂停", fg="#747d8c")
        elif res:
            stat = res.status
            if stat == "LOCKED":
                self.lbl_stat_status.configure(text="LOCKED", fg="#4cd137")
            elif stat == "TRACKING":
                self.lbl_stat_status.configure(text="TRACKING", fg="#fbc531")
            else:
                self.lbl_stat_status.configure(text="SEARCHING", fg="#747d8c")

            tid_txt = f"#{res.track_id}" if res.track_id else "None"
            self.lbl_stat_id.configure(text=tid_txt)
            self.lbl_stat_err.configure(text=f"dx:{res.error_x:+6.1f} px | dy:{res.error_y:+6.1f} px")
            self.lbl_stat_out.configure(text=f"dx:{res.dx:+4d} | dy:{res.dy:+4d}")
            self.lbl_stat_perf.configure(
                text=f"{fps:4.1f} FPS | 推理:{res.inference_time_ms:3.0f}ms | 驱动:{res.control_time_ms:2.0f}ms",
                fg="#2ed573",
            )

            if res.clicked:
                self.lbl_stat_click.configure(text="CLICK!", fg="#ff4757")
            elif not self.var_auto_click.get():
                self.lbl_stat_click.configure(text="OFF", fg="#747d8c")
            elif self.var_click_locked_only.get() and stat != "LOCKED":
                self.lbl_stat_click.configure(text="WAIT-LOCK", fg="#eccc68")
            else:
                self.lbl_stat_click.configure(text="ARMED", fg="#2ed573")

        # 3. Canvas image update (reuse item ID to avoid memory/handle leak)
        if thumb is not None and self.preview_enabled.get():
            try:
                img = Image.fromarray(thumb)
                self._photo_ref = ImageTk.PhotoImage(img)
                self.canvas.itemconfig(self.canvas_img_id, image=self._photo_ref)
            except Exception:
                pass

        self.root.after(35, self._gui_refresh_loop)

    def on_closing(self) -> None:
        self.is_running = False
        self.is_tracking_active = False
        with self.data_lock:
            if self.controller:
                self.controller.close()
        self.root.destroy()


def main() -> None:
    root = tk.Tk()
    app = AimControllerGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
