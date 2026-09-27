# [NEW - 2026-09-12]
# Reason: VisualAimController integrating VisionAimPipeline, DualAxisPID, and Interception driver.
# Content: Full-loop visual tracking and centering controller with safety bounds, right-button dragging,
#          status tracking (LOCKED / TRACKING / SEARCHING), and telemetry statistics.

from __future__ import annotations

import ctypes
from dataclasses import dataclass
import os
import sys
import threading
import time
from typing import Optional, Tuple

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
ROLL_ROOT = os.path.dirname(CURRENT_DIR)
PROJECT_ROOT = os.path.dirname(ROLL_ROOT)

if ROLL_ROOT not in sys.path:
    sys.path.insert(0, ROLL_ROOT)

from capture.screen_grabber import ScreenGrabber
from model.yolo_m import YOLOMDetector, YOLOMConfig
from model.pipeline import VisionAimPipeline, DetectionResult
from model.weights_manager import resolve_and_ensure_weights
from control.pid import DualAxisPID, PIDConfig
from control.tracker import TargetTracker, TrackedState


class InterceptionMouseStroke(ctypes.Structure):
    _fields_ = [
        ("state", ctypes.c_ushort),
        ("flags", ctypes.c_ushort),
        ("rolling", ctypes.c_short),
        ("x", ctypes.c_int),
        ("y", ctypes.c_int),
        ("information", ctypes.c_uint),
    ]

INTERCEPTION_MOUSE_LEFT_BUTTON_DOWN   = 0x001
INTERCEPTION_MOUSE_LEFT_BUTTON_UP     = 0x002
INTERCEPTION_MOUSE_RIGHT_BUTTON_DOWN  = 0x004
INTERCEPTION_MOUSE_RIGHT_BUTTON_UP    = 0x008
INTERCEPTION_MOUSE_MOVE_RELATIVE      = 0x000


@dataclass
class AimStepResult:
    status: str                         # "LOCKED" (已居中), "TRACKING" (跟瞄中), "SEARCHING" (未发现目标)
    target: Optional[DetectionResult]   # 锁定的目标检测信息
    track_id: Optional[int]             # 锁定目标的连续追踪 ID
    error_x: float                      # 水平像素偏差
    error_y: float                      # 垂直像素偏差
    dx: int                             # 本步注入的鼠标水平增量
    dy: int                             # 本步注入的鼠标垂直增量
    inference_time_ms: float            # 推理耗时 (毫秒)
    control_time_ms: float              # 控制计算与驱动耗时 (毫秒)
    clicked: bool = False               # 本步是否触发了鼠标左键连点射击/交互


# [UPDATE - 2026-09-12]
# Reason: Multi-target scene ambiguity. Previous logic used greedy nearest-to-crosshair which caused target flicking.
# Modification: Integrated TargetTracker with kinematic state association, ego-motion feedforward,
#               and lost-frame buffer to guarantee the controller consistently tracks the same target.
class VisualAimController:
    """
    Visual Servoing Aim Controller:
    Closes the loop between Vision Perception (YOLO-m) and Driver Actuation (Interception)
    via Dual-Axis PID control and TargetTracker multi-object continuity.
    """

    def __init__(
        self,
        weights_path: Optional[str] = None,
        monitor_index: int = 0,
        pid_config_x: Optional[PIDConfig] = None,
        pid_config_y: Optional[PIDConfig] = None,
        hold_right_button: bool = True,
        conf_threshold: float = 0.35,
        max_lost_frames: int = 8,
        auto_click: bool = True,
        click_interval: float = 0.15,
        click_only_when_locked: bool = False,
    ) -> None:
        self.monitor_index = monitor_index
        self.hold_right_button = hold_right_button

        # Auto-Click configuration
        self.auto_click = auto_click
        self.click_interval = click_interval
        self.click_only_when_locked = click_only_when_locked
        self._is_left_down = False
        self._left_down_time = 0.0
        self._last_click_time = 0.0

        # 1. Pipeline & Model
        # [UPDATE - 2026-09-19]
        # Reason: Out-of-the-box readiness with automatic GitHub Release weights download & fallback.
        # Modification: Replaced static local path search with resolve_and_ensure_weights().
        resolved_weights = resolve_and_ensure_weights(weights_path=weights_path, auto_download=True)
        cfg = YOLOMConfig(weights_path=resolved_weights, conf_threshold=conf_threshold)
        self.detector = YOLOMDetector(cfg)
        self.grabber = ScreenGrabber()
        self.pipeline = VisionAimPipeline(detector=self.detector, grabber=self.grabber, monitor_index=monitor_index)

        # 2. Dual Axis PID & Target Tracker
        self.pid = DualAxisPID(yaw_config=pid_config_x, pitch_config=pid_config_y)
        self.tracker = TargetTracker(max_lost_frames=max_lost_frames)
        self.last_dx: int = 0
        self.last_dy: int = 0
        self.last_step_time: Optional[float] = None
        self.last_frame: Optional[np.ndarray] = None
        self.last_detections: list = []

        # 3. Interception Driver
        self._lib = None
        self._ctx = None
        self.device = 11
        self._init_interception()

        # [UPDATE - 2026-09-12]
        # Reason: Interception context is not thread-safe. Concurrent sends cause driver deadlocks/hangs.
        # Modification: Added reentrant _driver_lock (RLock) to serialize all interception_send operations safely.
        self._driver_lock = threading.RLock()
        self._is_right_down = False

    def _init_interception(self) -> None:
        dll_candidates = [
            os.path.join(ROLL_ROOT, "interception.dll"),
            os.path.join(PROJECT_ROOT, "RocoKingdom_Clicker", "interception.dll"),
            r"d:\sProject\GameTraining\RocoClicker\RocoKingdom_Clicker\interception.dll",
        ]
        dll_path = next((p for p in dll_candidates if os.path.exists(p)), None)
        if not dll_path:
            raise FileNotFoundError("未找到 interception.dll 驱动库！")

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
            raise RuntimeError("无法初始化 Interception 上下文！")

        mouses = [d for d in range(11, 21) if self._lib.interception_is_mouse(d)]
        self.device = mouses[0] if mouses else 11

    def send_relative_move(self, dx: int, dy: int) -> bool:
        """Inject relative mouse movement into kernel driver (thread-safe)."""
        if not self._ctx or (dx == 0 and dy == 0):
            return True
        stroke = InterceptionMouseStroke(
            state=0,
            flags=INTERCEPTION_MOUSE_MOVE_RELATIVE,
            rolling=0,
            x=dx,
            y=dy,
            information=0,
        )
        with self._driver_lock:
            if not self._ctx:
                return False
            return self._lib.interception_send(self._ctx, self.device, ctypes.byref(stroke), 1) > 0

    def ensure_right_button(self, pressed: bool) -> None:
        """Manage right button state if free-cursor camera rotation is enabled (thread-safe)."""
        if not self.hold_right_button or not self._ctx:
            return

        with self._driver_lock:
            if not self._ctx:
                return
            if pressed and not self._is_right_down:
                down = InterceptionMouseStroke(
                    state=INTERCEPTION_MOUSE_RIGHT_BUTTON_DOWN,
                    flags=0, rolling=0, x=0, y=0, information=0
                )
                self._lib.interception_send(self._ctx, self.device, ctypes.byref(down), 1)
                self._is_right_down = True
            elif not pressed and self._is_right_down:
                up = InterceptionMouseStroke(
                    state=INTERCEPTION_MOUSE_RIGHT_BUTTON_UP,
                    flags=0, rolling=0, x=0, y=0, information=0
                )
                self._lib.interception_send(self._ctx, self.device, ctypes.byref(up), 1)
                self._is_right_down = False

    # [UPDATE - 2026-09-12]
    # Reason: Support auto-clicking while tracking target with camera movement.
    # Modification: Added thread-safe ensure_left_button and send_left_click methods.
    def ensure_left_button(self, pressed: bool) -> None:
        """Manage left button state for clicking/shooting (thread-safe)."""
        if not self._ctx:
            return

        with self._driver_lock:
            if not self._ctx:
                return
            if pressed and not self._is_left_down:
                down = InterceptionMouseStroke(
                    state=INTERCEPTION_MOUSE_LEFT_BUTTON_DOWN,
                    flags=0, rolling=0, x=0, y=0, information=0
                )
                self._lib.interception_send(self._ctx, self.device, ctypes.byref(down), 1)
                self._is_left_down = True
                self._left_down_time = time.perf_counter()
            elif not pressed and self._is_left_down:
                up = InterceptionMouseStroke(
                    state=INTERCEPTION_MOUSE_LEFT_BUTTON_UP,
                    flags=0, rolling=0, x=0, y=0, information=0
                )
                self._lib.interception_send(self._ctx, self.device, ctypes.byref(up), 1)
                self._is_left_down = False

    def send_left_click(self, hold_duration: float = 0.015) -> bool:
        """Inject discrete mouse left click (down + up) into kernel driver (thread-safe)."""
        if not self._ctx:
            return False
        with self._driver_lock:
            if not self._ctx:
                return False
            down = InterceptionMouseStroke(
                state=INTERCEPTION_MOUSE_LEFT_BUTTON_DOWN,
                flags=0, rolling=0, x=0, y=0, information=0
            )
            up = InterceptionMouseStroke(
                state=INTERCEPTION_MOUSE_LEFT_BUTTON_UP,
                flags=0, rolling=0, x=0, y=0, information=0
            )
            ok1 = self._lib.interception_send(self._ctx, self.device, ctypes.byref(down), 1) > 0
            if hold_duration > 0:
                time.sleep(hold_duration)
            ok2 = self._lib.interception_send(self._ctx, self.device, ctypes.byref(up), 1) > 0
            return ok1 and ok2

    def reset_target(self) -> None:
        """Force drop current lock and reset PID and button states."""
        self.tracker.reset()
        self.pid.reset()
        self.ensure_right_button(False)
        self.ensure_left_button(False)
        self.last_dx = 0
        self.last_dy = 0

    # [UPDATE - 2026-09-12]
    # Reason: Support real-time on-the-fly parameter tuning from GUI controls,
    #         including auto-click toggle, interval, and deadband lock condition.
    # Modification: Added auto_click, click_interval, and click_only_when_locked dynamic updating.
    def update_params(
        self,
        kp_x: Optional[float] = None,
        kd_x: Optional[float] = None,
        kp_y: Optional[float] = None,
        kd_y: Optional[float] = None,
        deadband: Optional[float] = None,
        conf_threshold: Optional[float] = None,
        hold_right_button: Optional[bool] = None,
        monitor_index: Optional[int] = None,
        auto_click: Optional[bool] = None,
        click_interval: Optional[float] = None,
        click_only_when_locked: Optional[bool] = None,
    ) -> None:
        """Dynamically update PID gains, detector confidence, or control flags in real-time."""
        if kp_x is not None:
            self.pid.yaw_pid.config.kp = float(kp_x)
        if kd_x is not None:
            self.pid.yaw_pid.config.kd = float(kd_x)
        if kp_y is not None:
            self.pid.pitch_pid.config.kp = float(kp_y)
        if kd_y is not None:
            self.pid.pitch_pid.config.kd = float(kd_y)
        if deadband is not None:
            self.pid.yaw_pid.config.deadband = float(deadband)
            self.pid.pitch_pid.config.deadband = float(deadband)
        if conf_threshold is not None:
            self.detector.config.conf_threshold = float(conf_threshold)
        if hold_right_button is not None:
            if not hold_right_button and self._is_right_down:
                self.ensure_right_button(False)
            self.hold_right_button = bool(hold_right_button)
        if auto_click is not None:
            if not auto_click and self._is_left_down:
                self.ensure_left_button(False)
            self.auto_click = bool(auto_click)
        if click_interval is not None:
            self.click_interval = float(click_interval)
        if click_only_when_locked is not None:
            self.click_only_when_locked = bool(click_only_when_locked)
        if monitor_index is not None and monitor_index != self.monitor_index:
            self.monitor_index = int(monitor_index)
            self.pipeline.monitor_index = int(monitor_index)
            self.reset_target()

    # [UPDATE - 2026-09-12]
    # Reason: Auto-click while tracking target with camera movement (一边移动镜头追踪，一边click).
    # Modification: Integrated non-blocking 15ms click release state machine, auto-click rate limiter,
    #               deadband condition checks, and safe left button release when target is lost.
    def step(self, force_reacquire: bool = False) -> AimStepResult:
        """
        Execute one complete control loop iteration:
        Sense (Capture + YOLO) -> Associate & Track (TargetTracker) -> Compute (PID) -> Actuate (Interception).
        """
        t0 = time.perf_counter()
        dt = (t0 - self.last_step_time) if self.last_step_time else 0.016
        self.last_step_time = t0

        # 0. Non-blocking pulse: release left button if held long enough (>= 15ms)
        if self._is_left_down and (t0 - self._left_down_time >= 0.015):
            self.ensure_left_button(False)

        frame, detections = self.pipeline.process_frame()
        self.last_frame = frame
        self.last_detections = detections
        t1 = time.perf_counter()
        infer_ms = (t1 - t0) * 1000.0

        # Continuous target association with ego-motion feedforward
        target = self.tracker.update(
            detections,
            dt=dt,
            ego_motion_dx=float(self.last_dx),
            ego_motion_dy=float(self.last_dy),
            force_reacquire=force_reacquire,
        )

        if target is None:
            # Target lost or absent
            self.pid.reset()
            self.ensure_right_button(False)
            self.ensure_left_button(False)
            self.last_dx = 0
            self.last_dy = 0
            return AimStepResult(
                status="SEARCHING",
                target=None,
                track_id=None,
                error_x=0.0,
                error_y=0.0,
                dx=0,
                dy=0,
                inference_time_ms=infer_ms,
                control_time_ms=0.0,
                clicked=False,
            )

        err_x = target.offset_x_from_crosshair
        err_y = target.offset_y_from_crosshair

        # Compute PID outputs
        t2 = time.perf_counter()
        dx, dy = self.pid.update(err_x, err_y, dt=dt)

        # Check if already within deadband
        deadband = self.pid.yaw_pid.config.deadband
        is_locked = abs(err_x) <= deadband and abs(err_y) <= deadband

        if is_locked:
            status = "LOCKED"
            self.ensure_right_button(False)
            self.last_dx = 0
            self.last_dy = 0
        else:
            status = "TRACKING"
            self.ensure_right_button(True)
            self.send_relative_move(dx, dy)
            self.last_dx = dx
            self.last_dy = dy

        # Auto-Click execution logic while tracking or locked
        clicked = False
        if self.auto_click:
            can_click = True
            if self.click_only_when_locked and not is_locked:
                can_click = False

            if can_click and not self._is_left_down:
                if (t0 - self._last_click_time) >= self.click_interval:
                    self.ensure_left_button(True)
                    self._last_click_time = t0
                    clicked = True

        t3 = time.perf_counter()
        ctrl_ms = (t3 - t2) * 1000.0

        return AimStepResult(
            status=status,
            target=target,
            track_id=self.tracker.locked_id,
            error_x=err_x,
            error_y=err_y,
            dx=dx,
            dy=dy,
            inference_time_ms=infer_ms,
            control_time_ms=ctrl_ms,
            clicked=clicked,
        )

    def close(self) -> None:
        """Safely release buttons and destroy context."""
        with self._driver_lock:
            self.ensure_right_button(False)
            self.ensure_left_button(False)
            if self._ctx and self._lib:
                try:
                    self._lib.interception_destroy_context(self._ctx)
                except Exception:
                    pass
                self._ctx = None
