# [NEW - 2026-09-12]
# Reason: Safe Refactoring test script for VisualAimController.step() with auto-click integration.
# Content: Unit test for interleaved camera movement + auto-clicking, deadband filtering, and dynamic parameter tuning.

import dataclasses
import os
import sys
import time
from typing import Optional

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
ROLL_ROOT = os.path.dirname(CURRENT_DIR)
if ROLL_ROOT not in sys.path:
    sys.path.insert(0, ROLL_ROOT)

from control.pid import DualAxisPID, PIDConfig
from control.tracker import TargetTracker, TrackedState
from model.pipeline import DetectionResult


@dataclasses.dataclass
class AimStepResult:
    status: str
    target: Optional[DetectionResult]
    track_id: Optional[int]
    error_x: float
    error_y: float
    dx: int
    dy: int
    inference_time_ms: float
    control_time_ms: float
    clicked: bool = False


# Mocked controller to verify the exact step() logic without requiring GPU YOLO inference
class MockVisualAimController:
    def __init__(
        self,
        auto_click: bool = True,
        click_interval: float = 0.05,
        click_only_when_locked: bool = False,
    ):
        self.auto_click = auto_click
        self.click_interval = click_interval
        self.click_only_when_locked = click_only_when_locked

        self._is_left_down = False
        self._left_down_time = 0.0
        self._last_click_time = 0.0
        self._is_right_down = False
        self.hold_right_button = True

        self.pid = DualAxisPID(
            yaw_config=PIDConfig(kp=0.45, kd=0.08, deadband=4.0),
            pitch_config=PIDConfig(kp=0.38, kd=0.06, deadband=4.0),
        )
        self.tracker = TargetTracker(max_lost_frames=8)
        self.last_dx: int = 0
        self.last_dy: int = 0
        self.last_step_time: Optional[float] = None

        # Tracking call logs for verification
        self.move_history = []
        self.left_button_events = []
        self.right_button_events = []

    def ensure_right_button(self, pressed: bool) -> None:
        if pressed != self._is_right_down:
            self._is_right_down = pressed
            self.right_button_events.append((time.perf_counter(), pressed))

    def ensure_left_button(self, pressed: bool) -> None:
        if pressed != self._is_left_down:
            self._is_left_down = pressed
            if pressed:
                self._left_down_time = time.perf_counter()
            self.left_button_events.append((time.perf_counter(), pressed))

    def send_relative_move(self, dx: int, dy: int) -> bool:
        self.move_history.append((dx, dy))
        return True

    def reset_target(self) -> None:
        self.tracker.reset()
        self.pid.reset()
        self.ensure_right_button(False)
        self.ensure_left_button(False)
        self.last_dx = 0
        self.last_dy = 0

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

    # EXACT STEP IMPLEMENTATION TO TEST AND MIGRATE
    def step(self, detections: list, force_reacquire: bool = False) -> AimStepResult:
        t0 = time.perf_counter()
        dt = (t0 - self.last_step_time) if self.last_step_time else 0.016
        self.last_step_time = t0

        # 0. Non-blocking pulse: release left button if held long enough (>= 15ms)
        if self._is_left_down and (t0 - self._left_down_time >= 0.015):
            self.ensure_left_button(False)

        # Continuous target association with ego-motion feedforward
        target = self.tracker.update(
            detections,
            dt=dt,
            ego_motion_dx=float(self.last_dx),
            ego_motion_dy=float(self.last_dy),
            force_reacquire=force_reacquire,
        )

        if target is None:
            # Target lost or absent: safe release
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
                inference_time_ms=5.0,
                control_time_ms=0.0,
                clicked=False,
            )

        err_x = target.offset_x_from_crosshair
        err_y = target.offset_y_from_crosshair

        # Compute PID outputs
        t2 = time.perf_counter()
        dx, dy = self.pid.update(err_x, err_y, dt=dt)

        # Check deadband
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

        # Auto-Click execution logic
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
            inference_time_ms=5.0,
            control_time_ms=ctrl_ms,
            clicked=clicked,
        )


def test_step_autoclick_suite():
    print("=" * 60)
    print("   VisualAimController.step() 自动连点与镜头追踪并发测试")
    print("=" * 60)

    # Screen dimensions: 1920x1080 -> Center = (960, 540)
    screen_w, screen_h = 1920, 1080
    cx, cy = screen_w / 2, screen_h / 2

    def make_det(target_x, target_y):
        return DetectionResult(
            x1=target_x - 20,
            y1=target_y - 20,
            x2=target_x + 20,
            y2=target_y + 20,
            confidence=0.9,
            class_id=0,
            class_name="target",
            center_x=float(target_x),
            center_y=float(target_y),
            offset_x_from_crosshair=float(target_x - cx),
            offset_y_from_crosshair=float(target_y - cy),
        )

    # Case 1: Tracking + Auto-Clicking concurrently
    print("[*] 测试用例 1: 追踪过程中一边旋转视角一边连点 (continuous tracking + click)...")
    ctrl = MockVisualAimController(auto_click=True, click_interval=0.04, click_only_when_locked=False)

    click_count = 0
    move_count = 0
    for i in range(20):
        tx = 1060 - i * 2
        ty = 600 - i
        dets = [make_det(tx, ty)]
        res = ctrl.step(dets)
        if res.clicked:
            click_count += 1
        if res.status == "TRACKING":
            move_count += 1
        time.sleep(0.016)

    print(f"    [+] 完成 20 帧模拟: 视角移动 {move_count} 次，触发连点 {click_count} 次")
    assert move_count > 0, "Tracking motion must be generated"
    assert click_count >= 3, f"Auto-click should trigger at least 3 times, got {click_count}"
    assert ctrl.left_button_events, "Left button events must be recorded"
    print("    [+] 测试用例 1 通过！")

    # Case 2: Click only when locked (click_only_when_locked = True)
    print("[*] 测试用例 2: 仅在锁定死区时连点 (click_only_when_locked = True)...")
    ctrl2 = MockVisualAimController(auto_click=True, click_interval=0.03, click_only_when_locked=True)

    # 2a: Target far from center (error = 100px) -> should track, but NOT click
    far_dets = [make_det(cx + 100, cy)]
    res_far = ctrl2.step(far_dets)
    assert res_far.status == "TRACKING"
    assert not res_far.clicked, "Should NOT click while far outside deadband"
    print("    [+] 未锁定时不连点，测试通过！")

    # 2b: Target inside deadband (error = 1px) -> should be LOCKED and trigger click
    locked_dets = [make_det(cx + 1, cy + 1)]
    res_locked = ctrl2.step(locked_dets)
    assert res_locked.status == "LOCKED"
    assert res_locked.clicked, "Should click when target is locked in deadband"
    print("    [+] 进入死区锁定后连点触发，测试通过！")

    # Case 3: Target lost -> left button immediately released
    print("[*] 测试用例 3: 目标丢失或重置，确保左键安全释放...")
    ctrl2._is_left_down = True
    res_lost = ctrl2.step([])
    assert res_lost.status == "SEARCHING"
    assert not ctrl2._is_left_down, "Left button must be released when target is lost"
    print("    [+] 目标丢失安全释放左键，测试通过！")

    print("=" * 60)
    print("       全部 3 组 step() 连点与追踪并发测试 100% PASS！")
    print("=" * 60)


if __name__ == "__main__":
    test_step_autoclick_suite()