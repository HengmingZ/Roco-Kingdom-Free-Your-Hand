# [NEW - 2026-09-12]
# Reason: Implementation of 2D visual servoing PID controller to bring detected target to screen center.
# Content: PIDConfig dataclass, PIDController with anti-windup clamping, deadband jitter suppression,
#          derivative low-pass filtering, and output rate limiting.

from __future__ import annotations

from dataclasses import dataclass
import math
import time
from typing import Optional, Tuple


@dataclass
class PIDConfig:
    """Configuration hyperparameters for PID visual servoing."""
    kp: float = 0.45            # Proportional gain (主要追踪响应强度)
    ki: float = 0.02            # Integral gain (消除稳态微小残差)
    kd: float = 0.08            # Derivative gain (阻尼项，抑制超调与镜头震荡)
    deadband: float = 4.0       # Deadband (像素死区，误差小于此值时不输出，防止准星微抖)
    max_integral: float = 150.0 # Anti-windup clamping (积分分离与限幅)
    max_output: float = 80.0    # Max mouse movement per step (单步最大相对位移，防止镜头瞬移甩飞)
    min_output: float = 1.0     # Minimum output when outside deadband (突破鼠标死区的最小动作)
    d_filter_alpha: float = 0.25# Low-pass filter for derivative (低通滤波平滑微分高频噪声)


class PIDController:
    """1D discrete-time PID controller with industrial-grade stability safeguards."""

    def __init__(self, config: Optional[PIDConfig] = None) -> None:
        self.config = config or PIDConfig()
        self.integral: float = 0.0
        self.prev_error: float = 0.0
        self.prev_derivative: float = 0.0
        self.last_time: Optional[float] = None

    def reset(self) -> None:
        """Reset internal states when target is lost or tracking re-engaged."""
        self.integral = 0.0
        self.prev_error = 0.0
        self.prev_derivative = 0.0
        self.last_time = None

    def compute(self, error: float, dt: Optional[float] = None) -> float:
        """
        Compute control output (relative mouse movement increment) from pixel error.
        :param error: current error (target_pos - screen_center) in pixels
        :param dt: delta time in seconds since last update. If None, auto-calculated.
        :return: control output in relative mouse counts
        """
        now = time.perf_counter()
        if dt is None:
            if self.last_time is None:
                dt = 0.016  # Default to ~60 FPS
            else:
                dt = max(0.001, min(0.1, now - self.last_time))
        self.last_time = now

        # 1. Deadband Check (死区抑制)
        if abs(error) <= self.config.deadband:
            self.integral *= 0.8  # Decay integral smoothly inside deadband
            self.prev_error = error
            return 0.0

        # 2. Proportional Term (比例项)
        p_term = self.config.kp * error

        # 3. Integral Term with Anti-windup Clamping (抗积分饱和)
        self.integral += error * dt
        self.integral = max(-self.config.max_integral, min(self.config.max_integral, self.integral))
        i_term = self.config.ki * self.integral

        # 4. Derivative Term with Frame-Normalized Delta (帧率自适应微分平滑项)
        # Normalized to 60 FPS standard frame (dt * 60.0) so Kd operates intuitively in pixels/frame
        dt_norm = max(0.2, min(5.0, dt * 60.0))
        raw_derivative = (error - self.prev_error) / dt_norm
        filtered_derivative = (
            self.config.d_filter_alpha * raw_derivative + (1.0 - self.config.d_filter_alpha) * self.prev_derivative
        )
        self.prev_derivative = filtered_derivative
        self.prev_error = error
        d_term = self.config.kd * filtered_derivative

        # 5. Raw Control Output (总控制量)
        output = p_term + i_term + d_term

        # 6. Minimum Action Boost (克服游戏/系统鼠标最小运动阻滞)
        if abs(output) < self.config.min_output and abs(error) > self.config.deadband:
            output = math.copysign(self.config.min_output, output)

        # 7. Saturation Output Clamping (输出饱和限幅)
        output = max(-self.config.max_output, min(self.config.max_output, output))
        return output


class DualAxisPID:
    """2D Yaw-Pitch PID controller for camera aiming."""

    def __init__(
        self,
        yaw_config: Optional[PIDConfig] = None,
        pitch_config: Optional[PIDConfig] = None,
    ) -> None:
        self.yaw_pid = PIDController(yaw_config or PIDConfig(kp=0.42, ki=0.015, kd=0.07, max_output=70.0))
        # Pitch axis typically requires slightly lower gain to prevent vertical recoil oscillation
        self.pitch_pid = PIDController(pitch_config or PIDConfig(kp=0.36, ki=0.012, kd=0.06, max_output=50.0))

    def reset(self) -> None:
        self.yaw_pid.reset()
        self.pitch_pid.reset()

    def update(self, offset_x: float, offset_y: float, dt: Optional[float] = None) -> Tuple[int, int]:
        """
        Calculate relative (dx, dy) mouse movements to center target.
        :param offset_x: horizontal pixel offset from screen center (x_target - x_center)
        :param offset_y: vertical pixel offset from screen center (y_target - y_center)
        :return: (dx, dy) integer relative mouse movement increments
        """
        out_x = self.yaw_pid.compute(offset_x, dt)
        out_y = self.pitch_pid.compute(offset_y, dt)
        return int(round(out_x)), int(round(out_y))
