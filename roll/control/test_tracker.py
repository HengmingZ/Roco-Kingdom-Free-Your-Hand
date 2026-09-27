# [TEST / ISOLATION - 2026-09-12]
# Reason: Standalone verification test for TargetTracker and multi-object continuity.
# Content: Tests target locking, distractor rejection, crossing targets, lost-frame buffer,
#          and ego-motion compensation.

from __future__ import annotations

from dataclasses import dataclass
import math
import os
import sys
import time
from typing import List, Optional, Tuple

# Ensure UTF-8 output
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")


@dataclass
class MockDetection:
    class_id: int
    center_x: float
    center_y: float
    w: float
    h: float
    confidence: float = 0.85

    @property
    def x1(self) -> float:
        return self.center_x - self.w / 2.0

    @property
    def y1(self) -> float:
        return self.center_y - self.h / 2.0

    @property
    def x2(self) -> float:
        return self.center_x + self.w / 2.0

    @property
    def y2(self) -> float:
        return self.center_y + self.h / 2.0

    @property
    def offset_x_from_crosshair(self) -> float:
        return self.center_x - 1280.0

    @property
    def offset_y_from_crosshair(self) -> float:
        return self.center_y - 720.0


@dataclass
class TrackedState:
    track_id: int
    x: float
    y: float
    w: float
    h: float
    vx: float = 0.0
    vy: float = 0.0
    lost_frames: int = 0
    total_tracked_frames: int = 1


class TargetTracker:
    """
    Multi-object continuous tracking and lock engine.
    Ensures that once a target is selected, the visual servoing loop consistently follows
    the same physical object across frames, ignoring distractors and handling crossing targets.
    """

    def __init__(
        self,
        max_lost_frames: int = 8,
        max_dist_threshold: float = 220.0,
        iou_weight: float = 0.4,
        dist_weight: float = 0.4,
        size_weight: float = 0.2,
    ) -> None:
        self.max_lost_frames = max_lost_frames
        self.max_dist_threshold = max_dist_threshold
        self.iou_weight = iou_weight
        self.dist_weight = dist_weight
        self.size_weight = size_weight

        self.current_target: Optional[TrackedState] = None
        self._next_id: int = 1
        self._last_time: Optional[float] = None

    def reset(self) -> None:
        """Clear current locked target."""
        self.current_target = None
        self._last_time = None

    @property
    def is_locked(self) -> bool:
        return self.current_target is not None and self.current_target.lost_frames == 0

    def compute_iou(self, b1_xywh: Tuple[float, float, float, float], b2_xywh: Tuple[float, float, float, float]) -> float:
        x1, y1, w1, h1 = b1_xywh
        x2, y2, w2, h2 = b2_xywh

        min_x = max(x1 - w1 / 2.0, x2 - w2 / 2.0)
        min_y = max(y1 - h1 / 2.0, y2 - h2 / 2.0)
        max_x = min(x1 + w1 / 2.0, x2 + w2 / 2.0)
        max_y = min(y1 + h1 / 2.0, y2 + h2 / 2.0)

        inter_w = max(0.0, max_x - min_x)
        inter_h = max(0.0, max_y - min_y)
        inter_area = inter_w * inter_h

        union_area = (w1 * h1) + (w2 * h2) - inter_area
        return inter_area / union_area if union_area > 0 else 0.0

    def update(
        self,
        detections: List[Any],
        dt: float = 0.016,
        ego_motion_dx: float = 0.0,
        ego_motion_dy: float = 0.0,
        camera_scale: float = 0.85,
        force_reacquire: bool = False,
    ) -> Optional[Any]:
        """
        Update tracker with current frame detections.
        :param detections: List of detection objects containing center_x, center_y, x1, y1, x2, y2, offset_*
        :param dt: Time interval in seconds
        :param ego_motion_dx: Controller injected mouse dx from previous step
        :param ego_motion_dy: Controller injected mouse dy from previous step
        :param camera_scale: Screen pixels shifted per 1 count of mouse relative movement
        :param force_reacquire: Force locking to new closest target to crosshair
        :return: Matched detection for the locked target, or None if target lost/absent
        """
        if force_reacquire:
            self.current_target = None

        if not detections:
            if self.current_target is not None:
                self.current_target.lost_frames += 1
                if self.current_target.lost_frames > self.max_lost_frames:
                    self.current_target = None
            return None

        # 1. State: No target currently locked -> Acquire closest target to crosshair
        if self.current_target is None:
            best_init = min(
                detections, key=lambda d: d.offset_x_from_crosshair**2 + d.offset_y_from_crosshair**2
            )
            w = best_init.x2 - best_init.x1
            h = best_init.y2 - best_init.y1
            self.current_target = TrackedState(
                track_id=self._next_id,
                x=best_init.center_x,
                y=best_init.center_y,
                w=w,
                h=h,
                vx=0.0,
                vy=0.0,
                lost_frames=0,
                total_tracked_frames=1,
            )
            self._next_id += 1
            return best_init

        # 2. State: Target is locked -> Predict expected position taking into account:
        #    a) Target's own velocity momentum: (vx * dt, vy * dt)
        #    b) Ego-motion camera shift: - (ego_motion_dx * scale, ego_motion_dy * scale)
        pred_x = self.current_target.x + self.current_target.vx * dt - ego_motion_dx * camera_scale
        pred_y = self.current_target.y + self.current_target.vy * dt - ego_motion_dy * camera_scale
        target_xywh = (pred_x, pred_y, self.current_target.w, self.current_target.h)

        # 3. Match candidate detections by Cost Function
        best_match = None
        min_cost = float("inf")

        for det in detections:
            det_w = det.x2 - det.x1
            det_h = det.y2 - det.y1
            det_xywh = (det.center_x, det.center_y, det_w, det_h)

            # Spatial euclidean distance
            dist = math.hypot(det.center_x - pred_x, det.center_y - pred_y)
            if dist > self.max_dist_threshold:
                continue

            dist_cost = dist / self.max_dist_threshold

            # IoU cost
            iou = self.compute_iou(target_xywh, det_xywh)
            iou_cost = 1.0 - iou

            # Scale/aspect ratio cost
            size_cost = abs(math.log(max(1.0, det_w) / max(1.0, self.current_target.w))) + abs(
                math.log(max(1.0, det_h) / max(1.0, self.current_target.h))
            )
            size_cost = min(1.0, size_cost)

            total_cost = (
                self.dist_weight * dist_cost
                + self.iou_weight * iou_cost
                + self.size_weight * size_cost
            )

            if total_cost < min_cost:
                min_cost = total_cost
                best_match = det

        # 4. Association Result
        if best_match is not None and min_cost < 0.65:
            # Update target velocity with exponential moving average
            new_vx = (best_match.center_x - self.current_target.x + ego_motion_dx * camera_scale) / max(0.001, dt)
            new_vy = (best_match.center_y - self.current_target.y + ego_motion_dy * camera_scale) / max(0.001, dt)

            self.current_target.vx = 0.6 * self.current_target.vx + 0.4 * new_vx
            self.current_target.vy = 0.6 * self.current_target.vy + 0.4 * new_vy
            self.current_target.x = best_match.center_x
            self.current_target.y = best_match.center_y
            self.current_target.w = best_match.x2 - best_match.x1
            self.current_target.h = best_match.y2 - best_match.y1
            self.current_target.lost_frames = 0
            self.current_target.total_tracked_frames += 1
            return best_match
        else:
            # Missed frame: extrapolate virtual position using momentum
            self.current_target.lost_frames += 1
            self.current_target.x = pred_x
            self.current_target.y = pred_y
            if self.current_target.lost_frames > self.max_lost_frames:
                self.current_target = None
            return None


def run_unit_tests():
    print("=" * 65)
    print("       TargetTracker 多目标连续锁定与抗干扰单元验证")
    print("=" * 65)

    tracker = TargetTracker()

    # Scenario 1: Initial Acquisition
    # Frame 0: Target A at (1350, 750), Target B at (1600, 900)
    det_a = MockDetection(class_id=0, center_x=1350.0, center_y=750.0, w=80.0, h=120.0)
    det_b = MockDetection(class_id=0, center_x=1600.0, center_y=900.0, w=70.0, h=100.0)

    matched = tracker.update([det_a, det_b], dt=0.016)
    print(f"[*] 帧 0 初始化锁定: 锁定目标 ID={tracker.current_target.track_id} (坐标: {matched.center_x}, {matched.center_y})")
    assert matched == det_a, "初次锁定应优先选择距离准星(1280,720)最近的目标 A"
    assert tracker.current_target.track_id == 1
    print("[+] 初次锁定 (Nearest Acquisition) 校验通过！")

    # Scenario 2: Distractor Appears Closer to Crosshair
    # Target A moved slightly to (1340, 740).
    # A NEW distractor Target C suddenly appears at (1290, 725) - very close to crosshair!
    det_a_f1 = MockDetection(class_id=0, center_x=1340.0, center_y=740.0, w=80.0, h=120.0)
    det_c_distractor = MockDetection(class_id=0, center_x=1290.0, center_y=725.0, w=75.0, h=110.0)

    matched_f1 = tracker.update([det_c_distractor, det_a_f1], dt=0.016)
    print(f"[*] 帧 1 干扰目标出现: 返回目标=({matched_f1.center_x}, {matched_f1.center_y})")
    assert matched_f1 == det_a_f1, "出现更近的干扰目标 C 时，必须坚定锁定已跟踪的目标 A，不可发生漂移！"
    assert tracker.current_target.track_id == 1
    print("[+] 干扰目标免疫 (Distractor Rejection) 校验通过！")

    # Scenario 3: Targets Crossing
    # Target A moves from (1340) -> (1310), Target B crosses nearby at (1320)
    det_a_f2 = MockDetection(class_id=0, center_x=1310.0, center_y=730.0, w=80.0, h=120.0)
    det_b_cross = MockDetection(class_id=0, center_x=1325.0, center_y=730.0, w=50.0, h=60.0) # different size

    matched_f2 = tracker.update([det_b_cross, det_a_f2], dt=0.016)
    assert matched_f2 == det_a_f2, "两目标交叉时，根据运动学与尺寸一致性必须稳锁 A"
    print("[+] 目标交叉穿越 (Crossing Disambiguation) 校验通过！")

    # Scenario 4: Temporary Occlusion / Missed Frames (失靶缓冲)
    # Frames 3 & 4: Detection empty (e.g. flash/smoke/special effect)
    m3 = tracker.update([], dt=0.016)
    assert m3 is None and tracker.current_target is not None, "单帧漏检时不应立即解除锁定，而应进入缓冲态"
    assert tracker.current_target.lost_frames == 1

    m4 = tracker.update([], dt=0.016)
    assert m4 is None and tracker.current_target.lost_frames == 2

    # Frame 5: Target reappears at (1305, 728)
    det_a_f5 = MockDetection(class_id=0, center_x=1305.0, center_y=728.0, w=80.0, h=120.0)
    matched_f5 = tracker.update([det_a_f5], dt=0.016)
    assert matched_f5 == det_a_f5, "遮挡结束后重新捕获原锁定目标"
    assert tracker.current_target.lost_frames == 0
    print("[+] 遮挡与瞬时漏检缓冲 (Occlusion Recovery) 校验通过！")

    # Scenario 5: Ego-Motion Compensation (相机自身大幅转动)
    # Suppose PID moved mouse by dx=50, dy=0.
    # Entire screen shifted left by ~42.5px. Target A now at (1305 - 42.5 = 1262.5).
    det_a_f6 = MockDetection(class_id=0, center_x=1262.5, center_y=728.0, w=80.0, h=120.0)
    det_other = MockDetection(class_id=0, center_x=1305.0, center_y=728.0, w=80.0, h=120.0) # stationary distractor

    matched_f6 = tracker.update([det_other, det_a_f6], dt=0.016, ego_motion_dx=50.0, ego_motion_dy=0.0)
    assert matched_f6 == det_a_f6, "镜头主动转动前馈补偿必须准确匹配随画面位移的目标 A"
    print("[+] 镜头自发运动前馈补偿 (Ego-Motion Compensation) 校验通过！")

    print("\n" + "=" * 65)
    print("      TargetTracker 单元测试全部顺利通过 (100% PASS)")
    print("=" * 65)


if __name__ == "__main__":
    run_unit_tests()
