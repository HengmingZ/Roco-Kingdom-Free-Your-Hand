# [NEW - 2026-09-12]
# Reason: TargetTracker module to ensure consistent tracking of the same physical object in multi-target scenes.
# Content: TrackedState, TargetTracker with multi-metric cost matching (spatial distance, IoU, size consistency),
#          lost-frame occlusion buffering, and camera ego-motion feedforward compensation.

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, List, Optional, Tuple


@dataclass
class TrackedState:
    """State vector of the currently locked target."""
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

    def reset(self) -> None:
        """Clear current locked target."""
        self.current_target = None

    @property
    def is_locked(self) -> bool:
        return self.current_target is not None and self.current_target.lost_frames == 0

    @property
    def locked_id(self) -> Optional[int]:
        return self.current_target.track_id if self.current_target else None

    @property
    def tracked_frames_count(self) -> int:
        return self.current_target.total_tracked_frames if self.current_target else 0

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
