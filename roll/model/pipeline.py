# [NEW - 2026-09-12]
# Reason: VisionAimPipeline connecting ScreenGrabber, YOLOMDetector, and calculating camera aim delta vectors.
# Content: DetectionResult dataclass, VisionAimPipeline class with target locking and relative offset calculation.

from __future__ import annotations

from dataclasses import dataclass
import os
import sys
from typing import List, Optional, Tuple

import numpy as np

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
ROLL_ROOT = os.path.dirname(CURRENT_DIR)

if ROLL_ROOT not in sys.path:
    sys.path.insert(0, ROLL_ROOT)

from capture.screen_grabber import ScreenGrabber
from model.yolo_m import YOLOMDetector, YOLOMConfig


@dataclass
class DetectionResult:
    """Bounding box detection result with camera offset calculation."""
    x1: float
    y1: float
    x2: float
    y2: float
    confidence: float
    class_id: int
    class_name: str
    center_x: float
    center_y: float
    offset_x_from_crosshair: float  # Horizontal offset from screen center (pixels)
    offset_y_from_crosshair: float  # Vertical offset from screen center (pixels)


class VisionAimPipeline:
    """
    Vision Detection & Aiming Pipeline for Roll Camera System.
    Captures target screen frame -> YOLO-m inference -> Target selection -> Camera aim offsets.
    """

    def __init__(
        self,
        detector: Optional[YOLOMDetector] = None,
        grabber: Optional[ScreenGrabber] = None,
        monitor_index: int = 0,
    ) -> None:
        self.grabber = grabber or ScreenGrabber()
        self.detector = detector or YOLOMDetector()
        self.monitor_index = monitor_index

    def process_frame(self, target_class_id: Optional[int] = None) -> Tuple[np.ndarray, List[DetectionResult]]:
        """
        Capture current frame, run YOLO-m detection, and compute crosshair offsets.
        :return: (raw_frame_bgr, list_of_detections)
        """
        frame_bgr = self.grabber.capture_screen(self.monitor_index)
        h, w = frame_bgr.shape[:2]
        crosshair_x = w / 2.0
        crosshair_y = h / 2.0

        detections: List[DetectionResult] = []
        results = self.detector.predict(frame_bgr)

        for res in results:
            boxes = getattr(res, "boxes", None)
            if boxes is None or len(boxes) == 0:
                continue

            xyxy = boxes.xyxy.cpu().numpy()
            confs = boxes.conf.cpu().numpy()
            clss = boxes.cls.cpu().numpy()
            names = getattr(res, "names", {})

            for i in range(len(xyxy)):
                cid = int(clss[i])
                if target_class_id is not None and cid != target_class_id:
                    continue

                x1, y1, x2, y2 = xyxy[i]
                conf = float(confs[i])
                cname = names.get(cid, str(cid))
                cx = (x1 + x2) / 2.0
                cy = (y1 + y2) / 2.0
                off_x = cx - crosshair_x
                off_y = cy - crosshair_y

                detections.append(
                    DetectionResult(
                        x1=float(x1),
                        y1=float(y1),
                        x2=float(x2),
                        y2=float(y2),
                        confidence=conf,
                        class_id=cid,
                        class_name=cname,
                        center_x=cx,
                        center_y=cy,
                        offset_x_from_crosshair=off_x,
                        offset_y_from_crosshair=off_y,
                    )
                )

        # Sort detections by proximity to screen crosshair
        detections.sort(key=lambda d: d.offset_x_from_crosshair**2 + d.offset_y_from_crosshair**2)
        return frame_bgr, detections
