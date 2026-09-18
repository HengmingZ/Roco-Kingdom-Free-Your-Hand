"""Model module for roll vision system.
"""

from __future__ import annotations

from .yolo_m import YOLOMDetector, YOLOMConfig
from .pipeline import VisionAimPipeline, DetectionResult

__all__ = ["YOLOMDetector", "YOLOMConfig", "VisionAimPipeline", "DetectionResult"]
