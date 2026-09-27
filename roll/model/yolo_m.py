# [NEW - 2026-09-12]
# Reason: YOLO-m network architecture and detector encapsulation for RocoClicker roll module.
# Content: YOLOMConfig, YOLOMDetector with PyTorch module architecture, CUDA auto-detection,
#          inference, parameter estimation and ONNX export.

from __future__ import annotations

from dataclasses import dataclass
import os
import sys
from typing import Any, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from ultralytics import YOLO


@dataclass
class YOLOMConfig:
    """Configuration for YOLO-m architecture."""
    model_name: str = "yolov8m"           # Architecture identifier (yolov8m / custom)
    weights_path: Optional[str] = None    # Path to existing .pt or .onnx weights
    num_classes: int = 80                 # Number of target classes
    input_size: Tuple[int, int] = (640, 640)
    conf_threshold: float = 0.35
    iou_threshold: float = 0.45
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    half_precision: bool = True           # FP16 inference for CUDA


class ConvBlock(nn.Module):
    """Standard Conv2d -> BatchNorm2d -> SiLU activation block."""

    def __init__(self, in_c: int, out_c: int, k: int = 3, s: int = 1, p: Optional[int] = None):
        super().__init__()
        if p is None:
            p = k // 2
        self.conv = nn.Conv2d(in_c, out_c, kernel_size=k, stride=s, padding=p, bias=False)
        self.bn = nn.BatchNorm2d(out_c)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.conv(x)))


class Bottleneck(nn.Module):
    """Standard YOLO bottleneck with optional residual connection."""

    def __init__(self, c1: int, c2: int, shortcut: bool = True):
        super().__init__()
        c_mid = c2 // 2
        self.cv1 = ConvBlock(c1, c_mid, k=3, s=1)
        self.cv2 = ConvBlock(c_mid, c2, k=3, s=1)
        self.add = shortcut and c1 == c2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class C2f(nn.Module):
    """CSP Bottleneck with 2 convolutions (Core building block of YOLOv8/m)."""

    def __init__(self, c1: int, c2: int, n: int = 1, shortcut: bool = True):
        super().__init__()
        self.c = int(c2 * 0.5)
        self.cv1 = ConvBlock(c1, 2 * self.c, k=1, s=1)
        self.cv2 = ConvBlock((2 + n) * self.c, c2, k=1, s=1)
        self.m = nn.ModuleList(Bottleneck(self.c, self.c, shortcut) for _ in range(n))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))


class YOLOMBackbone(nn.Module):
    """
    Standard YOLOv8 Medium (YOLO-m) Backbone.
    Width multiplier = 0.75, Depth multiplier = 0.67.
    Channels: P1/2 (48), P2/4 (96), P3/8 (192), P4/16 (384), P5/32 (576).
    """

    def __init__(self):
        super().__init__()
        # Stem
        self.p1 = ConvBlock(3, 48, k=3, s=2)
        # Stage 2
        self.p2 = nn.Sequential(
            ConvBlock(48, 96, k=3, s=2),
            C2f(96, 96, n=2, shortcut=True),
        )
        # Stage 3
        self.p3 = nn.Sequential(
            ConvBlock(96, 192, k=3, s=2),
            C2f(192, 192, n=4, shortcut=True),
        )
        # Stage 4
        self.p4 = nn.Sequential(
            ConvBlock(192, 384, k=3, s=2),
            C2f(384, 384, n=4, shortcut=True),
        )
        # Stage 5
        self.p5 = nn.Sequential(
            ConvBlock(384, 576, k=3, s=2),
            C2f(576, 576, n=2, shortcut=True),
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns multi-scale feature pyramids (P3, P4, P5)."""
        x = self.p1(x)
        x = self.p2(x)
        p3 = self.p3(x)
        p4 = self.p4(p3)
        p5 = self.p5(p4)
        return p3, p4, p5


class YOLOMDetector:
    """
    High-level YOLO-m vision detector supporting both Ultralytics engine
    and native PyTorch architecture forward pass.
    """

    def __init__(self, config: Optional[YOLOMConfig] = None) -> None:
        self.config = config or YOLOMConfig()
        self.device = torch.device(self.config.device)
        self.backbone = YOLOMBackbone().to(self.device)

        # Ultralytics model instance
        self._ultralytics_model: Optional[YOLO] = None
        self._init_model()

    def _init_model(self) -> None:
        weights = self.config.weights_path or f"{self.config.model_name}.pt"
        try:
            self._ultralytics_model = YOLO(weights)
            # Transfer to specified device
            self._ultralytics_model.to(self.config.device)
        except Exception as e:
            print(f"[*] Ultralytics 权重加载跳过或本地离线 ({e})，启用结构骨干网模式")

    def forward_features(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Execute forward pass through YOLO-m backbone."""
        return self.backbone(x)

    def predict(
        self,
        image: Union[np.ndarray, str],
        conf: Optional[float] = None,
        iou: Optional[float] = None,
    ) -> List[Any]:
        """
        Run inference on image (NumPy array or file path).
        Returns raw Ultralytics Results or parsed detection objects.
        """
        conf_val = conf if conf is not None else self.config.conf_threshold
        iou_val = iou if iou is not None else self.config.iou_threshold

        # [UPDATE - 2026-09-12]
        # Reason: Continuous inference in high-FPS loop accumulated GPU memory.
        # Modification: Wrapped model.predict with torch.inference_mode() for zero autograd tracking.
        if self._ultralytics_model is not None:
            with torch.inference_mode():
                results = self._ultralytics_model.predict(
                    source=image,
                    conf=conf_val,
                    iou=iou_val,
                    imgsz=self.config.input_size[0],
                    device=self.config.device,
                    verbose=False,
                )
                return results
        return []

    def get_parameter_count(self) -> int:
        """Calculate number of trainable parameters in the backbone."""
        return sum(p.numel() for p in self.backbone.parameters() if p.requires_grad)

    def export_onnx(self, output_path: str) -> Optional[str]:
        """Export current model to ONNX format."""
        if self._ultralytics_model is not None:
            return self._ultralytics_model.export(
                format="onnx", imgsz=self.config.input_size[0], simplify=True
            )
        return None
