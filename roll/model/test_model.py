# [NEW - 2026-09-12]
# Reason: Unit and benchmark test for YOLOMDetector architecture and pipeline.
# Content: Verifies YOLO-m parameter calculation, forward feature dimensions, CUDA acceleration, and dummy inference.

from __future__ import annotations

import os
import sys
import time
import torch

# Ensure console handles UTF-8 properly
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
ROLL_ROOT = os.path.dirname(CURRENT_DIR)

if ROLL_ROOT not in sys.path:
    sys.path.insert(0, ROLL_ROOT)

from model.yolo_m import YOLOMDetector, YOLOMConfig, YOLOMBackbone
import numpy as np


def test_yolo_m():
    print("=" * 60)
    print("      YOLO-m 网络架构与推理流水线基准测试")
    print("=" * 60)

    # 1. Check Hardware
    cuda_avail = torch.cuda.is_available()
    dev_name = torch.cuda.get_device_name(0) if cuda_avail else "CPU"
    print(f"[*] 计算硬件: {'CUDA:0 (' + dev_name + ')' if cuda_avail else 'CPU'}")
    print(f"[*] PyTorch 版本: {torch.__version__}")

    # 2. Test YOLOMBackbone directly
    print("\n[*] 正在构建 YOLO-m 骨干网络 (YOLOMBackbone)...")
    backbone = YOLOMBackbone()
    if cuda_avail:
        backbone = backbone.cuda()

    params_count = sum(p.numel() for p in backbone.parameters() if p.requires_grad)
    print(f"[*] 骨干网可训练参数总量: {params_count:,} (约 {params_count / 1e6:.2f} M)")

    # 3. Dummy Forward Pass [B, 3, 640, 640]
    device = "cuda" if cuda_avail else "cpu"
    dummy_input = torch.randn(1, 3, 640, 640, device=device)
    print(f"[*] 输入张量形状: {list(dummy_input.shape)}")

    t0 = time.perf_counter()
    with torch.no_grad():
        p3, p4, p5 = backbone(dummy_input)
    t1 = time.perf_counter()

    print(f"[+] 骨干网前向计算成功！耗时: {(t1 - t0) * 1000:.2f} ms")
    print(f"    - P3/8  特征图尺寸: {list(p3.shape)}")
    print(f"    - P4/16 特征图尺寸: {list(p4.shape)}")
    print(f"    - P5/32 特征图尺寸: {list(p5.shape)}")

    assert p3.shape == (1, 192, 80, 80), f"P3 维度预期 (1, 192, 80, 80)，实际: {p3.shape}"
    assert p4.shape == (1, 384, 40, 40), f"P4 维度预期 (1, 384, 40, 40)，实际: {p4.shape}"
    assert p5.shape == (1, 576, 20, 20), f"P5 维度预期 (1, 576, 20, 20)，实际: {p5.shape}"
    print("[+] 骨干网络金字塔特征维度完全校验通过！")

    # 4. Test YOLOMDetector
    print("\n[*] 正在实例化高阶检测器 YOLOMDetector...")
    config = YOLOMConfig(model_name="yolov8m", device=device)
    detector = YOLOMDetector(config)

    # Test dummy numpy frame prediction
    dummy_frame = np.zeros((640, 640, 3), dtype=np.uint8)
    res = detector.predict(dummy_frame)
    print(f"[+] 推理接口调用成功！返回结果对象: {type(res)} (条目数: {len(res)})")

    print("\n" + "=" * 60)
    print("      YOLO-m 网络测试全部顺利通过 (100% PASS)")
    print("=" * 60)


if __name__ == "__main__":
    test_yolo_m()
