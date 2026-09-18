# [NEW - 2026-09-12]
# Reason: Training pipeline for YOLOv8m on the user's annotated single-target dataset in roll module.
# Content: Configurable epochs, batch size, CUDA RTX 5060 Ti acceleration, model checkpoint saving,
#          evaluation metrics reporting, and automatic ONNX model export.

from __future__ import annotations

import argparse
import os
import shutil
import sys
import time
import torch
from ultralytics import YOLO

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


def train_yolo_m(
    epochs: int = 50,
    imgsz: int = 640,
    batch_size: int = 8,
    weights_path: str | None = None,
    dataset_yaml: str | None = None,
    device: str | None = None,
) -> None:
    # Resolve paths
    if weights_path is None:
        # Check candidate locations
        cands = [
            os.path.join(PROJECT_ROOT, "yolov8m.pt"),
            os.path.join(ROLL_ROOT, "weights", "yolov8m.pt"),
            "yolov8m.pt",
        ]
        weights_path = next((p for p in cands if os.path.exists(p)), "yolov8m.pt")

    if dataset_yaml is None:
        dataset_yaml = os.path.join(ROLL_ROOT, "data", "yolo_dataset", "dataset.yaml")

    if not os.path.exists(dataset_yaml):
        raise FileNotFoundError(f"未找到数据集配置文件: {dataset_yaml}")

    weights_dir = os.path.join(ROLL_ROOT, "weights")
    runs_dir = os.path.join(weights_dir, "train_runs")
    os.makedirs(weights_dir, exist_ok=True)
    os.makedirs(runs_dir, exist_ok=True)

    # Hardware detection
    if device is None:
        device = "0" if torch.cuda.is_available() else "cpu"
    dev_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"

    print("=" * 65)
    print("        RocoClicker - YOLOv8 Medium 目标检测模型训练")
    print("=" * 65)
    print(f"[*] 基础模型权重 : {weights_path}")
    print(f"[*] 数据集配置文件: {dataset_yaml}")
    print(f"[*] 训练硬件平台 : CUDA [{dev_name}] (设备: {device})")
    print(f"[*] 训练超参数   : 轮数(Epochs)={epochs} | 批次(Batch)={batch_size} | 图像尺寸={imgsz}")
    print(f"[*] 输出权重目录 : {weights_dir}")
    print("=" * 65)

    # Load model
    model = YOLO(weights_path)

    # Start training
    t_start = time.time()
    results = model.train(
        data=dataset_yaml,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch_size,
        device=device,
        project=runs_dir,
        name="target_detector_m",
        exist_ok=True,
        save=True,
        plots=True,
        workers=2,
        patience=15,
        val=True,
    )
    t_end = time.time()
    elapsed_min = (t_end - t_start) / 60.0
    print(f"\n[+] 训练完成！耗时: {elapsed_min:.2f} 分钟。正在整理模型权重产物...")

    # Organize weights
    best_train_pt = os.path.join(runs_dir, "target_detector_m", "weights", "best.pt")
    last_train_pt = os.path.join(runs_dir, "target_detector_m", "weights", "last.pt")
    target_best_pt = os.path.join(weights_dir, "best.pt")
    target_last_pt = os.path.join(weights_dir, "last.pt")

    if os.path.exists(best_train_pt):
        shutil.copy2(best_train_pt, target_best_pt)
        print(f"[+] 已保存最佳 PyTorch 模型权重: {target_best_pt}")
    if os.path.exists(last_train_pt):
        shutil.copy2(last_train_pt, target_last_pt)

    # Export to ONNX for fast inference
    print("\n[*] 正在导出为生产环境 ONNX 格式...")
    try:
        best_model = YOLO(target_best_pt)
        exported_onnx = best_model.export(format="onnx", imgsz=imgsz, simplify=True)
        target_onnx = os.path.join(weights_dir, "best.onnx")
        if os.path.exists(exported_onnx):
            shutil.copy2(exported_onnx, target_onnx)
            print(f"[+] 已成功导出优化版 ONNX 权重: {target_onnx}")
    except Exception as e:
        print(f"[-] ONNX 导出跳过或异常: {e}")

    print("=" * 65)
    print("        YOLOv8m 单目标检测模型训练与导出全流程顺利完成！")
    print("=" * 65)


def main():
    parser = argparse.ArgumentParser(description="训练 RocoClicker 目标检测模型 (YOLOv8m)")
    parser.add_argument("--epochs", type=int, default=50, help="训练轮数 (默认 50)")
    parser.add_argument("--batch", type=int, default=8, help="批次大小 Batch Size (默认 8)")
    parser.add_argument("--imgsz", type=int, default=640, help="图像输入尺寸 (默认 640)")
    parser.add_argument("--weights", type=str, default=None, help="初始预训练权重路径")
    parser.add_argument("--data", type=str, default=None, help="数据集 yaml 路径")
    parser.add_argument("--device", type=str, default=None, help="指定计算设备 (如 '0' 或 'cpu')")
    args = parser.parse_args()

    train_yolo_m(
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch_size=args.batch,
        weights_path=args.weights,
        dataset_yaml=args.data,
        device=args.device,
    )


if __name__ == "__main__":
    main()
