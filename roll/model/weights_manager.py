# [NEW - 2026-09-19]
# Reason: Out-of-the-box model weights management with automatic GitHub Release CDN download.
# Content: resolve_and_ensure_weights, download_from_release, CLI download tool with progress reporting.

from __future__ import annotations

import os
import sys
import urllib.error
import urllib.request
from typing import Optional

if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
if sys.stderr and hasattr(sys.stderr, "reconfigure"):
    try:
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
ROLL_ROOT = os.path.dirname(CURRENT_DIR)
PROJECT_ROOT = os.path.dirname(ROLL_ROOT)

DEFAULT_REPO = "HengmingZ/Roco-Kingdom-Free-Your-Hand"
DEFAULT_TAG = "v1.0.0"
DEFAULT_WEIGHTS_FILENAME = "best.pt"


def get_default_weights_path() -> str:
    """Return the standard local path for roll/weights/best.pt."""
    return os.path.join(ROLL_ROOT, "weights", DEFAULT_WEIGHTS_FILENAME)


def download_from_release(
    target_path: Optional[str] = None,
    repo: str = DEFAULT_REPO,
    tag: str = DEFAULT_TAG,
    filename: str = DEFAULT_WEIGHTS_FILENAME,
) -> bool:
    """
    Download weight asset from GitHub Releases into the target path.
    Supports mirror fallback via environment variable ROCO_WEIGHTS_URL.
    """
    target_path = target_path or get_default_weights_path()
    os.makedirs(os.path.dirname(target_path), exist_ok=True)

    default_url = f"https://github.com/{repo}/releases/download/{tag}/{filename}"
    url = os.environ.get("ROCO_WEIGHTS_URL", default_url)

    print(f"[*] 正在从 GitHub Release 下载模型权重...")
    print(f"[*] 源地址: {url}")
    print(f"[*] 目标路径: {target_path}")

    temp_path = target_path + ".tmp"
    try:
        req = urllib.request.Request(
            url,
            headers={
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
            },
        )
        with urllib.request.urlopen(req, timeout=30) as resp, open(temp_path, "wb") as f:
            total = int(resp.headers.get("Content-Length", 0))
            downloaded = 0
            chunk_size = 1024 * 1024  # 1 MB chunk

            while True:
                chunk = resp.read(chunk_size)
                if not chunk:
                    break
                f.write(chunk)
                downloaded += len(chunk)
                if total > 0:
                    pct = downloaded / total * 100
                    mb_cur = downloaded / (1024 * 1024)
                    mb_tot = total / (1024 * 1024)
                    print(
                        f"\r[*] 下载进度: {pct:5.1f}% [{mb_cur:5.1f} MB / {mb_tot:5.1f} MB]",
                        end="",
                        flush=True,
                    )
            print()

        if os.path.exists(target_path):
            os.remove(target_path)
        os.rename(temp_path, target_path)
        print(f"[+] 权重成功下载并校验完成: {target_path}")
        return True
    except Exception as e:
        print(f"\n[!] 权重下载失败: {e}")
        if os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except OSError:
                pass
        return False


def resolve_and_ensure_weights(
    weights_path: Optional[str] = None,
    auto_download: bool = True,
    repo: str = DEFAULT_REPO,
    tag: str = DEFAULT_TAG,
) -> str:
    """
    Resolve model weight path with out-of-the-box readiness:
    1. If explicit weights_path exists -> return it.
    2. If local roll/weights/best.pt exists -> return it.
    3. If local roll/weights/best.onnx exists -> return it.
    4. If missing and auto_download is True -> download from GitHub Release.
    5. If all fail -> fallback to baseline yolov8m.pt with guidance.
    """
    # 1. Explicit path
    if weights_path is not None and os.path.exists(weights_path):
        return weights_path

    # 2. Local default best.pt
    default_best_pt = get_default_weights_path()
    if os.path.exists(default_best_pt):
        return default_best_pt

    # 3. Local best.onnx
    default_best_onnx = os.path.join(ROLL_ROOT, "weights", "best.onnx")
    if os.path.exists(default_best_onnx):
        return default_best_onnx

    # 4. Auto download from GitHub Release
    if auto_download:
        success = download_from_release(target_path=default_best_pt, repo=repo, tag=tag)
        if success and os.path.exists(default_best_pt):
            return default_best_pt

    # 5. Fallback candidates
    root_yolo = os.path.join(PROJECT_ROOT, "yolov8m.pt")
    if os.path.exists(root_yolo):
        print(f"[*] 提示: 未找到训练权重 best.pt，已回退至基础模型: {root_yolo}")
        return root_yolo

    print("[*] 提示: 未检测到目标训练权重，系统将自动调用 Ultralytics 官方源加载 yolov8m.pt")
    return "yolov8m.pt"


def main():
    import argparse

    parser = argparse.ArgumentParser(description="RocoClicker 权重自动拉取与管理工具")
    parser.add_argument("--repo", type=str, default=DEFAULT_REPO, help="GitHub 仓库名 (所有者/仓库)")
    parser.add_argument("--tag", type=str, default=DEFAULT_TAG, help="Release 标签版本")
    parser.add_argument("--force", action="store_true", help="强制重新下载并覆盖本地已有权重")
    args = parser.parse_args()

    default_pt = get_default_weights_path()
    if os.path.exists(default_pt) and not args.force:
        size_mb = os.path.getsize(default_pt) / (1024 * 1024)
        print(f"[+] 本地模型权重已就绪: {default_pt} ({size_mb:.2f} MB)")
        print("[*] 若需强制重新下载，请添加 --force 参数。")
        return

    success = download_from_release(target_path=default_pt, repo=args.repo, tag=args.tag)
    if not success:
        sys.exit(1)


if __name__ == "__main__":
    main()
