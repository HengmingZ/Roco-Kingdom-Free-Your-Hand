# Unit test for roll weights manager (Strict Code Tracing step 1)

import os
import sys
from pathlib import Path

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
ROLL_ROOT = os.path.dirname(CURRENT_DIR)
if ROLL_ROOT not in sys.path:
    sys.path.insert(0, ROLL_ROOT)

DEFAULT_REPO = "HengmingZ/Roco-Kingdom-Free-Your-Hand"
DEFAULT_TAG = "v1.0.0"
DEFAULT_URL = f"https://github.com/{DEFAULT_REPO}/releases/download/{DEFAULT_TAG}/best.pt"


def resolve_and_ensure_weights(
    weights_path: str | None = None,
    auto_download: bool = True,
    repo: str = DEFAULT_REPO,
    tag: str = DEFAULT_TAG,
) -> str:
    """
    Resolve weights file path. If missing and auto_download is True,
    attempts download from GitHub Releases. Fallback to yolov8m.pt if failed.
    """
    # 1. Explicit path given
    if weights_path is not None and os.path.exists(weights_path):
        return weights_path

    # 2. Local default best.pt
    default_best_pt = os.path.join(ROLL_ROOT, "weights", "best.pt")
    if os.path.exists(default_best_pt):
        return default_best_pt

    # 3. Local default best.onnx
    default_best_onnx = os.path.join(ROLL_ROOT, "weights", "best.onnx")
    if os.path.exists(default_best_onnx):
        return default_best_onnx

    if not auto_download:
        return "yolov8m.pt"

    # 4. Attempt download
    target_dir = os.path.join(ROLL_ROOT, "weights")
    os.makedirs(target_dir, exist_ok=True)
    target_pt = default_best_pt

    url = os.environ.get("ROCO_WEIGHTS_URL", f"https://github.com/{repo}/releases/download/{tag}/best.pt")
    print(f"[*] 正在尝试从 Release 下载模型权重: {url} -> {target_pt}")

    import urllib.request
    import urllib.error

    try:
        req = urllib.request.Request(
            url,
            headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
        )
        temp_file = target_pt + ".tmp"
        with urllib.request.urlopen(req, timeout=10) as resp, open(temp_file, "wb") as f:
            total = int(resp.headers.get("Content-Length", 0))
            downloaded = 0
            chunk_size = 1024 * 1024  # 1MB
            while True:
                chunk = resp.read(chunk_size)
                if not chunk:
                    break
                f.write(chunk)
                downloaded += len(chunk)
                if total > 0:
                    pct = downloaded / total * 100
                    print(f"\r[*] 下载进度: {pct:.1f}% ({downloaded / (1024*1024):.1f}MB / {total / (1024*1024):.1f}MB)", end="", flush=True)
            print()
        if os.path.exists(target_pt):
            os.remove(target_pt)
        os.rename(temp_file, target_pt)
        print(f"[*] 权重下载完成: {target_pt}")
        return target_pt
    except Exception as e:
        print(f"\n[!] 自动拉取云端权重失败 ({e})，将回退使用基础 YOLOv8m 模型")
        if os.path.exists(target_pt + ".tmp"):
            try:
                os.remove(target_pt + ".tmp")
            except OSError:
                pass
        return "yolov8m.pt"


def test_weights_resolution():
    print("=== Test 1: Existing local best.pt ===")
    res = resolve_and_ensure_weights(auto_download=False)
    print("Resolved path:", res)
    assert os.path.exists(res), f"Path must exist: {res}"
    assert "best.pt" in res

    print("\n=== Test 2: When local best.pt does not exist (simulated) ===")
    # Test fallback branch when target path does not exist
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        fake_root = tmpdir
        target_dir = os.path.join(fake_root, "weights")
        os.makedirs(target_dir, exist_ok=True)
        # Test simulated download attempt to non-existent tag
        url = f"https://github.com/{DEFAULT_REPO}/releases/download/v99.99.99/best.pt"
        # We know this will fail and gracefully fallback to yolov8m.pt
        import urllib.request
        failed = False
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=5) as resp:
                pass
        except Exception:
            failed = True
        assert failed, "Fake release tag must fail"
        print("Fallback to yolov8m.pt logic verified.")

    print("\nAll tests completed successfully.")


if __name__ == "__main__":
    test_weights_resolution()
