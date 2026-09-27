# 架构与实现方案：RocoClicker/roll 模块升级（YOLOv8m + 屏幕采集 GUI）

本方案针对用户提出的需求：在 `d:\sProject\GameTraining\RocoClicker\roll\` 目录下进行架构升级，创建 `model` 架构体系（包含 YOLO-m 网络架构与推理流水线），并开发一个支持自定义屏幕选择与采集时间间隔的交互式屏幕 Capture GUI 工具。

---

## 一、用户与系统约束确认

1. **工作空间限制**：所有生成代码与配置文件全部放置在 `d:\sProject\GameTraining\RocoClicker\` 对应真实工作空间内，严禁写入临时目录。
2. **执行准则**：所有脚本必须支持通过 `uv run` 直接执行。
3. **依赖库一致性（uv-smart-add）**：
   - 检查历史 `pyproject.toml`（特别是 `D:\sProject\GameTraining\RocoCapturor` 和 `D:\uv_pip`）：
     - Python: `>=3.12`
     - PyTorch: `torch==2.11.0+cu128`, `torchvision==0.26.0+cu128`
     - 额外源: `https://download.pytorch.org/whl/cu128`
     - 核心依赖: `ultralytics>=8.4.149`, `opencv-python>=4.10.0.84`, `numpy>=2.0.0`, `pillow`
   - 为 `RocoClicker` 配置规范的 `pyproject.toml`，复用同款环境，使 `roll` 模块具备高性能 CUDA 加速与 Ultralytics YOLO 原生支持。

---

## 二、目标目录架构设计

```
d:\sProject\GameTraining\RocoClicker\
├── pyproject.toml              # [NEW] 项目标准依赖配置 (CUDA 12.8 + torch 2.11 + ultralytics + cv2)
└── roll\
    ├── __init__.py             # [NEW] roll 包入口
    ├── README.md               # [UPDATE] 增加 model 与 capture 模块使用文档
    ├── test_camera_roll.py     # [EXISTING] 已验证的鼠标移动与镜头转动测试脚本
    ├── capture\                # [NEW] 屏幕捕捉与 GUI 工具模块
    │   ├── __init__.py
    │   ├── win32_defs.py       # Win32 GDI 结构体定义与屏幕信息数据类
    │   ├── screen_grabber.py   # 高性能多显示器屏幕捕获引擎（原生 Win32 GDI + 内存位图）
    │   └── capture_gui.py      # 现代化 Tkinter GUI 工具（屏幕选择 + 自定义间隔定时抓取 + 预览 + 导出）
    ├── model\                  # [NEW] 模型网络与目标检测架构模块
    │   ├── __init__.py
    │   ├── yolo_m.py           # YOLO-m 网络封装与架构抽象（支持 YOLOv8m/YOLO11m 加载、推理、ONNX导出与前向分析）
    │   ├── pipeline.py         # 镜头画面目标检测 Pipeline（ScreenGrabber -> YOLO-m -> BoundingBox/视线偏差向量）
    │   └── test_model.py       # 模型网络基准测试与结构验证测试脚本
    └── run_capture_gui.bat     # [NEW] 一键快速启动屏幕采集 GUI 工具的便捷批处理脚本
```

---

## 三、核心组件实现细节

### 1. `roll/capture/`：多显示器与定时采集 GUI
- **`win32_defs.py` & `screen_grabber.py`**：
  - 采用原生 Windows Win32 GDI（`EnumDisplayMonitors`, `CreateDCW`, `BitBlt`, `GetDIBits`）。
  - 支持多显示器枚举、主显示器标识、各屏幕真实物理分辨率获取与动态切换抓取。
  - 抓取后转换为连续内存数组，支持一键无损保存（PNG/JPEG/BMP）。
- **`capture_gui.py`**：
  - 采用现代深色主题设计（与连点器主色调统一）。
  - **屏幕选择（Screen Selector）**：下拉框动态列出本机所有屏幕（含分辨率、设备名及主副屏标识），切换即生效。
  - **时间间隔设置（Interval Setting）**：
    - 单次立即截屏按钮（支持 3 秒延迟倒计时切回游戏）。
    - 连续定时采集模式（支持输入/下拉选择采集时间间隔：如 `0.2s`、`0.5s`、`1.0s`、`2.0s`、`5.0s` 等自定义秒数）。
    - 提供「开始定时采集」/「停止采集」状态切换，后台守护线程精确调度，避免阻塞 UI。
  - **实时缩略预览与历史记录**：右侧画廊/列表显示最新抓取的截图缩略图、分辨率、时间戳、文件大小及存储路径。
  - **快捷操作**：支持一键打开存储文件夹。

### 2. `roll/model/`：YOLO-m 模型架构
- **`yolo_m.py`**：
  - 封装 YOLO-m 系列模型（支持 YOLOv8m 加载或自定义结构），兼容 `ultralytics` 标准架构。
  - 提供纯 PyTorch Module 级架构定义和前向推理接口，定义输入图像张量尺寸 `[B, 3, H, W]`、特征提取骨干网（Backbone）、颈部网络（Neck）以及解耦检测头（Detect Head）。
  - 具备自动硬件环境探测（优先使用 CUDA，回退到 CPU）。
  - 提供参数统计（Params）、FLOPs 估算以及导出为 ONNX 格式的标准化方法。
- **`pipeline.py`**：
  - 连接 `ScreenGrabber` 与 `YOLO-m`。
  - 接收指定屏幕的实时画面输入，经过尺寸归一化，推理输出目标边界框 `(x1, y1, x2, y2, conf, cls_id)`。
  - 计算目标中心点与当前主屏幕/游戏十字准星中心的像素偏差向量 `(offset_x, offset_y)`，为下一步“带着镜头瞄准目标”提供相对增量输入。

---

## 四、验证计划

1. **环境与依赖确认**：
   - 配置 `RocoClicker/pyproject.toml`，确保能够调用 CUDA 与 Ultralytics。
2. **模型架构验证 (`test_model.py`)**：
   - 运行 `uv run roll/model/test_model.py`，验证 YOLO-m 网络实例化、参数量统计、Dummy Tensor 前向传递及输出维度正确性。
3. **屏幕采集 GUI 验证 (`capture_gui.py`)**：
   - 运行 `uv run roll/capture/capture_gui.py`，验证多屏幕枚举、时间间隔调节、定时后台抓取与图片落地。
