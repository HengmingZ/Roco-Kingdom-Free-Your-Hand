# 镜头转动、屏幕采集与 YOLO-m 视觉闭环体系 (Roll Vision & PID Control System)

本模块位于 `RocoClicker/roll`，集成了 **内核级鼠标相对移动带镜头旋转**、**多显示器屏幕采集 GUI 工具**、**单目标急速 YOLO 标注工具**、**YOLO-m (YOLOv8 Medium) 模型训练** 与 **2D 视觉伺服 PID 自动居中跟瞄控制系统**。

---

## 目录结构

```
roll/
├── __init__.py                 # roll 顶层模块包
├── README.md                   # 模块架构与使用文档
├── implementation_plan.md      # 架构设计与实现方案
├── run_capture_gui.bat         # 屏幕采集 GUI 工具一键启动脚本
├── run_annotator_gui.bat       # 单目标标注 GUI 工具一键启动脚本
├── run_train.bat               # YOLOv8m 模型训练一键启动脚本
├── run_pid_aim.bat             # 视觉伺服 PID 居中瞄准一键启动脚本 (终端仪表盘模式)
├── run_aim_gui.bat             # 视觉伺服 PID 自动瞄准与实时调参 GUI 控制台一键启动脚本
├── test_camera_roll.py         # 镜头转动与鼠标画圆测试脚本 (已验证)
│
├── capture/                    # 屏幕捕捉与采集工具模块
│   ├── __init__.py
│   ├── win32_defs.py           # Win32 GDI 结构体、多显示器信息定义
│   ├── screen_grabber.py       # 高性能多显示器屏幕捕获引擎 (Win32 GDI 原生)
│   └── capture_gui.py          # 交互式屏幕采集 GUI 工具 (支持多屏选择与自定义间隔)
│
├── annotation/                 # 数据集标注工具模块
│   ├── __init__.py
│   └── annotator_gui.py        # 单目标 (target) 急速标注器 (快捷键保存切图、一键导出)
│
├── model/                      # 视觉模型与目标检测网络模块
│   ├── __init__.py
│   ├── yolo_m.py               # YOLO-m 网络架构抽象 (含骨干网 C2f、Backbone、推理解析)
│   ├── pipeline.py             # 视觉推理流水线 (ScreenGrabber -> YOLO-m -> 准星偏移量向量)
│   ├── train.py                # YOLOv8m 模型微调训练与权重导出脚本
│   └── test_model.py           # 模型参数量、前向张量维度及 CUDA 推理基准测试
│
├── control/                    # 视觉伺服 PID 自动居中控制与多目标锁定模块
│   ├── __init__.py
│   ├── pid.py                  # 双轴 (Yaw/Pitch) PID 控制器 (抗饱和、死区防抖、自适应微分滤波)
│   ├── tracker.py              # TargetTracker 连续锁定引擎 (运动学关联、前馈补偿、失靶缓冲)
│   ├── aim_controller.py       # 闭环视觉控制器 (连接 Pipeline、DualAxisPID、Tracker 与 Interception 驱动)
│   ├── aim_gui.py              # 交互式自动瞄准与实时调参 GUI 控制台 (即拖即生效)
│   ├── run_aim_loop.py         # 实时跟瞄运行器 (支持 F1/N/F3/ESC 热键与终端仪表盘)
│   ├── test_pid.py             # PID 稳定性与收敛性离线仿真测试
│   └── test_tracker.py         # 多目标锁定、交叉免疫与遮挡恢复单元测试
│
└── weights/                    # 模型权重与训练产物
    ├── best.pt                 # 最佳 PyTorch 微调权重 (mAP@50: 89.0%)
    ├── last.pt                 # 最终轮次权重
    ├── best.onnx               # 优化导出的生产环境 ONNX 推理权重 (单帧耗时 ~3ms)
    └── train_runs/             # 训练指标曲线与日志
```

---

## 核心功能与使用说明

### 1. 视觉伺服 PID 自动跟瞄与实时调参 GUI 控制台

提供完整的图形化控制界面，允许在跟瞄运行或游戏过程中直接拖动滑块实时调节 PID 响应与阻尼参数：

#### 启动方式：
```powershell
# 推荐方式 (通过 uv run 执行图形控制台)
uv run roll/control/aim_gui.py

# 或双击批处理脚本:
roll/run_aim_gui.bat
```

#### GUI 特性：
- **即拖即生效**：无需重启程序，实时滑块调节水平 $K_p/K_d$、垂直 $K_p/K_d$、锁定死区半径（Deadband）与检测置信度。
- **视觉追踪 + 自动连点 (Click While Tracking)**：
  - **并发执行**：在向 Interception 注入镜头旋转位移 $dx, dy$ 的同时，非阻塞注入左键脉冲点击（15ms 极简脉冲状态机，零锁帧零顿挫）。
  - **模式可选**：支持“边转镜头边连点”（即时打击移动目标）与“仅在锁定(LOCKED)死区内连点”（高精度点射）。
  - **频次可调**：支持 0.03s ~ 0.50s（约 2 ~ 33 CPS，默认 0.15s ≈ 6.7 CPS）无级调节。
- **环境切换**：支持多显示器下拉切换与按住右键/自由准星模式切换。
- **视觉监控反馈**：可选开启实时画面缩略预览（带目标标注框、中心十字准星与指向偏差向量），并实时监控误差、驱动输出、连点状态与 FPS。
- **全局热键（游戏切至前台仍可随时按键触发）**：
  - **`F1` 键**：启动 / 恢复 PID 自动居中跟瞄与连点
  - **`F7` 键**：**紧急暂停**（立即释放右键与左键，停止所有注入，同时支持 `N`/`F2` 备用）
  - **`F3` 键**：**手动切换目标**（强制解除当前锁定，重新选择离准星最近的目标）
  - **`ESC` 键**：紧急挂起 / 退出

---

### 2. 终端低延迟仪表盘模式 (CLI Mode)

若追求极限帧率与零额外 UI 渲染开销，可直接运行终端模式：
```powershell
uv run roll/control/run_aim_loop.py
# 或带自定义连点参数:
uv run roll/control/run_aim_loop.py --click-interval 0.10 --click-when-locked
# 或双击 roll/run_pid_aim.bat
```

---

### 2. 模型训练与开箱即用权重 (YOLOv8 Medium Weights)

基于标注好的 130 张样本，在 NVIDIA GeForce RTX 5060 Ti 上完成 50 轮全量微调：
- **最终验证集指标**：**mAP@50: 89.0%**，**精确率: 81.0%**，**召回率: 77.4%**，**推理速度: 3.0 ms/帧**。
- **权重文件存放位置**：`roll/weights/best.pt`。
- **官方 Release 发布页**：[v1.0.0 Release 页面](https://github.com/HengmingZ/Roco-Kingdom-Free-Your-Hand/releases/tag/v1.0.0)（或直接下载附件 [best.pt (52.0 MB)](https://github.com/HengmingZ/Roco-Kingdom-Free-Your-Hand/releases/download/v1.0.0/best.pt)）。
- **开箱即用拉取**：系统在首次启动时会自动检测并从 GitHub Release CDN 静默下载补齐，亦可随时手动执行：
  ```powershell
  uv run roll/download_weights.py
  ```
  或双击运行 `roll/run_download_weights.bat`。

---

### 3. 单目标急速标注工具 (Single-Object Annotator GUI)

```powershell
uv run roll/annotation/annotator_gui.py
```

---

### 4. 屏幕视觉采集工具 (Screen Capture GUI)

```powershell
uv run roll/capture/capture_gui.py
```

---

## 🤝 致谢与结对协作 (Acknowledgements)

- **主导开发 (Author)**：[HengmingZ](https://github.com/HengmingZ)
- **架构结对 (AI Pair Programmer)**：[Google DeepMind Antigravity](https://deepmind.google/)

