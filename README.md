# 🎮 Roco-Kingdom-Free-Your-Hand (洛克王国：解放双手)

<p align="center">
  <b>基于深度学习目标检测 (YOLO-m) 与双轴视觉伺服 PID 的全自动瞄准追踪与脉冲连点系统</b><br>
  <i>Deep Learning Vision (YOLOv8 Medium) + Dual-Axis Visual Servoing PID Aiming & Non-blocking Pulse Auto-Clicker</i>
</p>

---

## 🌟 核心特性 (Key Features)

- 🎯 **双轴视觉伺服 PID 闭环控制 (Visual Servoing PID)**：
  - 水平 (Yaw) 与垂直 (Pitch) 独立自适应微分滤波 PID 控制器。
  - 动态抗积分饱和（Anti-windup）与死区（Deadband）防抖设计，准星平滑居中不甩头。
- ⚡ **多目标连续稳定锁定算法 (TargetTracker Engine)**：
  - 基于空间欧氏距离、IoU 重合度与长宽比一致性的多维运动学代价匹配（Cost Matrix）。
  - 集成**相机自我运动前馈补偿 (Ego-Motion Feedforward)** 与 8 帧失靶遮挡缓冲，彻底解决多个目标同屏时的频繁切目标与抖动问题。
- 🖱️ **无阻塞脉冲并发连点机制 (Concurrent Click While Tracking)**：
  - 15ms 硬件级极简脉冲状态机，一边转动镜头平滑跟瞄，一边全速注入左键点击，零锁帧零顿挫。
  - 连点频率支持无级调节（0.03s ~ 0.50s，即 2 ~ 33 CPS，默认 0.15s ≈ 6.7 CPS）。
- 🎛️ **实时交互式调参 GUI 控制台 (`roll/control/aim_gui.py`)**：
  - 滑块即拖即生效：在线调整 $K_p$ / $K_d$ 增益、锁定死区半径、置信度阈值及连点间隔。
  - 画面缩略图实时监控：可视化渲染目标框、准星中心与偏差向量。
  - **100Hz 独立高频全局热键守护线程**：脱离重度推理循环，`F7` 紧急暂停实现 `<5ms` 超灵敏熔断，并支持 `F7` / `N` / `ESC` 多键冗余容错。
- 🛠️ **一体化工业级工具链**：
  - 屏幕截图多屏采集器 (`roll/capture/capture_gui.py`)
  - 单目标急速标注器 (`roll/annotation/annotator_gui.py`)
  - YOLOv8m 微调训练与轻量导出脚本 (`roll/model/train.py`)
  - 内核驱动安装程序与动态链接库全内置 (`roll/interception.dll` & `roll/driver_installer/`)

---

## 📂 项目结构 (Project Structure)

```text
RocoClicker/
├── pyproject.toml              # UV 项目与依赖定义 (PyTorch CUDA 12.8 / Ultralytics)
├── uv.lock                     # 依赖版本锁定清单
├── README.md                   # 项目顶层说明文档
├── .gitignore                  # 严谨的大文件/模型/临时文件过滤规则
│
└── roll/                       # 核心视觉、控制与驱动系统
    ├── README.md               # roll 模块技术细节与实现架构
    ├── interception.dll        # 内核级鼠标位移与按键驱动动态库
    ├── install_driver.bat      # 一键安装 Interception 驱动脚本
    │
    ├── driver_installer/       # Interception 硬件级驱动安装器
    │   └── install-interception.exe
    │
    ├── capture/                # 屏幕采集引擎
    │   ├── screen_grabber.py   # Win32 GDI 原生多屏低延迟抓屏
    │   └── capture_gui.py      # 交互式屏幕采样 GUI
    │
    ├── annotation/             # 数据集标注工具
    │   └── annotator_gui.py    # 单目标急速标注器 (快捷键保存切图)
    │
    ├── model/                  # 目标检测网络与推理流水线
    │   ├── yolo_m.py           # YOLO-m 架构封装
    │   ├── pipeline.py         # 视觉流水线 (抓屏 -> YOLO -> 准星偏差)
    │   └── train.py            # YOLOv8m 微调训练脚本
    │
    ├── control/                # 闭环自动瞄准控制器与 GUI
    │   ├── pid.py              # 双轴自适应 PID 控制器
    │   ├── tracker.py          # TargetTracker 连续锁定引擎
    │   ├── aim_controller.py   # 视觉闭环控制器核心
    │   ├── aim_gui.py          # 实时调参图形化控制台
    │   └── run_aim_loop.py     # 终端超低延迟仪表盘运行器
    │
    └── weights/                # 模型权重存放目录 (.gitkeep)
```

---

## 🚀 快速上手 (Quick Start)

### 1. 环境准备 (Prerequisites)

本项目强制使用极速 Python 包管理工具 **[`uv`](https://docs.astral.sh/uv/)**：

```powershell
# 1. 克隆代码仓库
git clone https://github.com/HengmingZ/Roco-Kingdom-Free-Your-Hand.git
cd Roco-Kingdom-Free-Your-Hand

# 2. 自动同步并安装全部依赖 (包含 PyTorch CUDA 支持)
uv sync
```

### 2. 安装 Interception 内核鼠标驱动

驱动只需在首次使用时安装一次：
1. 以**管理员身份**打开终端或右键运行：
   ```powershell
   roll\install_driver.bat
   ```
2. 根据提示重启电脑使驱动生效。

### 3. 模型权重 (开箱即用 / Out of the Box)

本项目提供完整的训练权重自动下载与分发支持：
* **自动静默补齐（推荐）**：启动 `aim_gui.py` 或 `run_aim_loop.py` 时，若检测到本地缺失 `roll/weights/best.pt`，系统将自动从 GitHub Release 资产中下载微调模型，完全开箱即用。
* **手动一键下载**：亦可随时运行独立下载工具：
  ```powershell
  uv run roll/download_weights.py
  ```
  或双击运行批处理脚本：`roll/run_download_weights.bat`。
* **国内镜像加速（可选）**：如遇 GitHub 连接受限，可在终端指定环境变量下载：
  ```powershell
  $env:ROCO_WEIGHTS_URL="https://ghfast.top/https://github.com/HengmingZ/Roco-Kingdom-Free-Your-Hand/releases/download/v1.0.0/best.pt"
  uv run roll/download_weights.py
  ```

---

## 🎮 运行控制 (Usage)

### 方式 A：图形化控制台（推荐，支持边玩边调参）

```powershell
uv run roll/control/aim_gui.py
```
或直接双击运行脚本：`roll/run_aim_gui.bat`。

#### 全局热键说明（游戏切至前台亦可全局触发）：
| 快捷键 | 功能 | 说明 |
| :--- | :--- | :--- |
| **`F1`** | **启动 / 恢复跟瞄** | 自动搜索并锁定最近目标，开始视觉伺服对准与脉冲连点 |
| **`F7`** | **紧急暂停** | **熔断机制**：瞬间释放所有鼠标按键并冻结 PID 输出 |
| **`N` / `ESC`** | **备用紧急暂停** | 多键冗余保障，防止笔记本 Fn 键锁定冲突 |
| **`F3`** | **切换目标** | 强制解除当前锁定，重新吸附最靠近中心的目标 |

---

### 方式 B：终端低延迟仪表盘模式 (CLI Mode)

追求极限帧率与零额外 UI 开销：
```powershell
uv run roll/control/run_aim_loop.py
```
支持丰富的启动参数：
```powershell
uv run roll/control/run_aim_loop.py --kp-x 0.45 --kp-y 0.38 --click-interval 0.10
```

---

## 🔧 模块工具链独立运行

- **启动屏幕图像采集器**：
  ```powershell
  uv run roll/capture/capture_gui.py
  ```
- **启动数据集快速标注器**：
  ```powershell
  uv run roll/annotation/annotator_gui.py
  ```
- **启动 YOLO-m 模型训练**：
  ```powershell
  uv run roll/model/train.py
  ```

---

## ⚠️ 免责声明 (Disclaimer)

本项目仅供计算机视觉算法研究、双轴运动控制学术实验与人机交互自动化学习使用。严禁将本项目用于侵犯第三方软件服务条款、网络游戏作弊或任何商业不当用途。因使用本项目产生的一切后果由使用者自行承担。
