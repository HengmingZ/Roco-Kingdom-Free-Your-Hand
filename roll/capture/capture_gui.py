# [NEW - 2026-09-12]
# Reason: Interactive Screen Capture GUI tool allowing users to select target screen and capture time interval.
# Content: Tkinter GUI with monitor dropdown, interval slider/combobox, manual & continuous capture, live thumbnail preview, and disk persistence.

from __future__ import annotations

import datetime
import os
import sys
import threading
import time
import tkinter as tk
from tkinter import ttk, messagebox

import cv2
import numpy as np
from PIL import Image, ImageTk

# Ensure console handles UTF-8 properly
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
ROLL_ROOT = os.path.dirname(CURRENT_DIR)
PROJECT_ROOT = os.path.dirname(ROLL_ROOT)

if ROLL_ROOT not in sys.path:
    sys.path.insert(0, ROLL_ROOT)

from capture.screen_grabber import ScreenGrabber
from capture.win32_defs import MonitorInfo


class ScreenCaptureGUI:
    """Modern Tkinter GUI for multi-monitor screen capture with customizable intervals."""

    def __init__(self, root: tk.Tk, output_dir: str | None = None) -> None:
        self.root = root
        self.root.title("RocoClicker - 屏幕视觉采集工具 (Screen Capture GUI)")
        self.root.geometry("1060x700")
        self.root.minsize(960, 620)

        # Output directory: default to roll/data/raw_captures
        if output_dir is None:
            output_dir = os.path.join(ROLL_ROOT, "data", "raw_captures")
        self.output_dir = os.path.abspath(output_dir)
        os.makedirs(self.output_dir, exist_ok=True)

        # Screen grabber & monitor list
        self.grabber = ScreenGrabber()
        self.monitors: list[MonitorInfo] = self.grabber.get_monitors()
        self.current_monitor_idx = 0
        for i, m in enumerate(self.monitors):
            if m.is_primary:
                self.current_monitor_idx = i
                break

        current_mon = self.monitors[self.current_monitor_idx]
        self.screen_w = current_mon.width
        self.screen_h = current_mon.height

        # State tracking
        self.is_continuous_capturing = False
        self._capture_thread: threading.Thread | None = None
        self._current_photo: ImageTk.PhotoImage | None = None
        self._captured_files: list[str] = []
        self._total_captured_count = 0

        self._apply_styles()
        self._build_ui()
        self._refresh_history_list()

    def _apply_styles(self) -> None:
        """Theme colors & styling."""
        self.bg_color = "#0f172a"        # Dark slate
        self.card_bg = "#1e293b"         # Card slate
        self.text_color = "#f8fafc"      # Light white
        self.accent_color = "#3b82f6"    # Primary blue
        self.success_color = "#22c55e"   # Green
        self.danger_color = "#ef4444"    # Red
        self.border_color = "#334155"

        self.root.configure(bg=self.bg_color)
        self.style = ttk.Style(self.root)
        try:
            self.style.theme_use("clam")
        except Exception:
            pass

        self.style.configure(".", background=self.bg_color, foreground=self.text_color, font=("Microsoft YaHei UI", 9))
        self.style.configure("Card.TFrame", background=self.card_bg)

    def _build_ui(self) -> None:
        """Construct GUI components."""
        # Top Header Bar
        header = tk.Frame(self.root, bg=self.card_bg, height=60, padx=16, pady=10)
        header.pack(fill=tk.X, side=tk.TOP)

        title_label = tk.Label(
            header,
            text="🎯 屏幕视觉采集控制台",
            font=("Microsoft YaHei UI", 13, "bold"),
            bg=self.card_bg,
            fg=self.text_color,
        )
        title_label.pack(side=tk.LEFT)

        # Monitor dropdown in header
        lbl_mon = tk.Label(header, text="采集屏幕:", font=("Microsoft YaHei UI", 9), bg=self.card_bg, fg=self.text_color)
        lbl_mon.pack(side=tk.LEFT, padx=(20, 6))

        self.mon_combo = ttk.Combobox(
            header,
            state="readonly",
            values=[m.label for m in self.monitors],
            font=("Microsoft YaHei UI", 9),
            width=32,
        )
        self.mon_combo.pack(side=tk.LEFT, padx=(0, 10))
        self.mon_combo.current(self.current_monitor_idx)
        self.mon_combo.bind("<<ComboboxSelected>>", self.on_monitor_changed)

        self.res_badge = tk.Label(
            header,
            text=f"{self.screen_w} × {self.screen_h}",
            font=("Consolas", 10, "bold"),
            bg="#0284c7",
            fg="#ffffff",
            padx=10,
            pady=3,
        )
        self.res_badge.pack(side=tk.LEFT)

        btn_refresh_mon = tk.Button(
            header,
            text="↻ 刷新屏幕",
            font=("Microsoft YaHei UI", 8),
            bg="#334155",
            fg="#ffffff",
            relief=tk.FLAT,
            command=self.refresh_monitors,
            cursor="hand2",
        )
        btn_refresh_mon.pack(side=tk.LEFT, padx=(8, 0))

        dir_label = tk.Label(
            header,
            text=f"保存路径: {self.output_dir}",
            font=("Microsoft YaHei UI", 8),
            bg=self.card_bg,
            fg="#94a3b8",
        )
        dir_label.pack(side=tk.RIGHT)

        # Main Body
        body = tk.Frame(self.root, bg=self.bg_color, padx=12, pady=12)
        body.pack(fill=tk.BOTH, expand=True)

        # Left Column: Controls & History
        left_col = tk.Frame(body, bg=self.bg_color, width=360)
        left_col.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 12))
        left_col.pack_propagate(False)

        # Control Panel Box
        ctrl_frame = tk.LabelFrame(
            left_col,
            text=" 采集控制与间隔配置 ",
            bg=self.card_bg,
            fg=self.text_color,
            font=("Microsoft YaHei UI", 10, "bold"),
            padx=12,
            pady=12,
        )
        ctrl_frame.pack(fill=tk.X, pady=(0, 10))

        # Single capture
        self.btn_capture = tk.Button(
            ctrl_frame,
            text="📸 立即截取当前屏幕",
            font=("Microsoft YaHei UI", 11, "bold"),
            bg=self.accent_color,
            fg="#ffffff",
            activebackground="#2563eb",
            activeforeground="#ffffff",
            relief=tk.FLAT,
            pady=8,
            cursor="hand2",
            command=self.on_capture_click,
        )
        self.btn_capture.pack(fill=tk.X, pady=(0, 8))

        self.btn_delay = tk.Button(
            ctrl_frame,
            text="⏱ 延迟 3 秒截屏 (切回游戏用)",
            font=("Microsoft YaHei UI", 9),
            bg="#334155",
            fg="#ffffff",
            activebackground="#475569",
            activeforeground="#ffffff",
            relief=tk.FLAT,
            pady=6,
            cursor="hand2",
            command=self.on_delay_capture_click,
        )
        self.btn_delay.pack(fill=tk.X, pady=(0, 12))

        # Continuous Capture Interval Section
        sep = tk.Frame(ctrl_frame, height=1, bg=self.border_color)
        sep.pack(fill=tk.X, pady=(0, 10))

        interval_label = tk.Label(
            ctrl_frame,
            text="连续采集时间间隔 (秒):",
            font=("Microsoft YaHei UI", 9, "bold"),
            bg=self.card_bg,
            fg=self.text_color,
        )
        interval_label.pack(anchor=tk.W, pady=(0, 4))

        interval_box = tk.Frame(ctrl_frame, bg=self.card_bg)
        interval_box.pack(fill=tk.X, pady=(0, 10))

        self.interval_var = tk.StringVar(value="1.0")
        self.interval_combo = ttk.Combobox(
            interval_box,
            textvariable=self.interval_var,
            values=["0.1", "0.2", "0.5", "1.0", "1.5", "2.0", "3.0", "5.0"],
            font=("Consolas", 10),
            width=10,
        )
        self.interval_combo.pack(side=tk.LEFT, padx=(0, 8))

        tk.Label(
            interval_box,
            text="秒/张 (支持手动输入)",
            font=("Microsoft YaHei UI", 8),
            bg=self.card_bg,
            fg="#94a3b8",
        ).pack(side=tk.LEFT)

        # Continuous Capture Button
        self.btn_continuous = tk.Button(
            ctrl_frame,
            text="▶ 开始定时连续采集",
            font=("Microsoft YaHei UI", 11, "bold"),
            bg=self.success_color,
            fg="#ffffff",
            activebackground="#16a34a",
            activeforeground="#ffffff",
            relief=tk.FLAT,
            pady=8,
            cursor="hand2",
            command=self.toggle_continuous_capture,
        )
        self.btn_continuous.pack(fill=tk.X, pady=(0, 10))

        self.btn_open_folder = tk.Button(
            ctrl_frame,
            text="📁 打开保存文件夹",
            font=("Microsoft YaHei UI", 9),
            bg="#334155",
            fg="#ffffff",
            activebackground="#475569",
            activeforeground="#ffffff",
            relief=tk.FLAT,
            pady=6,
            cursor="hand2",
            command=self.on_open_folder,
        )
        self.btn_open_folder.pack(fill=tk.X)

        # History List
        history_frame = tk.LabelFrame(
            left_col,
            text=" 采集样本历史 (点击预览) ",
            bg=self.card_bg,
            fg=self.text_color,
            font=("Microsoft YaHei UI", 9, "bold"),
            padx=8,
            pady=8,
        )
        history_frame.pack(fill=tk.BOTH, expand=True)

        list_scroll = tk.Scrollbar(history_frame)
        list_scroll.pack(side=tk.RIGHT, fill=tk.Y)

        self.listbox = tk.Listbox(
            history_frame,
            bg="#0f172a",
            fg="#f8fafc",
            selectbackground=self.accent_color,
            selectforeground="#ffffff",
            font=("Consolas", 9),
            relief=tk.FLAT,
            yscrollcommand=list_scroll.set,
            activestyle="none",
        )
        self.listbox.pack(fill=tk.BOTH, expand=True)
        list_scroll.config(command=self.listbox.yview)
        self.listbox.bind("<<ListboxSelect>>", self.on_list_select)

        # Right Column: Live Thumbnail Preview & Metadata
        right_col = tk.Frame(body, bg=self.card_bg, padx=12, pady=12)
        right_col.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        preview_header = tk.Frame(right_col, bg=self.card_bg)
        preview_header.pack(fill=tk.X, pady=(0, 8))

        lbl_preview_title = tk.Label(
            preview_header,
            text="实时画面预览 (Live Preview)",
            font=("Microsoft YaHei UI", 11, "bold"),
            bg=self.card_bg,
            fg=self.text_color,
        )
        lbl_preview_title.pack(side=tk.LEFT)

        self.status_badge = tk.Label(
            preview_header,
            text="空闲就绪",
            font=("Microsoft YaHei UI", 9, "bold"),
            bg="#334155",
            fg="#ffffff",
            padx=8,
            pady=2,
        )
        self.status_badge.pack(side=tk.RIGHT)

        # Canvas for preview thumbnail
        self.preview_canvas = tk.Canvas(
            right_col,
            bg="#0f172a",
            highlightthickness=1,
            highlightbackground=self.border_color,
        )
        self.preview_canvas.pack(fill=tk.BOTH, expand=True)

        # Bottom Info Bar
        self.info_bar = tk.Label(
            right_col,
            text="提示: 点击左侧按钮即可触发截屏，画面将在此处实时缩略展示。",
            font=("Microsoft YaHei UI", 9),
            bg=self.card_bg,
            fg="#94a3b8",
            anchor=tk.W,
            pady=6,
        )
        self.info_bar.pack(fill=tk.X)

    def on_monitor_changed(self, event=None) -> None:
        """Handle user changing monitor in combobox."""
        idx = self.mon_combo.current()
        if 0 <= idx < len(self.monitors):
            self.current_monitor_idx = idx
            m = self.monitors[idx]
            self.screen_w, self.screen_h = m.width, m.height
            self.res_badge.config(text=f"{self.screen_w} × {self.screen_h}")
            self.info_bar.config(text=f"已切换目标屏幕为: {m.label}")

    def refresh_monitors(self) -> None:
        """Rescan system monitors."""
        self.monitors = self.grabber.get_monitors()
        self.mon_combo["values"] = [m.label for m in self.monitors]
        if self.current_monitor_idx >= len(self.monitors):
            self.current_monitor_idx = 0
        self.mon_combo.current(self.current_monitor_idx)
        self.on_monitor_changed()

    def on_capture_click(self) -> None:
        """Single capture."""
        self._execute_capture_async()

    def on_delay_capture_click(self) -> None:
        """Delayed capture with countdown."""
        def _delay_task():
            self.btn_delay.config(state=tk.DISABLED)
            for c in range(3, 0, -1):
                self.info_bar.config(text=f"倒计时 {c} 秒，请切回游戏画面...")
                time.sleep(1.0)
            self._execute_capture_async()
            self.btn_delay.config(state=tk.NORMAL)

        threading.Thread(target=_delay_task, daemon=True).start()

    def toggle_continuous_capture(self) -> None:
        """Toggle continuous interval capture."""
        if not self.is_continuous_capturing:
            try:
                interval = float(self.interval_var.get().strip())
                if interval < 0.05:
                    raise ValueError("间隔不能小于 0.05 秒")
            except Exception as e:
                messagebox.showerror("参数错误", f"请输入有效的时间间隔(秒数): {e}")
                return

            self.is_continuous_capturing = True
            self.btn_continuous.config(
                text="⏹ 停止连续采集",
                bg=self.danger_color,
                activebackground="#dc2626",
            )
            self.status_badge.config(text=f"连续采集运行中 ({interval}s)", bg=self.success_color)

            def _loop():
                while self.is_continuous_capturing:
                    t_start = time.time()
                    self._do_capture_once()
                    elapsed = time.time() - t_start
                    sleep_time = max(0.01, interval - elapsed)
                    time.sleep(sleep_time)

            self._capture_thread = threading.Thread(target=_loop, daemon=True)
            self._capture_thread.start()
        else:
            self.is_continuous_capturing = False
            self.btn_continuous.config(
                text="▶ 开始定时连续采集",
                bg=self.success_color,
                activebackground="#16a34a",
            )
            self.status_badge.config(text="空闲就绪", bg="#334155")
            self.info_bar.config(text="已停止连续采集。")

    def _execute_capture_async(self) -> None:
        threading.Thread(target=self._do_capture_once, daemon=True).start()

    def _do_capture_once(self) -> None:
        """Perform one screen grab and update GUI."""
        try:
            m_idx = self.current_monitor_idx
            mon = self.monitors[m_idx]
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:19]
            filename = f"screen_mon{m_idx + 1}_{timestamp}.png"
            filepath = os.path.join(self.output_dir, filename)

            img_bgr = self.grabber.capture_screen(m_idx)
            cv2.imwrite(filepath, img_bgr)

            self._total_captured_count += 1
            file_size_kb = os.path.getsize(filepath) / 1024.0

            # Schedule UI update on main thread
            self.root.after(0, self._on_capture_success, filepath, filename, img_bgr, file_size_kb, mon)
        except Exception as e:
            self.root.after(0, lambda: self.info_bar.config(text=f"[错误] 截屏失败: {e}"))

    def _on_capture_success(
        self, filepath: str, filename: str, img_bgr: np.ndarray, file_size_kb: float, mon: MonitorInfo
    ) -> None:
        self.info_bar.config(
            text=f"已保存: {filename} | 尺寸: {mon.width}x{mon.height} | 大小: {file_size_kb:.1f} KB (共计: {self._total_captured_count})"
        )
        self.listbox.insert(0, filename)
        self._captured_files.insert(0, filepath)
        self._render_preview(img_bgr)

    def _render_preview(self, img_bgr: np.ndarray) -> None:
        """Scale and render BGR image onto preview canvas."""
        canvas_w = max(100, self.preview_canvas.winfo_width())
        canvas_h = max(100, self.preview_canvas.winfo_height())

        h, w = img_bgr.shape[:2]
        scale = min(canvas_w / w, canvas_h / h)
        nw, nh = max(1, int(w * scale)), max(1, int(h * scale))

        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(img_rgb, (nw, nh), interpolation=cv2.INTER_AREA)

        pil_img = Image.fromarray(resized)
        self._current_photo = ImageTk.PhotoImage(pil_img)

        self.preview_canvas.delete("all")
        cx = canvas_w // 2
        cy = canvas_h // 2
        self.preview_canvas.create_image(cx, cy, image=self._current_photo, anchor=tk.CENTER)

    def on_list_select(self, event=None) -> None:
        """Handle selection in history listbox."""
        sel = self.listbox.curselection()
        if not sel:
            return
        idx = sel[0]
        if idx < len(self._captured_files):
            fp = self._captured_files[idx]
            if os.path.exists(fp):
                img_bgr = cv2.imread(fp)
                if img_bgr is not None:
                    sz = os.path.getsize(fp) / 1024.0
                    h, w = img_bgr.shape[:2]
                    self.info_bar.config(text=f"浏览查看: {os.path.basename(fp)} | {w}x{h} | {sz:.1f} KB")
                    self._render_preview(img_bgr)

    def _refresh_history_list(self) -> None:
        """Populate listbox from existing files in output_dir."""
        self.listbox.delete(0, tk.END)
        self._captured_files.clear()
        if not os.path.exists(self.output_dir):
            return
        files = sorted(
            [f for f in os.listdir(self.output_dir) if f.lower().endswith((".png", ".jpg", ".bmp"))],
            key=lambda x: os.path.getmtime(os.path.join(self.output_dir, x)),
            reverse=True,
        )
        for f in files:
            self.listbox.insert(tk.END, f)
            self._captured_files.append(os.path.join(self.output_dir, f))

    def on_open_folder(self) -> None:
        """Open output directory in Windows Explorer."""
        os.startfile(self.output_dir)


def main():
    root = tk.Tk()
    app = ScreenCaptureGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
