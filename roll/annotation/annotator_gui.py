# [UPDATE - 2026-09-12]
# Reason: User specified that only a single target object needs to be annotated ("我们只需要标注一个物体").
# Modification: Simplified GUI specifically for single-class (Class 0: Target/目标) high-speed labeling.
#               Removed multi-class palette, enabled instant drag-to-box, space/D auto-next,
#               and 1-click standard YOLO dataset export with nc=1.

from __future__ import annotations

import datetime
import json
import os
import shutil
import sys
import tkinter as tk
from tkinter import ttk, messagebox
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image, ImageTk

# UTF-8 console support
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
ROLL_ROOT = os.path.dirname(CURRENT_DIR)
PROJECT_ROOT = os.path.dirname(ROLL_ROOT)

if ROLL_ROOT not in sys.path:
    sys.path.insert(0, ROLL_ROOT)

TARGET_CLASS_NAME = "target"
TARGET_COLOR = "#00ff66"       # High-visibility neon green
TARGET_COLOR_FILL = "#00ff66"


class BoundingBox:
    """Single target bounding box representation."""

    def __init__(self, x1: float, y1: float, x2: float, y2: float) -> None:
        self.class_id = 0
        self.x1 = min(x1, x2)
        self.y1 = min(y1, y2)
        self.x2 = max(x1, x2)
        self.y2 = max(y1, y2)

    def to_yolo_norm(self, img_w: int, img_h: int) -> Tuple[int, float, float, float, float]:
        bw = max(1.0, self.x2 - self.x1)
        bh = max(1.0, self.y2 - self.y1)
        xc = self.x1 + bw / 2.0
        yc = self.y1 + bh / 2.0
        return (
            0,
            min(1.0, max(0.0, xc / img_w)),
            min(1.0, max(0.0, yc / img_h)),
            min(1.0, max(0.0, bw / img_w)),
            min(1.0, max(0.0, bh / img_h)),
        )

    @classmethod
    def from_yolo_norm(cls, xc: float, yc: float, w: float, h: float, img_w: int, img_h: int) -> BoundingBox:
        pw = w * img_w
        ph = h * img_h
        pxc = xc * img_w
        pyc = yc * img_h
        return cls(pxc - pw / 2.0, pyc - ph / 2.0, pxc + pw / 2.0, pyc + ph / 2.0)


class SingleClassAnnotatorGUI:
    """High-speed single-class YOLO annotation GUI."""

    def __init__(
        self,
        root: tk.Tk,
        raw_images_dir: Optional[str] = None,
        annotations_dir: Optional[str] = None,
    ) -> None:
        self.root = root
        self.root.title("RocoClicker - 单目标快速标注工具 (Single-Object Annotator)")
        self.root.geometry("1380x880")
        self.root.minsize(1080, 700)

        if raw_images_dir is None:
            raw_images_dir = os.path.join(ROLL_ROOT, "data", "raw_captures")
        if annotations_dir is None:
            annotations_dir = os.path.join(ROLL_ROOT, "data", "annotations")

        self.raw_images_dir = os.path.abspath(raw_images_dir)
        self.annotations_dir = os.path.abspath(annotations_dir)
        os.makedirs(self.raw_images_dir, exist_ok=True)
        os.makedirs(self.annotations_dir, exist_ok=True)

        # Image list
        self.image_files: List[str] = []
        self.filtered_files: List[str] = []
        self.current_img_idx = 0
        self.current_image_path: Optional[str] = None
        self.current_bgr: Optional[np.ndarray] = None
        self.current_photo: Optional[ImageTk.PhotoImage] = None

        # Dimensions & coordinates
        self.img_w = 0
        self.img_h = 0
        self.scale = 1.0
        self.offset_x = 0
        self.offset_y = 0

        # Boxes
        self.boxes: List[BoundingBox] = []
        self.selected_box_idx: Optional[int] = None
        self.undo_stack: List[List[BoundingBox]] = []

        # Interaction
        self.is_drawing = False
        self.drag_start_canvas: Optional[Tuple[float, float]] = None
        self.drag_curr_canvas: Optional[Tuple[float, float]] = None
        self.crosshair_pos: Optional[Tuple[float, float]] = None

        # Filter mode
        self.filter_mode = "all"  # "all", "unannotated", "annotated"

        self._apply_styles()
        self._build_ui()
        self._bind_events()
        self._load_image_list()

    def _apply_styles(self) -> None:
        self.bg_color = "#0b0f19"
        self.card_bg = "#161f30"
        self.text_color = "#f8fafc"
        self.accent_color = "#2563eb"
        self.border_color = "#334155"

        self.root.configure(bg=self.bg_color)
        self.style = ttk.Style(self.root)
        try:
            self.style.theme_use("clam")
        except Exception:
            pass
        self.style.configure(".", background=self.bg_color, foreground=self.text_color, font=("Microsoft YaHei UI", 9))

    def _build_ui(self) -> None:
        # Top Header
        header = tk.Frame(self.root, bg=self.card_bg, height=58, padx=16, pady=8)
        header.pack(fill=tk.X, side=tk.TOP)

        title_box = tk.Frame(header, bg=self.card_bg)
        title_box.pack(side=tk.LEFT)

        tk.Label(
            title_box,
            text="🎯 单目标急速标注器",
            font=("Microsoft YaHei UI", 13, "bold"),
            bg=self.card_bg,
            fg=self.text_color,
        ).pack(side=tk.LEFT)

        tk.Label(
            title_box,
            text="[类别: target]",
            font=("Consolas", 10, "bold"),
            bg="#0f766e",
            fg="#ffffff",
            padx=8,
            pady=2,
        ).pack(side=tk.LEFT, padx=(12, 0))

        self.progress_badge = tk.Label(
            header,
            text="已标注: 0 / 0 (0.0%)",
            font=("Consolas", 10, "bold"),
            bg="#0284c7",
            fg="#ffffff",
            padx=12,
            pady=4,
        )
        self.progress_badge.pack(side=tk.LEFT, padx=(20, 10))

        # Filter
        tk.Label(header, text="筛选:", font=("Microsoft YaHei UI", 9), bg=self.card_bg, fg=self.text_color).pack(
            side=tk.LEFT, padx=(12, 4)
        )
        self.filter_combo = ttk.Combobox(
            header,
            values=["全部图像 (All)", "仅未标注 (Unannotated)", "仅已标注 (Annotated)"],
            state="readonly",
            width=20,
            font=("Microsoft YaHei UI", 9),
        )
        self.filter_combo.current(0)
        self.filter_combo.pack(side=tk.LEFT, padx=(0, 15))
        self.filter_combo.bind("<<ComboboxSelected>>", self._on_filter_changed)

        btn_export = tk.Button(
            header,
            text="📦 一键导出 YOLO 训练数据集",
            font=("Microsoft YaHei UI", 9, "bold"),
            bg="#16a34a",
            fg="#ffffff",
            activebackground="#15803d",
            relief=tk.FLAT,
            padx=12,
            pady=4,
            cursor="hand2",
            command=self.export_yolo_dataset,
        )
        btn_export.pack(side=tk.RIGHT, padx=(8, 0))

        btn_open = tk.Button(
            header,
            text="📁 打开标注文件夹",
            font=("Microsoft YaHei UI", 9),
            bg="#334155",
            fg="#ffffff",
            relief=tk.FLAT,
            padx=8,
            pady=4,
            cursor="hand2",
            command=lambda: os.startfile(self.annotations_dir),
        )
        btn_open.pack(side=tk.RIGHT)

        # Main Body
        body = tk.Frame(self.root, bg=self.bg_color, padx=10, pady=10)
        body.pack(fill=tk.BOTH, expand=True)

        # Left Column: Image List & Quick Controls
        left_col = tk.Frame(body, bg=self.bg_color, width=280)
        left_col.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))
        left_col.pack_propagate(False)

        # Quick Tips Frame
        tips_frame = tk.LabelFrame(
            left_col,
            text=" 急速标注快捷键 ",
            bg=self.card_bg,
            fg=self.text_color,
            font=("Microsoft YaHei UI", 9, "bold"),
            padx=8,
            pady=8,
        )
        tips_frame.pack(fill=tk.X, pady=(0, 8))

        tips_text = (
            "• 左键拖拽: 框选目标物体\n"
            "• D / 空格键: 保存并跳到下一张\n"
            "• A: 跳到上一张\n"
            "• Delete / 右键: 删除选中框\n"
            "• Ctrl+Z: 撤销操作\n"
            "• Ctrl+S: 手动保存"
        )
        tk.Label(
            tips_frame,
            text=tips_text,
            font=("Microsoft YaHei UI", 8),
            bg=self.card_bg,
            fg="#94a3b8",
            justify=tk.LEFT,
        ).pack(anchor=tk.W)

        # Image List Frame
        list_frame = tk.LabelFrame(
            left_col,
            text=" 图像列表 (点击跳转) ",
            bg=self.card_bg,
            fg=self.text_color,
            font=("Microsoft YaHei UI", 9, "bold"),
            padx=6,
            pady=6,
        )
        list_frame.pack(fill=tk.BOTH, expand=True)

        list_scroll = tk.Scrollbar(list_frame)
        list_scroll.pack(side=tk.RIGHT, fill=tk.Y)

        self.file_listbox = tk.Listbox(
            list_frame,
            bg="#020617",
            fg="#f8fafc",
            selectbackground=self.accent_color,
            selectforeground="#ffffff",
            font=("Consolas", 8),
            relief=tk.FLAT,
            yscrollcommand=list_scroll.set,
            activestyle="none",
        )
        self.file_listbox.pack(fill=tk.BOTH, expand=True)
        list_scroll.config(command=self.file_listbox.yview)
        self.file_listbox.bind("<<ListboxSelect>>", self._on_listbox_select)

        # Navigation Bar
        nav_frame = tk.Frame(left_col, bg=self.bg_color)
        nav_frame.pack(fill=tk.X, pady=(8, 0))

        self.btn_prev = tk.Button(
            nav_frame,
            text="◀ 上一张 (A)",
            font=("Microsoft YaHei UI", 9, "bold"),
            bg="#334155",
            fg="#ffffff",
            relief=tk.FLAT,
            pady=6,
            cursor="hand2",
            command=self.prev_image,
        )
        self.btn_prev.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 4))

        self.btn_next = tk.Button(
            nav_frame,
            text="下一张 (D / 空格) ▶",
            font=("Microsoft YaHei UI", 9, "bold"),
            bg="#2563eb",
            fg="#ffffff",
            relief=tk.FLAT,
            pady=6,
            cursor="hand2",
            command=self.next_image,
        )
        self.btn_next.pack(side=tk.RIGHT, fill=tk.X, expand=True, padx=(4, 0))

        # Right Column: Box info & actions
        right_col = tk.Frame(body, bg=self.bg_color, width=220)
        right_col.pack(side=tk.RIGHT, fill=tk.Y, padx=(10, 0))
        right_col.pack_propagate(False)

        action_frame = tk.LabelFrame(
            right_col,
            text=" 标注控制 ",
            bg=self.card_bg,
            fg=self.text_color,
            font=("Microsoft YaHei UI", 9, "bold"),
            padx=8,
            pady=8,
        )
        action_frame.pack(fill=tk.X, pady=(0, 10))

        tk.Button(
            action_frame,
            text="💾 保存当前帧 (Ctrl+S)",
            font=("Microsoft YaHei UI", 9, "bold"),
            bg="#059669",
            fg="#ffffff",
            relief=tk.FLAT,
            pady=6,
            cursor="hand2",
            command=self.save_current_annotations,
        ).pack(fill=tk.X, pady=(0, 6))

        tk.Button(
            action_frame,
            text="↩ 撤销 (Ctrl+Z)",
            font=("Microsoft YaHei UI", 8),
            bg="#334155",
            fg="#ffffff",
            relief=tk.FLAT,
            pady=4,
            cursor="hand2",
            command=self.undo,
        ).pack(fill=tk.X, pady=(0, 4))

        tk.Button(
            action_frame,
            text="🗑 删除当前框 (Del)",
            font=("Microsoft YaHei UI", 8),
            bg="#dc2626",
            fg="#ffffff",
            relief=tk.FLAT,
            pady=4,
            cursor="hand2",
            command=self.delete_selected_box,
        ).pack(fill=tk.X, pady=(0, 4))

        tk.Button(
            action_frame,
            text="🧹 清空本帧所有框",
            font=("Microsoft YaHei UI", 8),
            bg="#475569",
            fg="#ffffff",
            relief=tk.FLAT,
            pady=4,
            cursor="hand2",
            command=self.clear_current_boxes,
        ).pack(fill=tk.X)

        # Target Objects in Current Frame
        cur_box_frame = tk.LabelFrame(
            right_col,
            text=" 本帧已标目标 ",
            bg=self.card_bg,
            fg=self.text_color,
            font=("Microsoft YaHei UI", 9, "bold"),
            padx=6,
            pady=6,
        )
        cur_box_frame.pack(fill=tk.BOTH, expand=True)

        box_scroll = tk.Scrollbar(cur_box_frame)
        box_scroll.pack(side=tk.RIGHT, fill=tk.Y)

        self.box_listbox = tk.Listbox(
            cur_box_frame,
            bg="#020617",
            fg="#f8fafc",
            selectbackground=self.accent_color,
            font=("Consolas", 9),
            relief=tk.FLAT,
            yscrollcommand=box_scroll.set,
            activestyle="none",
        )
        self.box_listbox.pack(fill=tk.BOTH, expand=True)
        box_scroll.config(command=self.box_listbox.yview)
        self.box_listbox.bind("<<ListboxSelect>>", self._on_box_listbox_select)

        # Center Column: Interactive Canvas
        center_col = tk.Frame(body, bg=self.card_bg)
        center_col.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.canvas = tk.Canvas(
            center_col,
            bg="#020617",
            cursor="crosshair",
            highlightthickness=1,
            highlightbackground=self.border_color,
        )
        self.canvas.pack(fill=tk.BOTH, expand=True)

        # Bottom Info Bar
        self.info_bar = tk.Label(
            self.root,
            text="极简操作: 鼠标直接拖拽拉框即可完成标注 -> 按 空格 或 D 键自动保存并切入下一张！",
            font=("Microsoft YaHei UI", 9),
            bg=self.card_bg,
            fg="#94a3b8",
            anchor=tk.W,
            padx=16,
            pady=6,
        )
        self.info_bar.pack(fill=tk.X, side=tk.BOTTOM)

    def _bind_events(self) -> None:
        self.canvas.bind("<ButtonPress-1>", self._on_canvas_press)
        self.canvas.bind("<B1-Motion>", self._on_canvas_drag)
        self.canvas.bind("<ButtonRelease-1>", self._on_canvas_release)
        self.canvas.bind("<Motion>", self._on_canvas_hover)
        self.canvas.bind("<Button-3>", self._on_canvas_right_click)
        self.canvas.bind("<Configure>", lambda e: self._render_canvas())

        # Shortcuts
        self.root.bind("<Key-a>", lambda e: self.prev_image())
        self.root.bind("<Key-A>", lambda e: self.prev_image())
        self.root.bind("<Left>", lambda e: self.prev_image())

        self.root.bind("<Key-d>", lambda e: self.next_image())
        self.root.bind("<Key-D>", lambda e: self.next_image())
        self.root.bind("<Right>", lambda e: self.next_image())
        self.root.bind("<space>", lambda e: self.next_image())

        self.root.bind("<Control-s>", lambda e: self.save_current_annotations())
        self.root.bind("<Control-S>", lambda e: self.save_current_annotations())
        self.root.bind("<Control-z>", lambda e: self.undo())
        self.root.bind("<Control-Z>", lambda e: self.undo())

        self.root.bind("<Delete>", lambda e: self.delete_selected_box())
        self.root.bind("<BackSpace>", lambda e: self.delete_selected_box())

    def _push_undo(self) -> None:
        snapshot = [BoundingBox(b.x1, b.y1, b.x2, b.y2) for b in self.boxes]
        self.undo_stack.append(snapshot)
        if len(self.undo_stack) > 30:
            self.undo_stack.pop(0)

    def undo(self) -> None:
        if not self.undo_stack:
            return
        self.boxes = self.undo_stack.pop()
        self.selected_box_idx = None
        self._update_box_listbox()
        self._render_canvas()
        self.info_bar.config(text="已撤销上一步操作。")

    def _load_image_list(self) -> None:
        if not os.path.exists(self.raw_images_dir):
            return

        valid_exts = {".png", ".jpg", ".jpeg", ".bmp"}
        self.image_files = sorted(
            [f for f in os.listdir(self.raw_images_dir) if os.path.splitext(f.lower())[1] in valid_exts]
        )
        self._apply_filter()

    def _on_filter_changed(self, event=None) -> None:
        modes = ["all", "unannotated", "annotated"]
        self.filter_mode = modes[self.filter_combo.current()]
        self._apply_filter()

    def _apply_filter(self) -> None:
        annotated_count = 0
        filtered = []

        for f in self.image_files:
            stem = os.path.splitext(f)[0]
            txt_path = os.path.join(self.annotations_dir, f"{stem}.txt")
            has_ann = os.path.exists(txt_path) and os.path.getsize(txt_path) > 0
            if has_ann:
                annotated_count += 1

            if self.filter_mode == "all":
                filtered.append(f)
            elif self.filter_mode == "unannotated" and not has_ann:
                filtered.append(f)
            elif self.filter_mode == "annotated" and has_ann:
                filtered.append(f)

        self.filtered_files = filtered
        total_cnt = len(self.image_files)
        pct = (annotated_count / total_cnt * 100.0) if total_cnt > 0 else 0.0
        self.progress_badge.config(text=f"已标注: {annotated_count} / {total_cnt} ({pct:.1f}%)")

        self.file_listbox.delete(0, tk.END)
        for f in self.filtered_files:
            stem = os.path.splitext(f)[0]
            txt_path = os.path.join(self.annotations_dir, f"{stem}.txt")
            tag = "[✓]" if os.path.exists(txt_path) and os.path.getsize(txt_path) > 0 else "[ ]"
            self.file_listbox.insert(tk.END, f"{tag} {f}")

        if self.filtered_files:
            if self.current_img_idx >= len(self.filtered_files):
                self.current_img_idx = 0
            self.file_listbox.selection_clear(0, tk.END)
            self.file_listbox.selection_set(self.current_img_idx)
            self.file_listbox.see(self.current_img_idx)
            self._load_current_image()
        else:
            self.canvas.delete("all")
            self.boxes.clear()
            self._update_box_listbox()
            self.info_bar.config(text="当前筛选条件下没有图片。")

    def _load_current_image(self) -> None:
        if not self.filtered_files or self.current_img_idx >= len(self.filtered_files):
            return

        filename = self.filtered_files[self.current_img_idx]
        self.current_image_path = os.path.join(self.raw_images_dir, filename)

        self.current_bgr = cv2.imread(self.current_image_path)
        if self.current_bgr is None:
            self.info_bar.config(text=f"[错误] 读取图片失败: {filename}")
            return

        self.img_h, self.img_w = self.current_bgr.shape[:2]
        self.undo_stack.clear()
        self.selected_box_idx = None
        self._load_current_annotations()
        self._render_canvas()

        self.info_bar.config(
            text=f"[{self.current_img_idx + 1}/{len(self.filtered_files)}] 正在标注: {filename} ({self.img_w}x{self.img_h}) | 当前已标目标: {len(self.boxes)} 个"
        )

    def _load_current_annotations(self) -> None:
        self.boxes.clear()
        if not self.current_image_path:
            return

        stem = os.path.splitext(os.path.basename(self.current_image_path))[0]
        txt_path = os.path.join(self.annotations_dir, f"{stem}.txt")

        if os.path.exists(txt_path):
            with open(txt_path, "r", encoding="utf-8") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        xc, yc, w, h = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
                        self.boxes.append(BoundingBox.from_yolo_norm(xc, yc, w, h, self.img_w, self.img_h))
        self._update_box_listbox()

    def save_current_annotations(self) -> None:
        if not self.current_image_path:
            return

        stem = os.path.splitext(os.path.basename(self.current_image_path))[0]
        txt_path = os.path.join(self.annotations_dir, f"{stem}.txt")

        if not self.boxes:
            if os.path.exists(txt_path):
                os.remove(txt_path)
            self._update_listbox_item_tag(False)
            return

        with open(txt_path, "w", encoding="utf-8") as f:
            for b in self.boxes:
                _, xc, yc, w, h = b.to_yolo_norm(self.img_w, self.img_h)
                f.write(f"0 {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}\n")

        self._update_listbox_item_tag(True)
        self.info_bar.config(text=f"[已保存] 目标框已写入: {os.path.basename(txt_path)} (共 {len(self.boxes)} 处目标)")

    def _update_listbox_item_tag(self, is_annotated: bool) -> None:
        idx = self.current_img_idx
        if idx < len(self.filtered_files):
            fname = self.filtered_files[idx]
            tag = "[✓]" if is_annotated else "[ ]"
            self.file_listbox.delete(idx)
            self.file_listbox.insert(idx, f"{tag} {fname}")
            self.file_listbox.selection_set(idx)

    def prev_image(self) -> None:
        self.save_current_annotations()
        if self.current_img_idx > 0:
            self.current_img_idx -= 1
            self.file_listbox.selection_clear(0, tk.END)
            self.file_listbox.selection_set(self.current_img_idx)
            self.file_listbox.see(self.current_img_idx)
            self._load_current_image()

    def next_image(self) -> None:
        self.save_current_annotations()
        if self.current_img_idx < len(self.filtered_files) - 1:
            self.current_img_idx += 1
            self.file_listbox.selection_clear(0, tk.END)
            self.file_listbox.selection_set(self.current_img_idx)
            self.file_listbox.see(self.current_img_idx)
            self._load_current_image()
        else:
            self.info_bar.config(text="🎉 已到达当前列表的最后一张图片！")

    def _on_listbox_select(self, event=None) -> None:
        sel = self.file_listbox.curselection()
        if not sel:
            return
        idx = sel[0]
        if idx != self.current_img_idx:
            self.save_current_annotations()
            self.current_img_idx = idx
            self._load_current_image()

    def img_to_canvas(self, ix: float, iy: float) -> Tuple[float, float]:
        cx = self.offset_x + ix * self.scale
        cy = self.offset_y + iy * self.scale
        return cx, cy

    def canvas_to_img(self, cx: float, cy: float) -> Tuple[float, float]:
        ix = (cx - self.offset_x) / self.scale
        iy = (cy - self.offset_y) / self.scale
        return max(0.0, min(float(self.img_w), ix)), max(0.0, min(float(self.img_h), iy))

    def _render_canvas(self) -> None:
        if self.current_bgr is None:
            return

        cw = self.canvas.winfo_width()
        ch = self.canvas.winfo_height()
        if cw <= 10 or ch <= 10:
            return

        scale_x = cw / self.img_w
        scale_y = ch / self.img_h
        self.scale = min(scale_x, scale_y) * 0.98

        nw = max(1, int(self.img_w * self.scale))
        nh = max(1, int(self.img_h * self.scale))

        self.offset_x = (cw - nw) // 2
        self.offset_y = (ch - nh) // 2

        resized = cv2.resize(self.current_bgr, (nw, nh), interpolation=cv2.INTER_AREA)
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb)
        self.current_photo = ImageTk.PhotoImage(pil_img)

        self.canvas.delete("all")
        self.canvas.create_image(self.offset_x, self.offset_y, image=self.current_photo, anchor=tk.NW)

        # Draw existing boxes
        for idx, box in enumerate(self.boxes):
            x1, y1 = self.img_to_canvas(box.x1, box.y1)
            x2, y2 = self.img_to_canvas(box.x2, box.y2)

            is_sel = idx == self.selected_box_idx
            color = "#ffffff" if is_sel else TARGET_COLOR

            # Box
            self.canvas.create_rectangle(x1, y1, x2, y2, outline=color, width=3 if is_sel else 2)

            # Badge
            badge_text = f"目标 #{idx + 1}"
            bw = len(badge_text) * 11 + 8
            self.canvas.create_rectangle(x1, max(0, y1 - 18), x1 + bw, y1, fill="#0f766e" if not is_sel else "#2563eb", outline="")
            self.canvas.create_text(
                x1 + 4,
                max(0, y1 - 18) + 9,
                text=badge_text,
                fill="#ffffff",
                anchor=tk.W,
                font=("Microsoft YaHei UI", 8, "bold"),
            )

        # Dynamic drag box
        if self.is_drawing and self.drag_start_canvas and self.drag_curr_canvas:
            sx, sy = self.drag_start_canvas
            ex, ey = self.drag_curr_canvas
            self.canvas.create_rectangle(sx, sy, ex, ey, outline=TARGET_COLOR, width=2, dash=(4, 2))

        # Crosshairs
        if self.crosshair_pos and not self.is_drawing:
            mx, my = self.crosshair_pos
            self.canvas.create_line(0, my, cw, my, fill="#64748b", dash=(2, 2))
            self.canvas.create_line(mx, 0, mx, ch, fill="#64748b", dash=(2, 2))

    def _on_canvas_press(self, event) -> None:
        self.is_drawing = True
        self.drag_start_canvas = (event.x, event.y)
        self.drag_curr_canvas = (event.x, event.y)

    def _on_canvas_drag(self, event) -> None:
        if self.is_drawing:
            self.drag_curr_canvas = (event.x, event.y)
            self._render_canvas()

    def _on_canvas_release(self, event) -> None:
        if not self.is_drawing or not self.drag_start_canvas:
            self.is_drawing = False
            return

        sx, sy = self.drag_start_canvas
        ex, ey = event.x, event.y
        self.is_drawing = False
        self.drag_start_canvas = None
        self.drag_curr_canvas = None

        # If pure click, select existing box
        if abs(ex - sx) < 8 and abs(ey - sy) < 8:
            clicked_box = self._find_box_under_cursor(ex, ey)
            self.selected_box_idx = clicked_box
            self._update_box_listbox()
            self._render_canvas()
            return

        ix1, iy1 = self.canvas_to_img(sx, sy)
        ix2, iy2 = self.canvas_to_img(ex, ey)

        if abs(ix2 - ix1) > 4 and abs(iy2 - iy1) > 4:
            self._push_undo()
            new_box = BoundingBox(ix1, iy1, ix2, iy2)
            self.boxes.append(new_box)
            self.selected_box_idx = len(self.boxes) - 1
            self._update_box_listbox()
            self._render_canvas()
            self.info_bar.config(text="目标框已拉取。按 空格 或 D 键直接保存并切到下一张！")

    def _on_canvas_hover(self, event) -> None:
        self.crosshair_pos = (event.x, event.y)
        if not self.is_drawing:
            self._render_canvas()

    def _on_canvas_right_click(self, event) -> None:
        box_idx = self._find_box_under_cursor(event.x, event.y)
        if box_idx is not None:
            self._push_undo()
            del self.boxes[box_idx]
            self.selected_box_idx = None
            self._update_box_listbox()
            self._render_canvas()
            self.info_bar.config(text=f"已删除目标框 #{box_idx + 1}")

    def _find_box_under_cursor(self, cx: float, cy: float) -> Optional[int]:
        ix, iy = self.canvas_to_img(cx, cy)
        for i in range(len(self.boxes) - 1, -1, -1):
            b = self.boxes[i]
            if b.x1 <= ix <= b.x2 and b.y1 <= iy <= b.y2:
                return i
        return None

    def _update_box_listbox(self) -> None:
        self.box_listbox.delete(0, tk.END)
        for idx, b in enumerate(self.boxes):
            w = int(b.x2 - b.x1)
            h = int(b.y2 - b.y1)
            self.box_listbox.insert(tk.END, f"目标 #{idx + 1} [{w}x{h}]")

        if self.selected_box_idx is not None and self.selected_box_idx < len(self.boxes):
            self.box_listbox.selection_set(self.selected_box_idx)

    def _on_box_listbox_select(self, event=None) -> None:
        sel = self.box_listbox.curselection()
        if sel:
            self.selected_box_idx = sel[0]
            self._render_canvas()

    def delete_selected_box(self) -> None:
        if self.selected_box_idx is not None and self.selected_box_idx < len(self.boxes):
            self._push_undo()
            del self.boxes[self.selected_box_idx]
            self.selected_box_idx = None
            self._update_box_listbox()
            self._render_canvas()
            self.info_bar.config(text="已删除选中目标框。")

    def clear_current_boxes(self) -> None:
        if self.boxes:
            self._push_undo()
            self.boxes.clear()
            self.selected_box_idx = None
            self._update_box_listbox()
            self._render_canvas()
            self.info_bar.config(text="已清空当前帧目标框。")

    def export_yolo_dataset(self) -> None:
        """Export all labeled images & labels to standard YOLO train/val directory structure."""
        labeled_stems = []
        for f in self.image_files:
            stem = os.path.splitext(f)[0]
            txt = os.path.join(self.annotations_dir, f"{stem}.txt")
            if os.path.exists(txt) and os.path.getsize(txt) > 0:
                labeled_stems.append(stem)

        if not labeled_stems:
            messagebox.showwarning("导出失败", "当前没有发现任何已标注的目标！请先在图片上拖拽拉框。")
            return

        export_dir = os.path.join(ROLL_ROOT, "data", "yolo_dataset")
        os.makedirs(export_dir, exist_ok=True)

        train_img_dir = os.path.join(export_dir, "images", "train")
        val_img_dir = os.path.join(export_dir, "images", "val")
        train_lbl_dir = os.path.join(export_dir, "labels", "train")
        val_lbl_dir = os.path.join(export_dir, "labels", "val")

        for d in [train_img_dir, val_img_dir, train_lbl_dir, val_lbl_dir]:
            os.makedirs(d, exist_ok=True)

        import random
        random.seed(42)
        shuffled = list(labeled_stems)
        random.shuffle(shuffled)
        split_idx = max(1, int(len(shuffled) * 0.8))
        train_set = set(shuffled[:split_idx])

        for stem in labeled_stems:
            orig_img = next(
                (os.path.join(self.raw_images_dir, f) for f in self.image_files if os.path.splitext(f)[0] == stem), None
            )
            orig_txt = os.path.join(self.annotations_dir, f"{stem}.txt")

            if orig_img and os.path.exists(orig_img):
                target_img_dir = train_img_dir if stem in train_set else val_img_dir
                target_lbl_dir = train_lbl_dir if stem in train_set else val_lbl_dir

                shutil.copy2(orig_img, os.path.join(target_img_dir, os.path.basename(orig_img)))
                shutil.copy2(orig_txt, os.path.join(target_lbl_dir, f"{stem}.txt"))

        # Generate dataset.yaml for YOLOv8m single-class training
        yaml_path = os.path.join(export_dir, "dataset.yaml")
        with open(yaml_path, "w", encoding="utf-8") as f:
            f.write(f"path: {export_dir.replace(os.sep, '/')}\n")
            f.write("train: images/train\n")
            f.write("val: images/val\n")
            f.write("nc: 1\n")
            f.write(f"names:\n  0: {TARGET_CLASS_NAME}\n")

        messagebox.showinfo(
            "导出成功",
            f"单类别 YOLO 训练集已顺利生成！\n\n"
            f"已标注样本总量: {len(labeled_stems)} 张\n"
            f"训练集 (Train): {len(train_set)} 张\n"
            f"验证集 (Val): {len(labeled_stems) - len(train_set)} 张\n"
            f"目标类别数量: nc = 1 ({TARGET_CLASS_NAME})\n\n"
            f"保存路径: {export_dir}\n"
            f"配置文件: {yaml_path}",
        )
        os.startfile(export_dir)


def main():
    root = tk.Tk()
    app = SingleClassAnnotatorGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
