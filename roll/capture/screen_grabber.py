# [NEW - 2026-09-12]
# Reason: ScreenGrabber implementation for roll module supporting multi-monitor enumeration and high-speed GDI capture.
# Content: Enumerate monitors, capture target monitor into NumPy array or save as image file.

from __future__ import annotations

import ctypes
from ctypes import wintypes
import os
import struct
from typing import Optional, Tuple, Any, List

import numpy as np

from .win32_defs import (
    RECT, MONITORINFOEX, BITMAPINFOHEADER, MonitorInfo,
    SM_CXSCREEN, SM_CYSCREEN, SRCCOPY, DIB_RGB_COLORS, BI_RGB
)


class ScreenGrabber:
    """Multi-monitor high-speed screen capture engine using native Windows GDI."""

    def __init__(self) -> None:
        self._u32 = ctypes.windll.user32
        self._g32 = ctypes.windll.gdi32
        try:
            self._u32.SetProcessDPIAware()
        except Exception:
            pass
        self.cached_monitors: List[MonitorInfo] = []
        self.get_monitors()

    def get_monitors(self) -> List[MonitorInfo]:
        """Enumerate all connected physical/logical monitors."""
        monitors: List[MonitorInfo] = []
        CMPPROC = ctypes.WINFUNCTYPE(
            wintypes.BOOL, wintypes.HMONITOR, wintypes.HDC, ctypes.POINTER(RECT), wintypes.LPARAM
        )

        def _enum_proc(hmon, hdc, lprc, lparam):
            mi = MONITORINFOEX()
            mi.cbSize = ctypes.sizeof(MONITORINFOEX)
            if self._u32.GetMonitorInfoW(hmon, ctypes.byref(mi)):
                dev = mi.szDevice
                l, t, r, b = int(mi.rcMonitor.left), int(mi.rcMonitor.top), int(mi.rcMonitor.right), int(mi.rcMonitor.bottom)
                w, h = max(1, r - l), max(1, b - t)
                is_pri = bool(mi.dwFlags & 1)
                idx = len(monitors)
                monitors.append(
                    MonitorInfo(
                        index=idx,
                        device_name=dev,
                        left=l,
                        top=t,
                        right=r,
                        bottom=b,
                        width=w,
                        height=h,
                        is_primary=is_pri,
                        label=f"屏幕 {idx + 1}{' (主屏幕)' if is_pri else ''} - {w}x{h} [{dev}]"
                    )
                )
            return True

        self._u32.EnumDisplayMonitors(0, 0, CMPPROC(_enum_proc), 0)
        if not monitors:
            w, h = int(self._u32.GetSystemMetrics(SM_CXSCREEN)), int(self._u32.GetSystemMetrics(SM_CYSCREEN))
            monitors.append(MonitorInfo(0, "DISPLAY", 0, 0, w, h, w, h, True, f"屏幕 1 (主屏幕) - {w}x{h}"))
        self.cached_monitors = monitors
        return monitors

    def get_primary_resolution(self) -> Tuple[int, int]:
        """Return (width, height) of the primary monitor."""
        for m in self.cached_monitors:
            if m.is_primary:
                return m.width, m.height
        return (self.cached_monitors[0].width, self.cached_monitors[0].height) if self.cached_monitors else (1920, 1080)

    def capture_screen(self, monitor_index: int = 0) -> np.ndarray:
        """Capture screen of specific monitor by index as BGR uint8 NumPy array."""
        if not self.cached_monitors or monitor_index >= len(self.cached_monitors):
            self.get_monitors()
        idx = monitor_index if (0 <= monitor_index < len(self.cached_monitors)) else 0
        mon = self.cached_monitors[idx]
        return self._capture_display_dc(mon.device_name, mon.width, mon.height)

    def capture_primary_screen(self) -> np.ndarray:
        """Capture primary monitor screen as BGR uint8 NumPy array."""
        for i, m in enumerate(self.cached_monitors):
            if m.is_primary:
                return self.capture_screen(i)
        return self.capture_screen(0)

    def _capture_display_dc(self, dev: str, w: int, h: int) -> np.ndarray:
        hdc = self._g32.CreateDCW(dev, None, None, None) or self._u32.GetDC(0)
        hdc_mem = self._g32.CreateCompatibleDC(hdc)
        hbm = self._g32.CreateCompatibleBitmap(hdc, w, h)
        hbm_old = self._g32.SelectObject(hdc_mem, hbm)

        self._g32.BitBlt(hdc_mem, 0, 0, w, h, hdc, 0, 0, SRCCOPY)

        bmi = BITMAPINFOHEADER()
        bmi.biSize = ctypes.sizeof(BITMAPINFOHEADER)
        bmi.biWidth, bmi.biHeight = w, -h
        bmi.biPlanes, bmi.biBitCount, bmi.biCompression = 1, 32, BI_RGB
        bmi.biSizeImage = w * h * 4

        buf = (ctypes.c_char * (w * h * 4))()
        self._g32.GetDIBits(
            hdc_mem, hbm, 0, h, ctypes.byref(buf), ctypes.cast(ctypes.byref(bmi), ctypes.c_void_p), DIB_RGB_COLORS
        )

        self._g32.SelectObject(hdc_mem, hbm_old)
        self._g32.DeleteObject(hbm)
        self._g32.DeleteDC(hdc_mem)
        self._g32.DeleteDC(hdc)

        raw_bytes = bytes(buf)
        img = np.frombuffer(raw_bytes, dtype=np.uint8).reshape((h, w, 4))[:, :, :3]
        return np.ascontiguousarray(img)

    def save_screenshot(self, filepath: str, monitor_index: int = 0) -> str:
        """Capture screen and save directly to disk (PNG or JPG)."""
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        img = self.capture_screen(monitor_index)
        import cv2
        cv2.imwrite(filepath, img)
        return filepath
