from __future__ import annotations

from typing import Tuple
import numpy as np
import cv2 as cv

import pyorbbecsdk as ob
from pyorbbecsdk import OBError, OBFormat  # OBSensorType is NOT used – your SDK has no IR enum.

from .capture_base import Capture


# ============================================================
#   SAFE IR NORMALIZATION (kept unchanged)
# ============================================================

def ir_to_8bit(ir16: np.ndarray) -> np.ndarray:
    """Convert 16-bit IR to 8-bit, clipping extremes."""
    if ir16.size == 0:
        return np.zeros_like(ir16, dtype=np.uint8)
    v_min, v_max = np.percentile(ir16, (2, 98))
    if v_max <= v_min:
        return np.zeros_like(ir16, dtype=np.uint8)
    ir_clipped = np.clip(ir16, v_min, v_max)
    ir_norm = (ir_clipped - v_min) / (v_max - v_min)
    return (ir_norm * 255).astype(np.uint8)


# ============================================================
#   FINAL FIXED CAPTURE CLASS
# ============================================================

class OrbbecIRCapture(Capture):
    """
    IR-only capture from an Orbbec camera.
    Automatically detects any sensor that supports Y16/Y8.
    """

    def __init__(self, width: int = 640, height: int = 576, fps: int = 30):
        self.width = width
        self.height = height
        self.fps = fps

        self.pipeline = ob.Pipeline()
        self.config = ob.Config()
        self._opened = False

        self._setup_stream()

    # ============================================================
    #   BASIC API
    # ============================================================

    def is_opened(self) -> bool:
        return self._opened

    def isOpened(self) -> bool:  # legacy shim
        return self._opened

    def read(self) -> Tuple[bool, np.ndarray]:
        if not self._opened:
            return False, None

        try:
            frames = self.pipeline.wait_for_frames(1000)
        except OBError as e:
            print("[OrbbecIRCapture] Frame wait error:", e)
            return False, None

        ir_frame = frames.get_ir_frame()
        if ir_frame is None:
            print("[OrbbecIRCapture] No IR frame received")
            return False, None

        w, h = ir_frame.get_width(), ir_frame.get_height()
        fmt = ir_frame.get_format()

        if fmt == OBFormat.Y16:
            data = np.frombuffer(ir_frame.get_data(), np.uint16).reshape(h, w)
        elif fmt == OBFormat.Y16:
            data = np.frombuffer(ir_frame.get_data(), np.uint8).reshape(h, w)
        else:
            print("[OrbbecIRCapture] Unsupported format:", fmt)
            return False, None

        ir8 = ir_to_8bit(data) if data.dtype == np.uint16 else data
        bgr = cv.cvtColor(ir8, cv.COLOR_GRAY2BGR)
        return True, bgr

    def release(self):
        if self._opened:
            try:
                self.pipeline.stop()
            except OBError:
                pass
            self._opened = False
            print("[OrbbecIRCapture] Stopped IR stream.")

    # ============================================================
    #   FINAL WORKING SETUP
    # ============================================================

    def _setup_stream(self):
        # Get device
        try:
            dev = self.pipeline.get_device()
        except OBError as e:
            print("[OrbbecIRCapture] No device:", e)
            return

        # Get sensor list
        try:
            sl = dev.get_sensor_list()
        except OBError as e:
            print("[OrbbecIRCapture] Cannot read sensor list:", e)
            return

        # AUTO-DETECT IR SENSOR BY FORMAT (Y16/Y8)
        ir_sensor = None

        try:
            count = sl.get_count()
        except OBError:
            print("[OrbbecIRCapture] get_count() failed.")
            return

        for i in range(count):
            try:
                s = sl.get_sensor_by_index(i)
                profiles = s.get_stream_profile_list()
            except OBError:
                continue

            # Check for Y16/Y8 support
            for fmt in [OBFormat.Y16, OBFormat.Y16]:
                try:
                    profiles.get_video_stream_profile(
                        self.width, self.height, fmt, self.fps
                    )
                    ir_sensor = s
                    print(f"[OrbbecIRCapture] IR sensor detected at index {i} (format={fmt}).")
                    break
                except OBError:
                    pass

            if ir_sensor is not None:
                break

        if ir_sensor is None:
            print("[OrbbecIRCapture] No IR-capable sensor found.")
            return

        # Try preferred Y16
        profiles = ir_sensor.get_stream_profile_list()
        try:
            ir_profile = profiles.get_video_stream_profile(
                self.width, self.height, OBFormat.Y16, self.fps
            )
            print("[OrbbecIRCapture] Using Y16 IR stream.")
        except OBError:
            # fallback to Y8
            try:
                ir_profile = profiles.get_video_stream_profile(
                    self.width, self.height, OBFormat.Y16, self.fps
                )
                print("[OrbbecIRCapture] Using fallback Y8 IR stream.")
            except OBError:
                print("[OrbbecIRCapture] Could not acquire IR profile.")
                return

        self.config.enable_stream(ir_profile)

        try:
            self.pipeline.start(self.config)
            self._opened = True
            print(
                f"[OrbbecIRCapture] Started IR stream {self.width}x{self.height}@{self.fps}"
            )
        except OBError as e:
            print("[OrbbecIRCapture] Failed to start pipeline:", e)
            self._opened = False
