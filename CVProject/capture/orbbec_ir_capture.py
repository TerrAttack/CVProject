from __future__ import annotations

from typing import Tuple
import numpy as np
import cv2 as cv

import pyorbbecsdk as ob
from pyorbbecsdk import OBError, OBFormat

from .capture_base import Capture


def ir_to_8bit_percentile(ir16: np.ndarray) -> np.ndarray:
    """
    Fallback 16->8 conversion using percentiles (2–98%) to avoid spikes.
    Used when visual range is disabled.
    """
    if ir16.size == 0:
        return np.zeros_like(ir16, dtype=np.uint8)

    v_min, v_max = np.percentile(ir16, (2, 98))
    if v_max <= v_min:
        return np.zeros_like(ir16, dtype=np.uint8)

    ir_clipped = np.clip(ir16, v_min, v_max)
    ir_norm = (ir_clipped - v_min) / (v_max - v_min)
    return (ir_norm * 255.0).astype(np.uint8)


def ir_to_8bit_window(ir16: np.ndarray, v_min: int, v_max: int) -> np.ndarray:
    """
    16-bit windowing: clamp to [v_min, v_max] and map linearly to 0–255.
    This mimics the IR VisualRange effect much more closely.
    """
    if ir16.size == 0:
        return np.zeros_like(ir16, dtype=np.uint8)

    lo = int(v_min)
    hi = int(v_max)

    # Keep sane bounds and avoid div by zero
    if hi <= lo:
        hi = lo + 1

    # Orbbec IR values are typically well below 65535, but clamp anyway
    lo = max(0, lo)
    hi = max(1, min(65535, hi))

    ir = ir16.astype(np.float32)
    ir = np.clip(ir, lo, hi)
    ir = (ir - lo) / (hi - lo) * 255.0
    return ir.astype(np.uint8)


class OrbbecIRCapture(Capture):
    """
    IR-only capture from an Orbbec camera.

    - Reads Y16 or Y8 IR frames.
    - Optionally applies a 16-bit intensity window (visual range).
    - Outputs an 8-bit BGR image for the rest of the pipeline.
    """

    def __init__(self, width: int = 640, height: int = 576, fps: int = 30) -> None:
        self.width = width
        self.height = height
        self.fps = fps

        self.pipeline = ob.Pipeline()
        self.config = ob.Config()
        self._opened = False

        # Visual range settings (in raw 16-bit units)
        self._use_visual_range = False
        self._vr_min = 0
        self._vr_max = 5000

        self._setup_stream()

    # ---- Public API -------------------------------------------------

    def is_opened(self) -> bool:
        return self._opened

    def isOpened(self) -> bool:  # legacy shim
        return self._opened

    def set_visual_range(self, enabled: bool, min_val: int, max_val: int) -> None:
        """
        Enable/disable visual range windowing on raw 16-bit IR.
        """
        self._use_visual_range = bool(enabled)
        self._vr_min = int(min_val)
        self._vr_max = int(max_val)
        # no need to restart the stream; applied per-frame in read()

    # ---- Capture interface -------------------------------------------

    def read(self) -> Tuple[bool, np.ndarray]:
        if not self._opened:
            return False, None  # type: ignore[return-value]

        try:
            frame_set = self.pipeline.wait_for_frames(1000)
        except OBError as e:
            print("[OrbbecIRCapture] Error waiting for frames:", e)
            return False, None  # type: ignore[return-value]

        ir_frame = frame_set.get_ir_frame()
        if ir_frame is None:
            print("[OrbbecIRCapture] No IR frame received")
            return False, None  # type: ignore[return-value]

        w = ir_frame.get_width()
        h = ir_frame.get_height()
        fmt = ir_frame.get_format()

        # Y16: full 16-bit IR – apply visual window or percentile mapping
        if fmt == OBFormat.Y16:
            data16 = np.frombuffer(ir_frame.get_data(), np.uint16).reshape(h, w)

            if self._use_visual_range:
                ir8 = ir_to_8bit_window(data16, self._vr_min, self._vr_max)
            else:
                ir8 = ir_to_8bit_percentile(data16)

        # Y8: some devices already give 8-bit IR
        elif fmt == OBFormat.Y8:
            data8 = np.frombuffer(ir_frame.get_data(), np.uint8).reshape(h, w)
            ir8 = data8
        else:
            print("[OrbbecIRCapture] Unsupported IR format:", fmt)
            return False, None  # type: ignore[return-value]

        bgr = cv.cvtColor(ir8, cv.COLOR_GRAY2BGR)
        return True, bgr

    def release(self) -> None:
        if self._opened:
            try:
                self.pipeline.stop()
            except OBError:
                pass
            self._opened = False
            print("[OrbbecIRCapture] Stopped IR stream.")

    # ---- internal setup ----------------------------------------------

    def _setup_stream(self) -> None:
        try:
            dev = self.pipeline.get_device()
        except OBError as e:
            print("[OrbbecIRCapture] No Orbbec device:", e)
            self._opened = False
            return

        try:
            sensor_list = dev.get_sensor_list()
        except OBError as e:
            print("[OrbbecIRCapture] Failed to get sensor list:", e)
            self._opened = False
            return

        ir_sensor = None
        try:
            count = sensor_list.get_count()
        except OBError as e:
            print("[OrbbecIRCapture] get_count failed:", e)
            self._opened = False
            return

        for i in range(count):
            try:
                sensor = sensor_list.get_sensor_by_index(i)
                profiles = sensor.get_stream_profile_list()
            except OBError:
                continue

            for fmt in [OBFormat.Y16, OBFormat.Y8]:
                try:
                    profiles.get_video_stream_profile(
                        self.width, self.height, fmt, self.fps
                    )
                    ir_sensor = sensor
                    print(f"[OrbbecIRCapture] IR sensor detected at index {i} (format={fmt}).")
                    break
                except OBError:
                    pass

            if ir_sensor is not None:
                break

        if ir_sensor is None:
            print("[OrbbecIRCapture] No IR-capable sensor found.")
            self._opened = False
            return

        try:
            profiles = ir_sensor.get_stream_profile_list()
        except OBError as e:
            print("[OrbbecIRCapture] Failed to get IR stream profiles:", e)
            self._opened = False
            return

        # Prefer Y16, fall back to Y8
        try:
            ir_profile = profiles.get_video_stream_profile(
                self.width, self.height, OBFormat.Y16, self.fps
            )
            print("[OrbbecIRCapture] Using Y16 IR stream.")
        except OBError:
            try:
                ir_profile = profiles.get_video_stream_profile(
                    self.width, self.height, OBFormat.Y8, self.fps
                )
                print("[OrbbecIRCapture] Using Y8 IR stream.")
            except OBError as e:
                print("[OrbbecIRCapture] Could not acquire IR profile:", e)
                self._opened = False
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
