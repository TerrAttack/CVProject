from __future__ import annotations
import numpy as np
import cv2 as cv
from processing.image_processor import FrameData


def make_visual_range_stage(v_min: int, v_max: int):
    """
    Nauwkeurige kopie van Orbbec / RawFolderCapture logic:
    - input = uint16 gray (0–65535)
    - clamp in [v_min, v_max]
    - normaliseer naar 0–255
    """

    v_min = int(v_min)
    v_max = int(v_max)

    def stage(data: FrameData, dt: float) -> FrameData:
        gray16 = data.gray

        if gray16 is None:
            return data
        if gray16.dtype != np.uint16:
            raise RuntimeError("VisualRangeStage requires uint16 gray input.")

        lo = float(v_min)
        hi = float(v_max)
        if hi <= lo:
            hi = lo + 1.0

        # 1. clamp in Y16 range
        clamped = np.clip(gray16.astype(np.float32), lo, hi)

        # 2. normaliseer naar 8-bit
        out8 = ((clamped - lo) / (hi - lo) * 255.0).astype(np.uint8)

        data.gray = out8
        data.bgr = cv.cvtColor(out8, cv.COLOR_GRAY2BGR)
        data.binary = None  # wordt opnieuw opgebouwd

        data.debug["visual_range"] = out8
        return data

    return stage
