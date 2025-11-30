from __future__ import annotations
import numpy as np
import cv2 as cv

from processing.image_processor import FrameData


def make_visual_range_stage(v_min: int, v_max: int):
    """
    Intensity windowing als preprocessing stage.

    - Werkt op data.gray (8-bit of 16-bit).
    - Clampt naar [v_min, v_max] in de *huidige units* van gray
      (dus bij 8-bit: 0–255, bij 16-bit: 0–65535).
    - Schakelt daarna terug naar 8-bit gray (0–255), net als de Orbbec mapping.

    Let op:
    - Voor echte 16-bit IR (rechtstreeks uit camera / raw) kun je v_min/v_max
      in 16-bit units gebruiken (bijv. 10000–20000).
    - Voor standaard 8-bit bronnen gebruik je v_min/v_max in 0–255.
    """
    v_min = int(v_min)
    v_max = int(v_max)

    def stage(data: FrameData, dt: float) -> FrameData:
        gray = data.gray
        if gray is None:
            return data

        # Bewaar originele dtype om debug eventueel te kunnen interpreteren
        orig_dtype = gray.dtype

        # Naar float32 voor veilige berekening
        gray_f = gray.astype(np.float32)

        lo = float(v_min)
        hi = float(v_max)
        if hi <= lo:
            hi = lo + 1.0

        # Clamp en schaal naar [0, 1]
        gray_f = np.clip(gray_f, lo, hi)
        gray_f = (gray_f - lo) / (hi - lo)

        # Map naar 0–255 uint8
        gray_8 = (gray_f * 255.0).astype(np.uint8)

        data.gray = gray_8
        data.bgr = cv.cvtColor(gray_8, cv.COLOR_GRAY2BGR)
        data.debug["gray_visual_range"] = gray_8
        data.meta["visual_range"] = {
            "v_min": v_min,
            "v_max": v_max,
            "orig_dtype": str(orig_dtype),
        }

        return data

    return stage