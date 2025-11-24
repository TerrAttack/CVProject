import cv2 as cv
import numpy as np
from processing.image_processor import FrameData

def make_otsu_threshold_stage():
    def stage(data: FrameData, dt: float) -> FrameData:
        gray = data.gray
        if gray is None:
            return data

        if gray.dtype != np.uint8:
            g_min, g_max = gray.min(), gray.max()
            if g_max > g_min:
                gray8 = ((gray - g_min) * (255.0 / (g_max - g_min))).astype(np.uint8)
            else:
                gray8 = np.zeros_like(gray, dtype=np.uint8)
        else:
            gray8 = gray

        _, binary = cv.threshold(
            gray8, 0, 255,
            cv.THRESH_BINARY + cv.THRESH_OTSU
        )

        data.binary = binary
        data.debug["binary_otsu"] = binary
        data.meta["threshold"] = "otsu"

        return data

    return stage