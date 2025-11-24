import cv2 as cv
import numpy as np
from processing.image_processor import FrameData

def make_clahe_stage(clip_limit: float = 2.0, tile_grid_size=(8, 8)):
    clahe = cv.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)

    def stage(data: FrameData, dt: float) -> FrameData:
        gray = data.gray
        if gray is None:
            return data

        if gray.dtype != np.uint8:
            g_min, g_max = gray.min(), gray.max()
            if g_max > g_min:
                gray = ((gray - g_min) * (255.0 / (g_max - g_min))).astype(np.uint8)
            else:
                gray = np.zeros_like(gray, dtype=np.uint8)

        clahe_gray = clahe.apply(gray)
        data.debug["gray_clahe"] = clahe_gray
        data.gray = clahe_gray
        data.meta["clahe"] = {"clip_limit": clip_limit, "tile_grid": tile_grid_size}

        return data

    return stage