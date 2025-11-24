import cv2 as cv
from processing.image_processor import FrameData

def make_scale_stage(scale: float = 1.0):
    scale = float(scale)

    def stage(data: FrameData, dt: float) -> FrameData:
        if scale == 1.0:
            return data

        bgr = cv.resize(
            data.original_bgr,
            None,
            fx=scale,
            fy=scale,
            interpolation=cv.INTER_AREA,
        )
        data.bgr = bgr
        data.gray = cv.cvtColor(bgr, cv.COLOR_BGR2GRAY)
        data.meta["scale"] = scale
        return data

    return stage