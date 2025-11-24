import cv2 as cv
from processing.image_processor import FrameData

def make_gaussian_blur_stage(ksize: int = 3):
    if ksize % 2 == 0:
        ksize += 1

    def stage(data: FrameData, dt: float) -> FrameData:
        gray = data.gray
        if gray is None:
            return data

        blurred = cv.GaussianBlur(gray, (ksize, ksize), 0)
        data.debug["gray_blur"] = blurred
        data.gray = blurred
        data.meta["blur_ksize"] = ksize

        return data

    return stage