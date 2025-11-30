import cv2 as cv
from processing.image_processor import FrameData


def make_median_blur_stage(ksize: int = 3):
    """
    Median blur op de gray-channel.
    ksize moet een oneven integer >= 1 zijn.
    """
    if ksize < 1:
        ksize = 1
    if ksize % 2 == 0:
        ksize += 1

    def stage(data: FrameData, dt: float) -> FrameData:
        gray = data.gray
        if gray is None:
            return data

        blurred = cv.medianBlur(gray, ksize)
        data.debug["gray_median_blur"] = blurred
        data.gray = blurred
        data.meta["median_blur_ksize"] = ksize

        return data

    return stage