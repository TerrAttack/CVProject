import cv2 as cv
import numpy as np
from processing.image_processor import FrameData


def _get_morph_input(data: FrameData):
    """
    Kies waar we de morfologische operatie op doen:
    - bij voorkeur op data.binary (als die bestaat, bv. na Otsu)
    - anders op data.gray
    Zorgt dat het beeld uint8 is.
    """
    img = data.binary if data.binary is not None else data.gray
    if img is None:
        return None

    if img.dtype != np.uint8:
        i_min, i_max = img.min(), img.max()
        if i_max > i_min:
            img8 = ((img - i_min) * (255.0 / (i_max - i_min))).astype(np.uint8)
        else:
            img8 = np.zeros_like(img, dtype=np.uint8)
    else:
        img8 = img

    return img8


def make_opening_stage(ksize: int = 3):
    """
    Morfologische opening (erosie gevolgd door dilatie).

    - Werkt op binary als die er is, anders op gray.
    - Resultaat wordt teruggezet in zowel gray als, indien aanwezig, binary.
    """
    if ksize < 1:
        ksize = 1
    if ksize % 2 == 0:
        ksize += 1

    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (ksize, ksize))

    def stage(data: FrameData, dt: float) -> FrameData:
        img = _get_morph_input(data)
        if img is None:
            return data

        opened = cv.morphologyEx(img, cv.MORPH_OPEN, kernel, iterations=1)

        # Als er al een binary was, overschrijf die ook
        if data.binary is not None:
            data.binary = opened

        data.gray = opened
        data.debug["morph_open"] = opened
        data.meta["morph_open_ksize"] = ksize

        return data

    return stage


def make_closing_stage(ksize: int = 3):
    """
    Morfologische closing (dilatie gevolgd door erosie).

    - Werkt op binary als die er is, anders op gray.
    - Resultaat wordt teruggezet in zowel gray als, indien aanwezig, binary.
    """
    if ksize < 1:
        ksize = 1
    if ksize % 2 == 0:
        ksize += 1

    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (ksize, ksize))

    def stage(data: FrameData, dt: float) -> FrameData:
        img = _get_morph_input(data)
        if img is None:
            return data

        closed = cv.morphologyEx(img, cv.MORPH_CLOSE, kernel, iterations=1)

        if data.binary is not None:
            data.binary = closed

        data.gray = closed
        data.debug["morph_close"] = closed
        data.meta["morph_close_ksize"] = ksize

        return data

    return stage
