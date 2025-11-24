import cv2 as cv
from typing import Tuple
import numpy as np
from .capture_base import Capture


class OpenCVCapture(Capture):
    def __init__(self, source=0):
        self.cap = cv.VideoCapture(source)

        if not self.cap.isOpened():
            raise RuntimeError(f"Failed to open OpenCV capture source: {source}")

    def is_opened(self) -> bool:
        return self.cap.isOpened()

    def read(self) -> Tuple[bool, np.ndarray]:
        return self.cap.read()

    def release(self) -> None:
        if self.cap:
            self.cap.release()
            self.cap = None