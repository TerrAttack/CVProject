import cv2 as cv
from typing import Tuple, Union
import numpy as np
from .capture_base import Capture


class OpenCVCapture(Capture):
    def __init__(self, source: Union[int, str] = 0):
        self.source = source
        self.cap = cv.VideoCapture(source)

        if not self.cap.isOpened():
            raise RuntimeError(f"Failed to open OpenCV capture source: {source}")

        # Beschouw string source als bestand -> loop bij einde
        self.is_file = isinstance(source, str)

    def is_opened(self) -> bool:
        return self.cap is not None and self.cap.isOpened()

    def read(self) -> Tuple[bool, np.ndarray]:
        if self.cap is None:
            return False, None

        ok, frame = self.cap.read()

        if not ok or frame is None:
            # Voor video-bestanden: terug naar begin en opnieuw proberen
            if self.is_file:
                self.cap.set(cv.CAP_PROP_POS_FRAMES, 0)
                ok, frame = self.cap.read()
                if not ok or frame is None:
                    print("[OpenCVCapture] End of file or cannot read after reset.")
                    return False, None
            else:
                # Voor camera's: echt fout
                print("[OpenCVCapture] Failed to read from non-file source.")
                return False, None

        return True, frame

    def release(self) -> None:
        if self.cap is not None:
            self.cap.release()
            self.cap = None
