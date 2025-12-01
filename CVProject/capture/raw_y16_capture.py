from __future__ import annotations
import cv2 as cv
import numpy as np
from pathlib import Path
from typing import Tuple, List

from capture.capture_base import Capture


class RawY16FolderCapture(Capture):
    """
    Leest een map met 16-bit IR PNG's *zonder* enige mapping.
    Dit is nodig voor analyse, zodat Visual Range stage
    precies hetzelfde werkt als in de echte Orbbec Y16 pipeline.
    """

    def __init__(self, folder: str):
        self.folder = Path(folder)
        self.files: List[Path] = sorted(self.folder.glob("*.png"))
        self.index = 0

        if len(self.files) == 0:
            raise RuntimeError(f"No PNG frames found in {folder}")

        # Check eerste frame
        tmp = cv.imread(str(self.files[0]), cv.IMREAD_UNCHANGED)
        if tmp is None or tmp.dtype != np.uint16:
            raise RuntimeError("Expected 16-bit PNG frames (uint16)")

        self._opened = True

    def is_opened(self) -> bool:
        return self._opened

    def read(self) -> Tuple[bool, np.ndarray]:
        if not self._opened:
            return False, None

        if self.index >= len(self.files):
            return False, None

        img16 = cv.imread(str(self.files[self.index]), cv.IMREAD_UNCHANGED)
        self.index += 1

        return True, img16  # <<<< BELANGRIJK: Y16, geen 8-bit

    def release(self) -> None:
        self._opened = False
