from __future__ import annotations
import cv2 as cv
import numpy as np
from pathlib import Path
from typing import Tuple, List

from .capture_base import Capture
from .orbbec_ir_capture import ir_to_8bit_percentile  # <- use same mapping as live capture


class RawFolderCapture(Capture):
    """
    Plays back a folder of 16-bit IR PNG frames as if it were a camera.

    - Reads uint16 grayscale PNGs from a folder (raw IR recording).
    - Uses the same 16->8 bit percentile mapping as OrbbecIRCapture
      (ir_to_8bit_percentile), so the output looks like your live view.
    - Converts to BGR so the rest of the pipeline can treat it like a normal camera.
    """

    def __init__(self, folder: str):
        self.folder = Path(folder)
        self.files: List[Path] = sorted(self.folder.glob("*.png"))
        self.index = 0

        if len(self.files) == 0:
            raise RuntimeError(f"No PNG frames found in: {folder}")

        # Probe first frame to get shape / sanity check
        tmp = cv.imread(str(self.files[0]), cv.IMREAD_UNCHANGED)
        if tmp is None:
            raise RuntimeError(f"Failed to read first frame in: {folder}")
        if tmp.ndim != 2:
            raise RuntimeError(f"Expected single-channel 16-bit PNGs, got shape {tmp.shape}")
        if tmp.dtype != np.uint16:
            print(f"[RawFolderCapture] WARNING: expected uint16, got {tmp.dtype}")

        self.h, self.w = tmp.shape
        self._opened = True

    def is_opened(self) -> bool:
        return self._opened

    def read(self) -> Tuple[bool, np.ndarray]:
        if not self._opened:
            return False, None

        if self.index >= len(self.files):
            # End of sequence
            return False, None

        path = self.files[self.index]
        img16 = cv.imread(str(path), cv.IMREAD_UNCHANGED)

        if img16 is None:
            print(f"[RawFolderCapture] Failed to read frame: {path}")
            return False, None

        if img16.ndim != 2:
            print(f"[RawFolderCapture] Unexpected shape {img16.shape} in {path}")
            return False, None

        # --- IMPORTANT ---
        # Map raw 16-bit IR to 8-bit using the same percentile mapping
        # that OrbbecIRCapture uses when visual range is disabled.
        img8 = ir_to_8bit_percentile(img16)

        img_bgr = cv.cvtColor(img8, cv.COLOR_GRAY2BGR)

        self.index += 1
        return True, img_bgr

    def release(self) -> None:
        self._opened = False