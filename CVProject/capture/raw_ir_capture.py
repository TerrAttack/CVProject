from __future__ import annotations
import cv2 as cv
import numpy as np
from pathlib import Path
from typing import Tuple, List

from .capture_base import Capture
from .orbbec_ir_capture import (
    ir_to_8bit_percentile,
    ir_to_8bit_window,
)


class RawFolderCapture(Capture):
    """
    Plays back a folder of 16-bit IR PNG frames as if it were a camera.

    - Reads uint16 grayscale PNGs from a folder (raw IR recording).
    - Can apply the same 16-bit VisualRange windowing as OrbbecIRCapture
      via set_visual_range(enabled, min, max).
    - Falls back to percentile-based mapping when visual range is disabled.
    - Converts to BGR so the rest of the pipeline can treat it like a normal camera.
    - Loopt automatisch door de frames in de folder.
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

        # Visual range settings (in raw 16-bit units, like OrbbecIRCapture)
        self._use_visual_range = False
        self._vr_min = 0
        self._vr_max = 5000

    # ------------------------------------------------------------------
    #   Public API (Capture)
    # ------------------------------------------------------------------
    def is_opened(self) -> bool:
        return self._opened

    def read(self) -> Tuple[bool, np.ndarray]:
        if not self._opened:
            return False, None

        if len(self.files) == 0:
            return False, None

        # Loop: als we bij het einde zijn, ga terug naar het begin
        if self.index >= len(self.files):
            self.index = 0

        path = self.files[self.index]
        img16 = cv.imread(str(path), cv.IMREAD_UNCHANGED)

        if img16 is None:
            print(f"[RawFolderCapture] Failed to read frame: {path}")
            return False, None

        if img16.ndim != 2:
            print(f"[RawFolderCapture] Unexpected shape {img16.shape} in {path}")
            return False, None

        # 16-bit -> 8-bit mapping
        if self._use_visual_range:
            img8 = ir_to_8bit_window(img16, self._vr_min, self._vr_max)
        else:
            img8 = ir_to_8bit_percentile(img16)

        img_bgr = cv.cvtColor(img8, cv.COLOR_GRAY2BGR)

        self.index += 1
        return True, img_bgr

    def release(self) -> None:
        self._opened = False

    # ------------------------------------------------------------------
    #   VisualRange API (mimic OrbbecIRCapture)
    # ------------------------------------------------------------------
    def set_visual_range(self, enabled: bool, min_val: int, max_val: int) -> None:
        """
        Enable/disable visual range windowing on raw 16-bit PNG frames.

        Parameters are in raw 16-bit units (0..65535), typically something like
        10000..20000 to mimic Orbbec's VisualRange behaviour.
        """
        self._use_visual_range = bool(enabled)
        self._vr_min = int(min_val)
        self._vr_max = int(max_val)

        if self._vr_max <= self._vr_min:
            self._vr_max = self._vr_min + 1

        print(f"[RawFolderCapture] VisualRange set: enabled={self._use_visual_range}, "
              f"min={self._vr_min}, max={self._vr_max}")
