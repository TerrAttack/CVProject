from __future__ import annotations
import cv2 as cv
import numpy as np
from typing import List, Dict

from detection.detection_strategy import Blob, StrategyOutput, DetectionStrategy
from processing.image_processor import FrameData


def _blobs_to_keypoints(blobs: List[Blob]):
    kps = []
    for b in blobs:
        kps.append(cv.KeyPoint(float(b.x), float(b.y), float(b.size)))
    return kps


class CCLBlobStrategy(DetectionStrategy):
    def __init__(
        self,
        min_area: float = 1.0,
        max_area: float = 1e6,
        connectivity: int = 8,
        draw: bool = True,
        debug_outputs: bool = True,
    ):
        """
        Connected Components Labeling op een binair beeld.

        - Gebruikt bij voorkeur `frame.binary` (bv. van Otsu-stage).
        - Als `frame.binary` None is, doet zelf een Otsu threshold op gray.
        """
        self.min_area = float(min_area)
        self.max_area = float(max_area)
        self.connectivity = 4 if connectivity == 4 else 8
        self.draw = draw
        self.debug_outputs = debug_outputs

    def _ensure_binary(self, frame: FrameData) -> np.ndarray:
        if frame.binary is not None:
            binary = frame.binary
        else:
            gray = frame.gray
            if gray is None:
                raise ValueError("No gray image available for CCLBlobStrategy.")

            if gray.dtype != np.uint8:
                g_min, g_max = gray.min(), gray.max()
                if g_max > g_min:
                    gray8 = ((gray - g_min) * (255.0 / (g_max - g_min))).astype(np.uint8)
                else:
                    gray8 = np.zeros_like(gray, dtype=np.uint8)
            else:
                gray8 = gray

            _, binary = cv.threshold(
                gray8, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU
            )

        if binary.dtype != np.uint8:
            binary = binary.astype(np.uint8)
        binary = (binary > 0).astype(np.uint8) * 255
        return binary

    def process_frame(self, frame: FrameData, dt: float) -> StrategyOutput:
        if frame.gray is None and frame.binary is None:
            vis = frame.bgr if frame.bgr is not None else frame.original_bgr
            return StrategyOutput(vis=vis, debug={}, detections=[])

        binary = self._ensure_binary(frame)

        num_labels, labels, stats, centroids = cv.connectedComponentsWithStats(
            binary, connectivity=self.connectivity
        )

        blobs: List[Blob] = []
        h, w = binary.shape[:2]

        for label in range(1, num_labels):  # 0 = achtergrond
            area = float(stats[label, cv.CC_STAT_AREA])
            if area < self.min_area or area > self.max_area:
                continue

            cx, cy = centroids[label]
            if not (0 <= cx < w and 0 <= cy < h):
                continue

            radius = np.sqrt(area / np.pi)
            size = float(2.0 * radius)

            blobs.append(Blob(
                x=float(cx),
                y=float(cy),
                size=size,
                area=area,
                response=area,
            ))

        # Basisbeeld: gray of binary
        base_gray = frame.gray
        if base_gray is None:
            base_gray = binary
        if base_gray.dtype != np.uint8:
            g_min, g_max = base_gray.min(), base_gray.max()
            if g_max > g_min:
                base_gray = ((base_gray - g_min) * (255.0 / (g_max - g_min))).astype(np.uint8)
            else:
                base_gray = np.zeros_like(base_gray, dtype=np.uint8)

        vis = cv.cvtColor(base_gray, cv.COLOR_GRAY2BGR)

        if self.draw:
            kps = _blobs_to_keypoints(blobs)
            vis = cv.drawKeypoints(
                vis,
                kps,
                None,
                (0, 0, 255),
                cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS,
            )

        cv.putText(
            vis,
            f"CCL blobs: {len(blobs)}",
            (5, 20),
            cv.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
            cv.LINE_AA,
        )

        debug: Dict[str, np.ndarray] = {}
        if self.debug_outputs:
            debug["binary_ccl"] = binary
            debug["gray_input"] = base_gray

        return StrategyOutput(vis=vis, debug=debug, detections=blobs)
