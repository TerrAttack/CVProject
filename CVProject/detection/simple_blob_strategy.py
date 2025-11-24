from __future__ import annotations
import cv2 as cv
import numpy as np
from typing import List, Dict

from detection.detection_strategy import Blob, StrategyOutput, DetectionStrategy
from processing.image_processor import FrameData


class SimpleBlobStrategy(DetectionStrategy):
    def __init__(
        self,
        min_threshold: float = 100,
        max_threshold: float = 255,
        threshold_step: float = 10,
        min_area: float = 1,
        max_area: float = 20000,
        min_dist: float = 1,
        use_blur: bool = True,
        ksize: int = 3,
        draw: bool = True,
        debug_gray: bool = True,
    ):
        params = cv.SimpleBlobDetector_Params()
        params.minThreshold = min_threshold
        params.maxThreshold = max_threshold
        params.thresholdStep = threshold_step
        params.filterByArea = True
        params.minArea = min_area
        params.maxArea = max_area
        params.minDistBetweenBlobs = min_dist
        params.filterByCircularity = False
        params.filterByColor = True
        params.blobColor = 255

        self.detector = cv.SimpleBlobDetector_create(params)

        self.use_blur = use_blur
        self.ksize = ksize if ksize % 2 == 1 else ksize + 1
        self.draw = draw
        self.debug_gray = debug_gray

    def process_frame(self, frame: FrameData, dt: float) -> StrategyOutput:
        gray = frame.gray

        if self.use_blur:
            gray = cv.GaussianBlur(gray, (self.ksize, self.ksize), 0)

        keypoints = self.detector.detect(gray)
        blobs = self._kp_to_blobs(keypoints)

        vis = frame.bgr.copy()
        if self.draw:
            vis = cv.drawKeypoints(
                vis,
                keypoints,
                None,
                (0, 0, 255),
                cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS,
            )

        # Blob counter text (first line)
        cv.putText(
            vis,
            f"Blobs: {len(blobs)}",
            (5, 20),
            cv.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
            cv.LINE_AA,
        )

        debug: Dict[str, np.ndarray] = {}
        if self.debug_gray:
            debug["gray"] = gray

        return StrategyOutput(
            vis=vis,
            debug=debug,
            detections=blobs,
        )

    def _kp_to_blobs(self, kps) -> List[Blob]:
        result: List[Blob] = []
        for kp in kps:
            x, y = kp.pt
            size = kp.size
            area = float(np.pi * (size / 2) ** 2)
            result.append(Blob(x, y, size, area, kp.response))
        return result