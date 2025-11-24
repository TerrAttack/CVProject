from __future__ import annotations
import cv2 as cv
import numpy as np
from typing import List, Dict

from detection.detection_strategy import Blob, StrategyOutput, DetectionStrategy
from processing.image_processor import FrameData


class SimpleBlobStrategy(DetectionStrategy):
    def __init__(
        self,
        min_threshold=5,
        max_threshold=255,
        threshold_step=5,
        filter_by_color = True,
        blob_color = 255,
        min_area=1,
        max_area=20000,
        min_dist=1,
        draw=True,
        debug_gray=True,
    ):
        params = cv.SimpleBlobDetector_Params()
        params.minThreshold = min_threshold
        params.maxThreshold = max_threshold
        params.thresholdStep = threshold_step

        params.filterByColor = filter_by_color
        params.blobColor = blob_color

        params.filterByArea = True
        params.minArea = min_area
        params.maxArea = max_area

        params.minDistBetweenBlobs = min_dist
        params.filterByCircularity = False

        self.detector = cv.SimpleBlobDetector_create(params)

        # Blur is now expected to be done in the preprocessing pipeline
        self.draw = draw
        self.debug_gray = debug_gray

    # ——— IMPLEMENTATION ———

    def process_frame(self, frame: FrameData, dt: float) -> StrategyOutput:
        # Use whatever the pipeline gives us as gray (already scaled / CLAHE / blurred)
        gray = frame.gray

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

        # Blob counter overlay
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