from __future__ import annotations
import cv2 as cv
import numpy as np
from typing import List, Dict

from detection.detection_strategy import Blob, StrategyOutput, DetectionStrategy
from processing.image_processor import FrameData


class SimpleBlobStrategy(DetectionStrategy):
    def __init__(
        self,
        min_threshold=10,
        max_threshold=255,
        threshold_step=10,
        min_area=1,
        max_area=20000,
        min_dist=1,
        filter_by_color = True,
        blob_color = 255,
        draw=True,
        debug_gray=True,
    ):
        params = cv.SimpleBlobDetector_Params()
        params.minThreshold = min_threshold
        params.maxThreshold = max_threshold
        params.thresholdStep = threshold_step

        params.filterByArea = True
        params.minArea = min_area
        params.maxArea = max_area

        params.filterByColor = filter_by_color
        params.blobColor = blob_color
        params.minDistBetweenBlobs = min_dist
        params.filterByCircularity = False

        self.detector = cv.SimpleBlobDetector_create(params)

        self.draw = draw
        self.debug_gray = debug_gray

    def process_frame(self, frame: FrameData, dt: float) -> StrategyOutput:
        # Use preprocessed gray from the pipeline as detector input
        gray = frame.gray

        keypoints = self.detector.detect(gray)
        blobs = self._kp_to_blobs(keypoints)

        # *** IMPORTANT CHANGE ***
        # Base visualization on the *processed gray* (converted to BGR),
        # so you truly see what the detector sees
        vis = cv.cvtColor(gray, cv.COLOR_GRAY2BGR)

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

        # Minimal debug dict (you can ignore this for now)
        debug: Dict[str, np.ndarray] = {}
        if self.debug_gray:
            debug["gray"] = gray
        if frame.binary is not None:
            debug["binary"] = frame.binary

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
