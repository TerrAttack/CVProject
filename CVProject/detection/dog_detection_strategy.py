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


class DoGBlobStrategy(DetectionStrategy):
    """
    Difference of Gaussians (DoG) blob detector (single scale).

    - gray -> float32 [0,1]
    - Gaussian blur met sigma1 en sigma2
    - DoG = |G(sigma1) - G(sigma2)|
    - threshold op relatieve drempel
    - CCL -> blobs

    Voor jouw reflectors:
      - sigma1 ~ 1.0–1.5
      - sigma2 ~ 2.0–3.0
      - threshold_rel ~ 0.1–0.3
    """

    def __init__(
        self,
        sigma1: float = 1.5,
        sigma2: float = 2.5,
        threshold_rel: float = 0.2,
        min_area: float = 3.0,
        max_area: float = 1e6,
        draw: bool = True,
        debug_outputs: bool = True,
    ):
        if sigma2 <= sigma1:
            raise ValueError("sigma2 must be > sigma1 for DoG.")
        self.sigma1 = float(sigma1)
        self.sigma2 = float(sigma2)
        self.threshold_rel = float(threshold_rel)
        self.min_area = float(min_area)
        self.max_area = float(max_area)
        self.draw = draw
        self.debug_outputs = debug_outputs

    def process_frame(self, frame: FrameData, dt: float) -> StrategyOutput:
        gray = frame.gray
        if gray is None:
            vis = frame.bgr if frame.bgr is not None else frame.original_bgr
            return StrategyOutput(vis=vis, debug={}, detections=[])

        # 1) Normaliseer naar float32 [0,1]
        if gray.dtype == np.uint8:
            gray_f = gray.astype(np.float32) / 255.0
        else:
            g_min, g_max = gray.min(), gray.max()
            if g_max > g_min:
                gray_f = (gray.astype(np.float32) - g_min) / (g_max - g_min)
            else:
                gray_f = np.zeros_like(gray, dtype=np.float32)

        # 2) Gaussian blurs
        blur1 = cv.GaussianBlur(gray_f, (0, 0), self.sigma1)
        blur2 = cv.GaussianBlur(gray_f, (0, 0), self.sigma2)

        # 3) DoG-respons
        dog = np.abs(blur1 - blur2)

        max_val = float(dog.max())
        if max_val <= 0:
            mask = np.zeros_like(gray, dtype=np.uint8)
        else:
            thresh_val = self.threshold_rel * max_val
            mask = (dog >= thresh_val).astype(np.uint8) * 255

        # 4) CCL op mask
        num_labels, labels, stats, centroids = cv.connectedComponentsWithStats(
            mask, connectivity=8
        )

        blobs: List[Blob] = []
        h, w = gray.shape[:2]

        for label in range(1, num_labels):  # 0 = achtergrond
            area = float(stats[label, cv.CC_STAT_AREA])
            if area < self.min_area or area > self.max_area:
                continue

            cx, cy = centroids[label]
            if not (0 <= cx < w and 0 <= cy < h):
                continue

            radius = np.sqrt(area / np.pi)
            size = float(2.0 * radius)

            lbl_mask = (labels == label)
            mean_resp = float(dog[lbl_mask].mean()) if lbl_mask.any() else 0.0

            blobs.append(
                Blob(
                    x=float(cx),
                    y=float(cy),
                    size=size,
                    area=area,
                    response=mean_resp,
                )
            )

        # 5) Visualisatie – zelfde style als Simple/LoG/CCL/DoH
        if gray.dtype == np.uint8:
            vis_gray = gray
        else:
            vis_gray = (gray_f * 255.0).astype(np.uint8)

        vis = cv.cvtColor(vis_gray, cv.COLOR_GRAY2BGR)

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
            f"DoG blobs: {len(blobs)}",
            (5, 20),
            cv.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
            cv.LINE_AA,
        )

        debug: Dict[str, np.ndarray] = {}
        if self.debug_outputs:
            if max_val > 0:
                dog_vis = (dog / max_val * 255.0).astype(np.uint8)
            else:
                dog_vis = np.zeros_like(gray, dtype=np.uint8)
            debug["dog_response"] = dog_vis
            debug["dog_mask"] = mask

        return StrategyOutput(vis=vis, debug=debug, detections=blobs)
