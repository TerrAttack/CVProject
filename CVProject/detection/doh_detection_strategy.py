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


class DoHBlobStrategy(DetectionStrategy):
    """
    Single-scale Determinant of Hessian (DoH) blob detector.

    Stap:
      - gray -> float32 [0,1]
      - Gaussian blur met sigma
      - 2e orde afgeleiden (Lxx, Lyy, Lxy) via Sobel
      - det(H) = Lxx * Lyy - Lxy^2
      - threshold op relatieve drempel
      - CCL op mask -> blobs
    """

    def __init__(
        self,
        sigma: float = 2.0,
        threshold_rel: float = 0.2,
        min_area: float = 1.0,
        max_area: float = 1e6,
        draw: bool = True,
        debug_outputs: bool = True,
    ):
        self.sigma = float(sigma)
        self.threshold_rel = float(threshold_rel)
        self.min_area = float(min_area)
        self.max_area = float(max_area)
        self.draw = draw
        self.debug_outputs = debug_outputs

    def _gaussian_blur(self, gray_f: np.ndarray) -> np.ndarray:
        # ksize = 0 laat OpenCV zelf afleiden op basis van sigma
        return cv.GaussianBlur(gray_f, (0, 0), self.sigma)

    def _hessian_det(self, img: np.ndarray) -> np.ndarray:
        # img: float32 [0,1], al geblurd
        # 1e en 2e orde afgeleiden via Sobel
        Lx = cv.Sobel(img, cv.CV_32F, 1, 0, ksize=3)
        Ly = cv.Sobel(img, cv.CV_32F, 0, 1, ksize=3)

        Lxx = cv.Sobel(Lx, cv.CV_32F, 1, 0, ksize=3)
        Lyy = cv.Sobel(Ly, cv.CV_32F, 0, 1, ksize=3)

        Lxy = cv.Sobel(Lx, cv.CV_32F, 0, 1, ksize=3)

        det = Lxx * Lyy - (Lxy * Lxy)
        return det

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

        # 2) Gaussian blur
        blurred = self._gaussian_blur(gray_f)

        # 3) Determinant van Hessian
        det = self._hessian_det(blurred)

        # We zoeken hier blobs met positieve determinant (heldere blobs op donkere achtergrond)
        det_pos = np.maximum(det, 0)
        max_val = float(det_pos.max())

        if max_val <= 0:
            mask = np.zeros_like(gray, dtype=np.uint8)
        else:
            thresh_val = self.threshold_rel * max_val
            mask = (det_pos >= thresh_val).astype(np.uint8) * 255

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

            label_mask = (labels == label)
            mean_det = float(det_pos[label_mask].mean()) if label_mask.any() else 0.0

            blobs.append(Blob(
                x=float(cx),
                y=float(cy),
                size=size,
                area=area,
                response=mean_det,
            ))

        # 5) Visualisatie (zelfde stijl als SimpleBlob / LoG / CCL)
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
            f"DoH blobs: {len(blobs)}",
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
                det_vis = (det_pos / max_val * 255.0).astype(np.uint8)
            else:
                det_vis = np.zeros_like(gray, dtype=np.uint8)
            debug["doh_det"] = det_vis
            debug["doh_mask"] = mask

        return StrategyOutput(vis=vis, debug=debug, detections=blobs)
