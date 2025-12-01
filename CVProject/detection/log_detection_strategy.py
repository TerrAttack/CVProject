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


class LoGBlobStrategy(DetectionStrategy):
    def __init__(
        self,
        sigma: float = 1,
        threshold_rel: float = 0.1,
        min_area: float = 1.0,
        max_area: float = 1e6,
        merge_distance: float = 1.0,  # nieuwe variable: afstand om blobs te mergen (px)
        draw: bool = True,
        debug_outputs: bool = True,
    ):
        """
        Eenvoudige LoG blobdetector (single-scale):

        - gray -> float32 in [0,1]
        - GaussianBlur met gegeven sigma
        - Laplacian -> |LoG|-respons
        - threshold op threshold_rel * max_response
        - connectedComponents op het mask
        - optional NMS: merge blobs binnen merge_distance
        """
        self.sigma = float(sigma)
        self.threshold_rel = float(threshold_rel)
        self.min_area = float(min_area)
        self.max_area = float(max_area)
        self.merge_distance = float(merge_distance)
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

        # 2) Gaussian blur + Laplacian
        blurred = cv.GaussianBlur(gray_f, (0, 0), self.sigma)
        log = cv.Laplacian(blurred, cv.CV_32F, ksize=3)

        response = (self.sigma ** 2) * np.abs(log)

        max_val = float(response.max())
        if max_val <= 0:
            mask = np.zeros_like(gray, dtype=np.uint8)
        else:
            thresh_val = self.threshold_rel * max_val
            mask = (response >= thresh_val).astype(np.uint8) * 255

        # 3) CCL op LoG-mask
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

            mask_label = (labels == label)
            mean_resp = float(response[mask_label].mean()) if mask_label.any() else 0.0

            blobs.append(
                Blob(
                    x=float(cx),
                    y=float(cy),
                    size=size,
                    area=area,
                    response=mean_resp,
                )
            )

        # 4) Optionele NMS / merge op basis van afstand
        if self.merge_distance > 0 and len(blobs) > 1:
            blobs = self._apply_distance_nms(blobs, self.merge_distance)

        # 5) Visualisatie basis
        if gray.dtype == np.uint8:
            vis_gray = gray
        else:
            vis_gray = (gray_f * 255.0).astype(np.uint8)

        vis = cv.cvtColor(vis_gray, cv.COLOR_GRAY2BGR)

        # 6) Blobs tekenen in dezelfde stijl als SimpleBlobStrategy
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
            f"LoG blobs: {len(blobs)}",
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
                resp_vis = (response / max_val * 255.0).astype(np.uint8)
            else:
                resp_vis = np.zeros_like(gray, dtype=np.uint8)
            debug["log_response"] = resp_vis
            debug["log_mask"] = mask

        return StrategyOutput(vis=vis, debug=debug, detections=blobs)

    def _apply_distance_nms(self, blobs: List[Blob], dist: float) -> List[Blob]:
        """Houd alleen de sterkste blob binnen een gegeven afstand (px)."""
        if len(blobs) <= 1:
            return blobs

        keep: List[Blob] = []
        dist2 = dist * dist

        # sorteer op aflopende response (sterkste eerst)
        sorted_blobs = sorted(blobs, key=lambda b: b.response, reverse=True)

        for b in sorted_blobs:
            ok = True
            for kb in keep:
                dx = kb.x - b.x
                dy = kb.y - b.y
                if dx * dx + dy * dy <= dist2:
                    ok = False
                    break
            if ok:
                keep.append(b)

        return keep
