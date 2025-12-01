from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List
from pathlib import Path

import cv2 as cv
import numpy as np

from processing.image_processor import CompositeProcessor
from processing.stages import (
    make_gaussian_blur_stage,
    make_median_blur_stage,
    make_otsu_threshold_stage,
)
from detection.simple_blob_strategy import SimpleBlobStrategy
from detection.ccl_detection_strategy import CCLBlobStrategy
from detection.doh_detection_strategy import DoHBlobStrategy
from detection.dog_detection_strategy import DoGBlobStrategy
from detection.log_detection_strategy import LoGBlobStrategy
from detection.detection_strategy import Blob
from capture.orbbec_ir_capture import ir_to_8bit_window  # same VR mapping as live Orbbec


# We know the recording should always contain 12 blobs
EXPECTED_BLOBS = 12

# Max distance (pixels) to consider blobs “the same” between frames
MAX_MATCH_DIST = 30.0

# Visual range in Y16 space, same as your GUI defaults
VR_MIN = 15000
VR_MAX = 20000


@dataclass
class FrameMetrics:
    index: int
    count: int
    fp: int
    fn: int
    jitter_mean: float  # -1 if no matches for this frame


def blob_positions(blobs: List[Blob]) -> np.ndarray:
    if not blobs:
        return np.empty((0, 2), dtype=np.float32)
    return np.array([[b.x, b.y] for b in blobs], dtype=np.float32)


def match_blobs(prev: List[Blob], curr: List[Blob], max_dist: float) -> List[float]:
    p = blob_positions(prev)
    c = blob_positions(curr)
    if p.shape[0] == 0 or c.shape[0] == 0:
        return []

    dists = np.linalg.norm(p[:, None, :] - c[None, :, :], axis=2)
    used_p = set()
    used_c = set()
    matches: List[float] = []

    while True:
        minval = math.inf
        mini = -1
        minj = -1

        for i in range(dists.shape[0]):
            if i in used_p:
                continue
            for j in range(dists.shape[1]):
                if j in used_c:
                    continue
                d = float(dists[i, j])
                if d < minval:
                    minval = d
                    mini = i
                    minj = j

        if mini < 0 or minj < 0:
            break
        if minval > max_dist:
            break

        used_p.add(mini)
        used_c.add(minj)
        matches.append(minval)

    return matches


def analyze_strategy() -> None:
    # 1) Collect all raw Y16 frames
    folder = Path("data/raw_ir")
    files = sorted(folder.glob("*.png"))
    if not files:
        print(f"[analyze_recording] No frames found in {folder}")
        return

    # 2) Build pipeline: this is your *pre-processing* after VR
    stages = [
        # turn these on/off depending on which preprocessing you’re evaluating
         make_gaussian_blur_stage(3),
        #make_median_blur_stage(3),
        # make_otsu_threshold_stage(),
    ]
    processor = CompositeProcessor(stages)

    # 3) Choose the detector
    strategy = SimpleBlobStrategy()
    # swap here to LoG / DoH / DoG / CCL strategies when testing those

    metrics: List[FrameMetrics] = []
    prev_blobs: List[Blob] = []

    for idx, path in enumerate(files):
        # ---- load 16-bit IR frame ----
        frame16 = cv.imread(str(path), cv.IMREAD_UNCHANGED)
        if frame16 is None:
            print(f"[analyze_recording] Failed to read {path}")
            continue
        if frame16.dtype != np.uint16:
            print(f"[analyze_recording] Expected uint16, got {frame16.dtype} in {path}")
            continue

        # ---- apply VisualRange the same way OrbbecCapture does ----
        ir8 = ir_to_8bit_window(frame16, VR_MIN, VR_MAX)   # Y16 -> window -> 8-bit
        bgr = cv.cvtColor(ir8, cv.COLOR_GRAY2BGR)

        # ---- run through your normal pipeline ----
        data = processor.process(bgr, dt=0.0)

        # ---- detect blobs ----
        out = strategy.process_frame(data, dt=0.0)
        blobs = out.detections
        count = len(blobs)

        fp = max(0, count - EXPECTED_BLOBS)
        fn = max(0, EXPECTED_BLOBS - count)

        jitter_vals = match_blobs(prev_blobs, blobs, MAX_MATCH_DIST)
        jitter_mean = float(np.mean(jitter_vals)) if jitter_vals else -1.0

        metrics.append(FrameMetrics(idx, count, fp, fn, jitter_mean))
        prev_blobs = blobs

    if not metrics:
        print("[analyze_recording] No metrics collected.")
        return

    summarize_metrics(metrics)


def summarize_metrics(metrics: List[FrameMetrics]) -> None:
    counts = np.array([m.count for m in metrics], dtype=np.float32)

    if len(counts) > 1:
        diffs = np.abs(np.diff(counts))
        blobcount_var_mean = float(np.mean(diffs))
        blobcount_var_std = float(np.std(diffs))
        blobcount_var_max = float(np.max(diffs))
    else:
        diffs = np.array([0.0], dtype=np.float32)
        blobcount_var_mean = blobcount_var_std = blobcount_var_max = 0.0

    jitter_vals = np.array(
        [m.jitter_mean for m in metrics if m.jitter_mean >= 0.0],
        dtype=np.float32,
    )
    if jitter_vals.size > 0:
        jitter_mean = float(np.mean(jitter_vals))
        jitter_std = float(np.std(jitter_vals))
        jitter_p95 = float(np.percentile(jitter_vals, 95))
    else:
        jitter_mean = jitter_std = jitter_p95 = 0.0

    total_fp = int(sum(m.fp for m in metrics))
    total_fn = int(sum(m.fn for m in metrics))

    print("==== Metrics summary ====")
    print(f"Frames: {len(metrics)}")
    print(f"Expected blobs per frame: {EXPECTED_BLOBS}")
    print("")
    print("Blobcount variation (|count_t - count_{t-1}|):")
    print(f"  mean: {blobcount_var_mean:.3f}")
    print(f"  std : {blobcount_var_std:.3f}")
    print(f"  max : {blobcount_var_max:.3f}")
    print("")
    print("Jitter (mean displacement of matched blobs, px):")
    print(f"  mean: {jitter_mean:.3f}")
    print(f"  std : {jitter_std:.3f}")
    print(f"  p95 : {jitter_p95:.3f}")
    print("")
    print("False positives / negatives (vs 12 blobs):")
    print(f"  total FP: {total_fp}")
    print(f"  total FN: {total_fn}")


if __name__ == "__main__":
    analyze_strategy()
