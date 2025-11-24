from __future__ import annotations
import cv2 as cv
import numpy as np
from typing import Dict, Any, Callable, List, Optional
from dataclasses import dataclass, field


@dataclass
class FrameData:
    original_bgr: np.ndarray
    bgr: np.ndarray
    gray: np.ndarray
    binary: Optional[np.ndarray] = None
    debug: Dict[str, np.ndarray] = field(default_factory=dict)
    meta: Dict[str, Any] = field(default_factory=dict)


class ImageProcessor:
    def process(self, frame_bgr: np.ndarray, dt: float) -> FrameData:
        raise NotImplementedError()


Stage = Callable[[FrameData, float], FrameData]


class CompositeProcessor(ImageProcessor):
    def __init__(self, stages: List[Stage]):
        if not stages:
            raise ValueError("CompositeProcessor requires at least one stage")
        self.stages = stages

    def process(self, frame_bgr: np.ndarray, dt: float) -> FrameData:
        data = FrameData(
            original_bgr=frame_bgr,
            bgr=frame_bgr,
            gray=cv.cvtColor(frame_bgr, cv.COLOR_BGR2GRAY),
        )

        for stage in self.stages:
            data = stage(data, dt)

        return data
