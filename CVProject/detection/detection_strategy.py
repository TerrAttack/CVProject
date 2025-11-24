from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Any, Optional, List
import numpy as np

from processing.image_processor import FrameData


@dataclass
class Blob:
    x: float
    y: float
    size: float
    area: float
    response: float


@dataclass
class StrategyOutput:
    vis: Optional[np.ndarray]
    debug: Dict[str, np.ndarray]
    detections: Any


class DetectionStrategy:
    def process_frame(self, frame: FrameData, dt: float) -> StrategyOutput:
        raise NotImplementedError()