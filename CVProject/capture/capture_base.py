from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Tuple
import numpy as np


class Capture(ABC):
    @abstractmethod
    def is_opened(self) -> bool:
        pass

    @abstractmethod
    def read(self) -> Tuple[bool, np.ndarray]:
        pass

    @abstractmethod
    def release(self) -> None:
        pass