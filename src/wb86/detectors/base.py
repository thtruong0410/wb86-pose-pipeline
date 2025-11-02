from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Tuple
import numpy as np


class BaseDetector(ABC):
    @abstractmethod
    def detect(self, img_bgr: np.ndarray) -> List[Tuple[float, float, float, float, float]]:
        """Return list of (x1,y1,x2,y2,score) for persons."""
        raise NotImplementedError

