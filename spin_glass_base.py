from abc import ABC, abstractmethod
import numpy as np
import uuid
from typing import Dict, Any, Optional
from enum import Enum

class DistributionType(Enum):
    BIMODAL = "bimodal"
    GAUSSIAN = "gaussian"

class SpinGlassBase(ABC):
    def __init__(self, N: int, distribution: DistributionType = DistributionType.BIMODAL,
                 spins: Optional[np.ndarray] = None):
        self.N = N
        self.id = str(uuid.uuid4())
        self.distribution = distribution
        self.spins = spins if spins is not None else self.initialize_spins()
        self.couplings = self.generate_couplings()

    @abstractmethod
    def initialize_spins(self) -> np.ndarray:
        pass

    @abstractmethod
    def generate_couplings(self) -> np.ndarray:
        pass

    @abstractmethod
    def calculate_energy(self) -> float:
        pass

    @abstractmethod
    def to_dict(self) -> Dict[str, Any]:
        pass

    def _generate_distribution(self, size: tuple) -> np.ndarray:
        if self.distribution == DistributionType.BIMODAL:
            return np.random.choice([-1, 1], size=size)
        elif self.distribution == DistributionType.GAUSSIAN:
            return np.random.normal(0, 1, size=size)
        else:
            raise ValueError(f"Invalid distribution type: {self.distribution}")
