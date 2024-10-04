from spin_glass_base import SpinGlassBase, DistributionType
import numpy as np
from typing import Dict, Any, Optional

class TwoDSpinGlass(SpinGlassBase):
    def __init__(self, N: int, distribution: DistributionType = DistributionType.BIMODAL,
                 spins: Optional[np.ndarray] = None, horizontal_couplings: Optional[np.ndarray] = None,
                 vertical_couplings: Optional[np.ndarray] = None):
        self.horizontal_couplings = horizontal_couplings
        self.vertical_couplings = vertical_couplings
        super().__init__(N, distribution, spins)

    def initialize_spins(self) -> np.ndarray:
        return np.random.choice([-1, 1], size=(self.N, self.N))

    def generate_couplings(self) -> Dict[str, np.ndarray]:
        if self.horizontal_couplings is None:
            self.horizontal_couplings = self._generate_distribution((self.N, self.N))
        if self.vertical_couplings is None:
            self.vertical_couplings = self._generate_distribution((self.N, self.N))
        return {'horizontal': self.horizontal_couplings, 'vertical': self.vertical_couplings}

    def calculate_energy(self) -> float:
        horizontal_energy = np.sum(self.horizontal_couplings * self.spins * np.roll(self.spins, shift=-1, axis=1))
        vertical_energy = np.sum(self.vertical_couplings * self.spins * np.roll(self.spins, shift=-1, axis=0))
        return -(horizontal_energy + vertical_energy)

    def to_dict(self) -> Dict[str, Any]:
        return {
            'N': self.N,
            'id': self.id,
            'problem_type': '2D Spin-Glass',
            'distribution': self.distribution.value,
            'spins': self.spins.tolist(),
            'horizontal_couplings': self.horizontal_couplings.tolist(),
            'vertical_couplings': self.vertical_couplings.tolist()
        }