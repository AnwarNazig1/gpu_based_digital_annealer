from spin_glass_base import SpinGlassBase, DistributionType
import numpy as np
from typing import Dict, Any, Optional

class SKSpinGlass(SpinGlassBase):
    def __init__(self, N: int, distribution: DistributionType = DistributionType.BIMODAL,
                 spins: Optional[np.ndarray] = None, couplings: Optional[np.ndarray] = None):
        self.couplings = couplings
        super().__init__(N, distribution, spins)

    def initialize_spins(self) -> np.ndarray:
        return np.random.choice([-1, 1], size=self.N)

    def generate_couplings(self) -> np.ndarray:
        if self.couplings is not None:
            return self.couplings

        couplings = self._generate_distribution((self.N, self.N))
        couplings = np.triu(couplings, 1)
        couplings += couplings.T
        np.fill_diagonal(couplings, 0)
        return couplings

    def calculate_energy(self) -> float:
        return -np.sum(np.triu(self.couplings, 1) * np.outer(self.spins, self.spins))

    def to_dict(self) -> Dict[str, Any]:
        return {
            'N': self.N,
            'id': self.id,
            'problem_type': 'SK Spin-Glass',
            'distribution': self.distribution.value,
            'spins': self.spins.tolist(),
            'couplings': self.couplings.tolist()
        }