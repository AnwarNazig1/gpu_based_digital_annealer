# spin_glass_generator.py
from typing import List, Type
import numpy as np
from spin_glass_base import SpinGlassBase, DistributionType

class SpinGlassGenerator:
    @staticmethod
    def generate_instance(N: int, instance_type: Type[SpinGlassBase], 
                          distribution: DistributionType = DistributionType.BIMODAL, 
                          spins: np.ndarray = None) -> SpinGlassBase:
        return instance_type(N=N, distribution=distribution, spins=spins)
    
    @staticmethod
    def generate_instances(num_instances: int, N: int, instance_type: Type[SpinGlassBase], 
                           distribution: DistributionType = DistributionType.BIMODAL) -> List[SpinGlassBase]:
        return [SpinGlassGenerator.generate_instance(N, instance_type, distribution) 
                for _ in range(num_instances)]