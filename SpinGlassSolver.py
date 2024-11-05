from abc import ABC, abstractmethod
from SpinGlassSolution import SpinGlassSolution
from spin_glass_base import SpinGlassBase

class SpinGlassSolver(ABC):
    @abstractmethod
    def solve(self, instance: SpinGlassBase) -> SpinGlassSolution:
        pass