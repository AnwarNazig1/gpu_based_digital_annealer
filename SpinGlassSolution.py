import numpy as np

class SpinGlassSolution:
    def __init__(self, spins: np.ndarray, energy: float,  solving_time: float):
        self.spins = spins
        self.energy = energy
        self.solving_time = solving_time