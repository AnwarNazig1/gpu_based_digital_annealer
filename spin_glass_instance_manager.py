import json
import numpy as np
from typing import Type
from spin_glass_base import SpinGlassBase, DistributionType
from sk_spin_glass import SKSpinGlass
from two_d_spin_glass import TwoDSpinGlass
from SpinGlassSolution import SpinGlassSolution

# class SpinGlassSolution:
#     def __init__(self, energy: float, spins: np.ndarray, solving_time: float):
#         self.energy = energy
#         self.spins = spins
#         self.solving_time = solving_time

class SpinGlassManager:
    @staticmethod
    def save_instance(instance: SpinGlassBase, filename: str):
        with open(filename, 'w') as file:
            json.dump(instance.to_dict(), file, indent=4)

    @staticmethod
    def load_instance(filename: str, instance_type: Type[SpinGlassBase]) -> SpinGlassBase:
        with open(filename, 'r') as file:
            data = json.load(file)
        
        distribution = DistributionType(data['distribution'])
        
        if instance_type == TwoDSpinGlass:
            instance = TwoDSpinGlass(
                data['N'],
                distribution,
                spins=np.array(data['spins']),
                horizontal_couplings=np.array(data['horizontal_couplings']),
                vertical_couplings=np.array(data['vertical_couplings'])
            )
        elif instance_type == SKSpinGlass:
            instance = SKSpinGlass(
                data['N'],
                distribution,
                spins=np.array(data['spins']),
                couplings=np.array(data['couplings'])
            )
        else:
            raise ValueError(f"Unsupported instance type: {instance_type}")

        instance.id = data['id']
        return instance

    @staticmethod
    def save_solution(instance: SpinGlassBase, solution: SpinGlassSolution, filename: str):
        data = instance.to_dict()
        data['best_energy'] = float(solution.energy)
        data['best_spins'] = solution.spins.tolist()
        data['time_to_solve'] = solution.solving_time
        with open(filename, 'w') as file:
            json.dump(data, file, indent=4)