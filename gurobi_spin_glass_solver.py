import time
import numpy as np
from gurobipy import Model, GRB, QuadExpr
from SpinGlassSolution import SpinGlassSolution
from SpinGlassSolver import SpinGlassSolver
from spin_glass_base import SpinGlassBase
from sk_spin_glass import SKSpinGlass
from two_d_spin_glass import TwoDSpinGlass

class GurobiSpinGlassSolver(SpinGlassSolver):
    def __init__(self, time_limit: int = 180, mip_gap: float = 0.01):
        self.time_limit = time_limit
        self.mip_gap = mip_gap
        
    def solve(self, instance: SpinGlassBase) -> SpinGlassSolution:
        if not isinstance(instance, (SKSpinGlass, TwoDSpinGlass)):
            raise ValueError("This solver only supports SK and 2D Lattice Spin Glass instances")
        
        start_time = time.time()

        # Create a new model
        model = Model("SpinGlass")
        model.setParam('OutputFlag', 0)  # Suppress Gurobi output
        model.setParam('TimeLimit', self.time_limit)
        model.setParam('MIPGap', self.mip_gap)
       
        # model.setParam('PreQLinearize', 0)
        # model.setParam('MIQCPMethod', 1)
        # model.setParam('MIPFocus', 1)
        # model.setParam('Heuristics', 0.5)
        # model.setParam('Cuts', 2)
        # model.setParam('NodeMethod', 2)

        # Create variables
        if isinstance(instance, SKSpinGlass):
            x = model.addVars(instance.N, vtype=GRB.BINARY, name="x")
        else:  # TwoDSpinGlass
            x = model.addVars(instance.N, instance.N, vtype=GRB.BINARY, name="x")
        
        model.update()

        # Set objective
        obj = QuadExpr()
        
        if isinstance(instance, SKSpinGlass):
            for i in range(instance.N):
                for j in range(i + 1, instance.N):
                    obj += -instance.couplings[i, j] * (2 * x[i] - 1) * (2 * x[j] - 1)
        else:  # TwoDSpinGlass
            for i in range(instance.N):
                for j in range(instance.N):
                    # Horizontal interactions
                    obj += -instance.horizontal_couplings[i, j] * (2 * x[i, j] - 1) * (2 * x[i, (j + 1) % instance.N] - 1)
                    # Vertical interactions
                    obj += -instance.vertical_couplings[i, j] * (2 * x[i, j] - 1) * (2 * x[(i + 1) % instance.N, j] - 1)
        
        model.setObjective(obj, GRB.MINIMIZE)
        
        # Optimize model
        model.optimize()
        
        # Extract solution
        if isinstance(instance, SKSpinGlass):
            best_spins = np.array([2 * int(x[i].X) - 1 for i in range(instance.N)], dtype=int)
        else:  # TwoDSpinGlass
            best_spins = np.array([[2 * int(x[i, j].X) - 1 for j in range(instance.N)] for i in range(instance.N)], dtype=int)
        
        best_energy = model.objVal
        solving_time = time.time() - start_time
        
        # Check optimization status
        status = model.status
        if status == GRB.OPTIMAL:
            print("Optimal solution found within time limit.")
        elif status == GRB.TIME_LIMIT:
            print(f"Time limit reached. Best solution has a gap of {model.MIPGap * 100:.2f}%.")
        elif status == GRB.SUBOPTIMAL:
            print(f"Suboptimal solution found with a gap of {model.MIPGap * 100:.2f}%.")
        else:
            print(f"Optimization ended with status {status}")
        
        return SpinGlassSolution(best_spins, best_energy, solving_time)