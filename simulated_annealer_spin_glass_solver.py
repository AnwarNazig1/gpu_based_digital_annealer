import time
import numpy as np
from typing import Optional
from SpinGlassSolution import SpinGlassSolution
from SpinGlassSolver import SpinGlassSolver
from spin_glass_base import SpinGlassBase

def compute_spin_glass_energy(spins: np.ndarray, couplings: np.ndarray) -> float:
    """
    Computes the energy of a spin glass configuration.
    
    Parameters:
        spins (np.ndarray): Spin configuration.
                             - For fully connected models: shape should be (N,)
                             - For 2D lattice models: shape should be (N, N)
                             Each spin should be either +1 or -1.
                             
        couplings (np.ndarray): Coupling matrix.
                                - For fully connected models: shape should be (N, N)
                                - For 2D lattice models: shape should be (N, N, 2)
                                Couplings should satisfy:
                                - couplings[i, j] = couplings[j, i] for all i, j (for 1D)
                                - couplings[i, j, 0] corresponds to horizontal coupling (i.e., coupling to (i, j+1))
                                - couplings[i, j, 1] corresponds to vertical coupling (i.e., coupling to (i+1, j))
    
    Returns:
        float: Energy of the spin configuration.
    """
    if spins.ndim == 1:
        # Fully connected model (e.g., SK Spin Glass)
        N = len(spins)
        energy = 0.0
        for i in range(N):
            for j in range(i+1, N):  # Only upper triangular to avoid double counting
                energy -= couplings[i, j] * spins[i] * spins[j]
    elif spins.ndim == 2:
        # 2D lattice Spin Glass
        N = spins.shape[0]
        energy = 0.0
        for i in range(N):
            for j in range(N):
                if j < N-1:  # Horizontal coupling to (i, j+1)
                    energy -= couplings[i, j, 0] * spins[i, j] * spins[i, j+1]
                if i < N-1:  # Vertical coupling to (i+1, j)
                    energy -= couplings[i, j, 1] * spins[i, j] * spins[i+1, j]
    else:
        raise ValueError("Spins array must be either 1D or 2D.")
    
    return energy

class SimulatedAnnealerSpinGlassSolver(SpinGlassSolver):
    """
    Simulated Annealing Solver for Spin Glass Problems.
    
    Implements the Simulated Annealing algorithm to find near-optimal solutions 
    for Spin Glass optimization problems.
    """
    
    def __init__(self, 
                 T_initial: float = 100.0, 
                 T_final: float = 0.00001, 
                 alpha: float = 0.99, 
                 num_iterations: int = 2000, 
                 num_runs: int = 2000):
        """
        Initializes the SimulatedAnnealerSpinGlassSolver.
    
        Parameters:
            T_initial (float): Initial temperature.
            T_final (float): Final temperature.
            alpha (float): Cooling rate (0 < alpha < 1).
            num_iterations (int): Number of annealing iterations.
            num_runs (int): Number of independent annealing runs for parallel processing.
        """
        self.T_initial = T_initial
        self.T_final = T_final
        self.alpha = alpha
        self.num_iterations = num_iterations
        self.num_runs = num_runs

    def solve(self, instance: SpinGlassBase) -> SpinGlassSolution:
        """
        Solves the Spin Glass optimization problem using Simulated Annealing.
    
        Parameters:
            instance (SpinGlassBase): An instance of a Spin Glass problem.
    
        Returns:
            SpinGlassSolution: The best spin configuration and its energy.
        """
        start_time = time.time()
    
        # Determine the model type based on spin dimensions
        if instance.spins is not None:
            initial_spins = instance.spins.copy()
            model_type = '2D' if instance.spins.ndim == 2 else '1D'
        else:
            # If no initial spins are provided, initialize randomly
            if instance.couplings.ndim == 3:
                model_type = '2D'
                initial_spins = np.random.choice([-1, 1], size=(instance.N, instance.N))
            elif instance.couplings.ndim == 2:
                model_type = '1D'
                initial_spins = np.random.choice([-1, 1], size=instance.N)
            else:
                raise ValueError("Coupling matrix must be either 2D or 3D.")
    
        # Initialize spins for all runs
        if model_type == '1D':
            # Shape: [num_runs, N]
            spins = np.tile(initial_spins, (self.num_runs, 1))
        elif model_type == '2D':
            # Shape: [num_runs, N, N]
            spins = np.tile(initial_spins, (self.num_runs, 1, 1))
        else:
            raise ValueError("Unsupported model type.")
    
        # Compute initial energies for all runs
        energies = np.array([compute_spin_glass_energy(spins[run], instance.couplings) 
                             for run in range(self.num_runs)])
    
        # Initialize best solutions
        best_energies = energies.copy()
        best_spins = spins.copy()
    
        # Initialize temperatures for all runs
        temperatures = np.full(self.num_runs, self.T_initial)
    
        for iteration in range(self.num_iterations):
            if model_type == '1D':
                # 1D Spin Glass
                
                # Randomly select a spin index to flip for each run
                flip_indices = np.random.randint(0, instance.N, size=self.num_runs)
                
                # Compute energy change ΔE for each selected spin
                s_i = spins[np.arange(self.num_runs), flip_indices]
                
                # Calculate sum_Js = sum over j of J[i,j] * s_j for each run
                sum_Js = np.einsum('ij,ij->i', instance.couplings[flip_indices], spins)
                
                # Energy change ΔE = 2 * s_i * sum_Js
                delta_Es = 2 * s_i * sum_Js
                
            elif model_type == '2D':
                # 2D Spin Glass
                
                # Randomly select a spin position (i,j) to flip for each run
                flip_i = np.random.randint(0, instance.N, size=self.num_runs)
                flip_j = np.random.randint(0, instance.N, size=self.num_runs)
                
                # Compute energy change ΔE for selected spins:
                # ΔE = 2 * s[i,j] * (J[i,j,0] * s[i,j+1] + J[i,j,1] * s[i+1,j])
                s_i = spins[np.arange(self.num_runs), flip_i, flip_j]
                
                # Initialize ΔE with 0
                delta_Es = np.zeros(self.num_runs)
                
                # Horizontal coupling (if not on the last column)
                mask_horiz = flip_j < (instance.N - 1)
                indices_horiz = np.where(mask_horiz)[0]
                if len(indices_horiz) > 0:
                    J_horiz = instance.couplings[flip_i[indices_horiz], flip_j[indices_horiz], 0]
                    s_j_horiz = spins[indices_horiz, flip_i[indices_horiz], flip_j[indices_horiz]+1]
                    delta_Es[indices_horiz] += 2 * s_i[indices_horiz] * J_horiz * s_j_horiz
                
                # Vertical coupling (if not on the last row)
                mask_vert = flip_i < (instance.N - 1)
                indices_vert = np.where(mask_vert)[0]
                if len(indices_vert) > 0:
                    J_vert = instance.couplings[flip_i[indices_vert], flip_j[indices_vert], 1]
                    s_j_vert = spins[indices_vert, flip_i[indices_vert]+1, flip_j[indices_vert]]
                    delta_Es[indices_vert] += 2 * s_i[indices_vert] * J_vert * s_j_vert
            else:
                raise ValueError("Unsupported model type.")
    
            # Determine acceptance of spin flips
            accept = delta_Es < 0  # Always accept if ΔE < 0
            prob_accept = np.exp(-delta_Es / temperatures)  # Calculate acceptance probability for ΔE >= 0
            random_vals = np.random.uniform(0, 1, size=self.num_runs)  # Random values for acceptance decision
            accept = accept | (random_vals < prob_accept)  # Accept if ΔE < 0 or with probability P
    
            # Perform spin flips where accepted
            if model_type == '1D':
                # Create a factor: -1 if accept=True, 1 otherwise
                flip_factors = 1 - 2 * accept
                # Flip the selected spins
                spins[np.arange(self.num_runs), flip_indices] *= flip_factors
            elif model_type == '2D':
                # Identify runs where spin flip is accepted
                indices_flip = np.where(accept)[0]
                if len(indices_flip) > 0:
                    # Flip the spins
                    spins[indices_flip, flip_i[indices_flip], flip_j[indices_flip]] *= -1
            else:
                raise ValueError("Unsupported model type.")
    
            # Update energies where flips were accepted
            energies += accept * delta_Es  # Only update energies for accepted flips
    
            # Update best solutions
            improved = energies < best_energies
            best_energies[improved] = energies[improved]
            best_spins[improved] = spins[improved].copy()
    
            # Update temperatures
            temperatures *= self.alpha
            temperatures = np.maximum(temperatures, self.T_final)
    
            # Optional: Early stopping if all temperatures have reached T_final
            # if np.all(temperatures <= self.T_final):
            #     print(f"Early stopping at iteration {iteration + 1} as all temperatures reached T_final.")
            #     break
    
        # Identify the best run overall
        min_energy_index = np.argmin(best_energies)
        best_energy = best_energies[min_energy_index]
        optimal_spins = best_spins[min_energy_index]
    
        solving_time = time.time() - start_time
    
        # Verification: Compute the energy externally
        verification_energy = compute_spin_glass_energy(optimal_spins, instance.couplings)
        if not np.isclose(best_energy, verification_energy):
            print(f"Warning: Mismatch in energy calculations! Solver reported {best_energy}, but computed {verification_energy}.")
        else:
            # print(f"Energy verification passed: {best_energy} == {verification_energy}")
            pass
        return SpinGlassSolution(optimal_spins, best_energy, solving_time)