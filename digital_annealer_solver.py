import math
import time
import torch
from SpinGlassSolution import SpinGlassSolution
from SpinGlassSolver import SpinGlassSolver
from spin_glass_base import SpinGlassBase

class DigitalAnnealerSpinGlassSolver(SpinGlassSolver):
    def solve(self, instance: SpinGlassBase) -> SpinGlassSolution:
        start_time = time.time()
        initial_temperature = 100.0
        final_temperature = 0.01
        alpha = 0.95
        num_iterations = 500
        num_runs = 500
        offset_increase_rate = 1
        N = instance.N
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Keep the large matrix and spins on the GPU
        J = torch.tensor(instance.couplings, dtype=torch.float32, device=device)  # coupling matrix on GPU
        best_energy = float('inf')
        best_spins = None  # Initialize best spins as None
        
        # Keep other variables on CPU to avoid frequent transfers
        initial_state = torch.tensor(instance.spins, dtype=torch.float32)  # Keep initial_state on CPU

        # Main loop for multiple runs
        for _ in range(num_runs):
            spins = initial_state.clone()  # Clone initial state on CPU for each run
            temperature = initial_temperature
            Eoffset = 0.0  # Offset initialized on CPU
            
            # Monte Carlo steps
            for step in range(num_iterations):
                if temperature > final_temperature:
                    temperature *= alpha  # Update temperature on CPU

                # Calculate energy changes ΔE for flipping each spin
                spins_gpu = spins.to(device)  # Transfer spins to GPU for matrix operations
                delta_Es = 2 * spins_gpu * torch.matmul(J, spins_gpu)  # Compute energy changes on GPU
                delta_Es = delta_Es.cpu()  # Move result back to CPU
                
                # Adjust ΔE with the offset and decide on flips (on CPU)
                accept_probabilities = torch.exp(-(delta_Es - Eoffset) / temperature)
                accepted_flips = (delta_Es < Eoffset) | (torch.rand(N) < accept_probabilities)

                if torch.any(accepted_flips):
                    # If at least one flip is accepted, randomly select one and apply it
                    accepted_indices = torch.nonzero(accepted_flips).squeeze()
                    if accepted_indices.dim() == 0:
                        # Only one accepted flip, directly use it
                        selected_flip = accepted_indices.item()
                    else:
                        # Multiple accepted flips, choose one randomly
                        selected_flip = accepted_indices[torch.randint(len(accepted_indices), (1,)).item()]
                    spins[selected_flip] *= -1  # Apply the selected flip on CPU
                    Eoffset = 0.0  # Reset the offset on CPU
                else:
                    # Increase the offset if no flip is accepted
                    Eoffset += offset_increase_rate  # Offset updated on CPU

            # Calculate the final energy for this run
            spins_gpu = spins.to(device)  # Transfer spins to GPU for energy calculation
            final_energy = -0.5 * torch.sum(spins_gpu * torch.matmul(J, spins_gpu)).item()  # Fully GPU computation

            if final_energy < best_energy:
                best_energy = final_energy  # Update best energy on CPU
                best_spins = spins.clone()  # Update best spins on CPU

        solving_time = time.time() - start_time

        # Return final result
        final_spins = best_spins.numpy()  # Final spins are already on CPU
        return SpinGlassSolution(final_spins, best_energy, solving_time)

#################

class DigitalAnnealerSpinGlassSolver_OptimizedX2(SpinGlassSolver):
    def __init__(self, initial_temperature = 100.0, final_temperature=0.001, alpha=0.99, num_iterations=2000, num_runs=2000, offset_increase_rate=1.0):
        self.initial_temperature = initial_temperature
        self.final_temperature = final_temperature
        self.alpha = alpha
        self.num_iterations = num_iterations
        self.num_runs = num_runs
        self.offset_increase_rate = offset_increase_rate
        
        
    def solve(self, instance: SpinGlassBase) -> SpinGlassSolution:
        
        initial_temperature = self.initial_temperature
        final_temperature = self.final_temperature
        alpha = self.alpha
        num_iterations = self.num_iterations 
        num_runs = self.num_runs
        offset_increase_rate = self.offset_increase_rate
        N = instance.N
        
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Transfer couplings and initial spins to GPU
        J = torch.tensor(instance.couplings, dtype=torch.float32, device=device)
        initial_state = torch.tensor(instance.spins, dtype=torch.float32, device=device)

        # Initialize parameters on GPU
        temperature = torch.full((num_runs,), initial_temperature, device=device)
        final_temperature_tensor = torch.full((num_runs,), final_temperature, device=device)
        alpha_tensor = torch.full((num_runs,), alpha, device=device)
        Eoffset = torch.zeros((num_runs,), dtype=torch.float32, device=device)

        # Initialize spins for all runs
        spins = initial_state.unsqueeze(0).repeat(num_runs, 1).to(device)  # Shape: [num_runs, N]
        
        # Generate random initial states for all runs
        spins = torch.randint(0, 2, (num_runs, instance.N), dtype=torch.float32, device=device)  # Randomly initializes spins to 0 or 1
        spins = 2 * spins - 1  # Convert spins from {0,1} to {-1,+1}
        best_energies = torch.full((num_runs,), float('inf'), device=device)
        best_spins = spins.clone()

        start_time = time.time()

        for step in range(num_iterations):
            # Update temperature
            above_final = temperature > final_temperature_tensor
            temperature = temperature * alpha_tensor * above_final + final_temperature_tensor * (~above_final)

            # Calculate ΔE for all spins in all runs
            energy_intermediate = torch.matmul(spins, J)  # Shape: [num_runs, N]
            delta_Es = 2 * spins * energy_intermediate  # Shape: [num_runs, N]

            # Compute acceptance probabilities
            accept_probs = torch.exp(-(delta_Es - Eoffset.unsqueeze(1)) / temperature.unsqueeze(1))
            
            # Generate random numbers on GPU
            rand_vals = torch.rand(accept_probs.shape, device=device)
            
            # Determine which spins to flip
            acceptance = (delta_Es < Eoffset.unsqueeze(1)) | (rand_vals < accept_probs)
            
            # Randomize spins to flip to avoid bias
            random_flip = torch.rand(acceptance.shape, device=device) < 0.5
            flip_mask = acceptance & random_flip

            # Apply spin flips
            spins = spins * (1 - 2 * flip_mask.float())

            # Update Eoffset
            any_flips = flip_mask.any(dim=1)
            Eoffset = torch.where(any_flips, torch.tensor(0.0, device=device), Eoffset + offset_increase_rate)

            # Update best energies and spins
            energies = -0.5 * torch.sum(spins * torch.matmul(J, spins.t()).t(), dim=1)
            improved = energies < best_energies
            best_energies = torch.where(improved, energies, best_energies)
            best_spins = torch.where(improved.unsqueeze(1), spins, best_spins)

        # After all iterations, find the best solution across all runs  
        min_energy, min_idx = torch.min(best_energies, dim=0)
        optimal_spins = best_spins[min_idx].cpu().numpy()
        best_energy = min_energy.item()
        solving_time = time.time() - start_time

        return SpinGlassSolution(optimal_spins, best_energy, solving_time)

#################


class DigitalAnnealerSpinGlassSolver_OptimizedX(SpinGlassSolver):
    def __init__(self, initial_temperature = 100.0, final_temperature=0.001, alpha=0.99, num_iterations=2000, num_runs=2000, offset_increase_rate=1.0):
        self.initial_temperature = initial_temperature
        self.final_temperature = final_temperature
        self.alpha = alpha
        self.num_iterations = num_iterations
        self.num_runs = num_runs
        self.offset_increase_rate = offset_increase_rate
        
        
    def solve(self, instance: SpinGlassBase) -> SpinGlassSolution:
        
        initial_temperature = self.initial_temperature
        final_temperature = self.final_temperature
        alpha = self.alpha
        num_iterations = self.num_iterations 
        num_runs = self.num_runs
        offset_increase_rate = self.offset_increase_rate
        N = instance.N
        
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Transfer couplings and initial spins to GPU
        J = torch.tensor(instance.couplings, dtype=torch.float32, device=device)
        initial_state = torch.tensor(instance.spins, dtype=torch.float32, device=device)

        # Initialize parameters on GPU
        temperature = torch.full((num_runs,), initial_temperature, device=device)
        final_temperature_tensor = torch.full((num_runs,), final_temperature, device=device)
        alpha_tensor = torch.full((num_runs,), alpha, device=device)
        Eoffset = torch.zeros((num_runs,), dtype=torch.float32, device=device)

        # Initialize spins for all runs
        spins = initial_state.unsqueeze(0).repeat(num_runs, 1).to(device)  # Shape: [num_runs, N]
        best_energies = torch.full((num_runs,), float('inf'), device=device)
        best_spins = spins.clone()

        start_time = time.time()

        for step in range(num_iterations):
            single_run_time_start = time.time()
            # Update temperature
            above_final = temperature > final_temperature_tensor
            temperature = temperature * alpha_tensor * above_final + final_temperature_tensor * (~above_final)

            # Calculate ΔE for all spins in all runs
            energy_intermediate = torch.matmul(spins, J)  # Shape: [num_runs, N]
            delta_Es = 2 * spins * energy_intermediate  # Shape: [num_runs, N]

            # Compute acceptance probabilities
            accept_probs = torch.exp(-(delta_Es - Eoffset.unsqueeze(1)) / temperature.unsqueeze(1))
            
            # Generate random numbers on GPU
            rand_vals = torch.rand(accept_probs.shape, device=device)
            
            # Determine which spins to flip
            acceptance = (delta_Es < Eoffset.unsqueeze(1)) | (rand_vals < accept_probs)
            
            # Randomize spins to flip to avoid bias
            random_flip = torch.rand(acceptance.shape, device=device) < 0.5
            flip_mask = acceptance & random_flip

            # Apply spin flips
            spins = spins * (1 - 2 * flip_mask.float())

            # Update Eoffset
            any_flips = flip_mask.any(dim=1)
            Eoffset = torch.where(any_flips, torch.tensor(0.0, device=device), Eoffset + offset_increase_rate)

            # Update best energies and spins
            energies = -0.5 * torch.sum(spins * torch.matmul(J, spins.t()).t(), dim=1)
            improved = energies < best_energies
            best_energies = torch.where(improved, energies, best_energies)
            best_spins = torch.where(improved.unsqueeze(1), spins, best_spins)
            
            single_run_time_end = time.time() - single_run_time_start
        # After all iterations, find the best solution across all runs  
        min_energy, min_idx = torch.min(best_energies, dim=0)
        optimal_spins = best_spins[min_idx].cpu().numpy()
        best_energy = min_energy.item()
        solving_time = time.time() - start_time

        return SpinGlassSolution(optimal_spins, best_energy, solving_time)

#################

class DigitalAnnealerSpinGlassSolver_OptimizedX_test(SpinGlassSolver):
    def __init__(self, initial_temperature=100.0, final_temperature=0.001, alpha=0.99,
                 num_iterations=1000, num_runs=1000, offset_increase_rate=1.0, refrence_energy=0.0):
        self.initial_temperature = initial_temperature
        self.final_temperature = final_temperature
        self.alpha = alpha
        self.num_iterations = num_iterations
        self.num_runs = num_runs
        self.offset_increase_rate = offset_increase_rate
        self.refrence_energy = refrence_energy
    def solve(self, instance: SpinGlassBase) -> SpinGlassSolution:
        import csv  # Import CSV module within the method

        # Extract parameters
        initial_temperature = self.initial_temperature
        final_temperature = self.final_temperature
        alpha = self.alpha
        num_iterations = self.num_iterations
        num_runs = self.num_runs
        offset_increase_rate = self.offset_increase_rate
        N = instance.N

        # Set device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Transfer couplings and initial spins to GPU
        J = torch.tensor(instance.couplings, dtype=torch.float32, device=device)
        initial_state = torch.tensor(instance.spins, dtype=torch.float32, device=device)

        # Initialize parameters on GPU
        temperature = torch.full((num_runs,), initial_temperature, device=device)
        final_temperature_tensor = torch.full((num_runs,), final_temperature, device=device)
        alpha_tensor = torch.full((num_runs,), alpha, device=device)
        Eoffset = torch.zeros((num_runs,), dtype=torch.float32, device=device)

        # Initialize spins for all runs
        spins = initial_state.unsqueeze(0).repeat(num_runs, 1).to(device)  # Shape: [num_runs, N]
        best_energies = torch.full((num_runs,), float('inf'), device=device)
        best_spins = spins.clone()

        # Extract instance-specific information
        instance_id = instance.id  # Ensure SpinGlassBase has this attribute
        reference_energy =  self.refrence_energy   # Ensure SpinGlassBase has this attribute

        # Define CSV filename based on instance_id
        csv_filename = f"tts_64_bimodal_gpu.csv"

        # Open CSV file and write headers
        with open(csv_filename, mode='w', newline='') as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(['instance_id', 'iteration', 't', 'energy', 'reference_energy', 'success'])

            start_time = time.time()

            for step in range(num_iterations):
                single_run_time_start = time.time()

                # Update temperature
                above_final = temperature > final_temperature_tensor
                temperature = temperature * alpha_tensor * above_final + final_temperature_tensor * (~above_final)

                # Calculate ΔE for all spins in all runs
                energy_intermediate = torch.matmul(spins, J)  # Shape: [num_runs, N]
                delta_Es = 2 * spins * energy_intermediate  # Shape: [num_runs, N]

                # Compute acceptance probabilities
                accept_probs = torch.exp(-(delta_Es - Eoffset.unsqueeze(1)) / temperature.unsqueeze(1))

                # Generate random numbers on GPU
                rand_vals = torch.rand(accept_probs.shape, device=device)

                # Determine which spins to flip
                acceptance = (delta_Es < Eoffset.unsqueeze(1)) | (rand_vals < accept_probs)

                # Randomize spins to flip to avoid bias
                random_flip = torch.rand(acceptance.shape, device=device) < 0.5
                flip_mask = acceptance & random_flip

                # Apply spin flips
                spins = spins * (1 - 2 * flip_mask.float())

                # Update Eoffset
                any_flips = flip_mask.any(dim=1)
                Eoffset = torch.where(any_flips, torch.tensor(0.0, device=device), Eoffset + offset_increase_rate)

                # Update best energies and spins
                energies = -0.5 * torch.sum(spins * torch.matmul(J, spins.t()).t(), dim=1)
                improved = energies < best_energies
                best_energies = torch.where(improved, energies, best_energies)
                best_spins = torch.where(improved.unsqueeze(1), spins, best_spins)

                # Calculate time taken for this iteration
                single_run_time_end = time.time() - single_run_time_start

                # Find current best energy
                current_min_energy, _ = torch.min(best_energies, dim=0)

                # Calculate number of successes (runs achieving energy <= reference_energy)
                success_count = (best_energies <= reference_energy).sum().item()

                # Write data to CSV
                writer.writerow([instance_id, step, single_run_time_end, current_min_energy.item(),
                                 reference_energy, success_count])

        # After all iterations, find the best solution across all runs
        min_energy, min_idx = torch.min(best_energies, dim=0)
        optimal_spins = best_spins[min_idx].cpu().numpy()
        best_energy = min_energy.item()
        solving_time = time.time() - start_time

        return SpinGlassSolution(optimal_spins, best_energy, solving_time)
#################
class DigitalAnnealerSpinGlassSolverX2(SpinGlassSolver):
    def solve(self, instance: SpinGlassBase) -> SpinGlassSolution:
        start_time = time.time()
        initial_temperature = 100.0
        final_temperature = 0.01
        alpha = 0.95
        num_iterations = 1000
        num_runs = 1000
        offset_increase_rate = 1.0
        N = instance.N
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Transfer couplings and initial spins to GPU
        J = torch.tensor(instance.couplings, dtype=torch.float32, device=device)
        initial_state = torch.tensor(instance.spins, dtype=torch.float32, device=device)

        # Initialize parameters on GPU
        temperature = torch.full((num_runs,), initial_temperature, device=device)
        final_temperature_tensor = torch.full((num_runs,), final_temperature, device=device)
        alpha_tensor = torch.full((num_runs,), alpha, device=device)
        Eoffset = torch.zeros((num_runs,), dtype=torch.float32, device=device)

        # Initialize spins for all runs
        spins = initial_state.unsqueeze(0).repeat(num_runs, 1).to(device)  # Shape: [num_runs, N]
        best_energies = torch.full((num_runs,), float('inf'), device=device)
        best_spins = spins.clone()

        for step in range(num_iterations):
            # Update temperature: temperature = temperature * alpha if temp > final_temperature else final_temperature
            mask_temp = temperature > final_temperature_tensor
            temperature = temperature * alpha * mask_temp.float() + final_temperature_tensor * (~mask_temp).float()

            # Calculate ΔE for all spins in all runs
            energy_intermediate = torch.matmul(spins, J)  # Shape: [num_runs, N]
            delta_Es = 2 * spins * energy_intermediate  # Shape: [num_runs, N]

            # Compute acceptance probabilities
            # Broadcast Eoffset and temperature to match delta_Es shape
            Eoffset_broadcast = Eoffset.unsqueeze(1)  # Shape: [num_runs, 1]
            temperature_broadcast = temperature.unsqueeze(1)  # Shape: [num_runs, 1]
            accept_probs = torch.exp(-(delta_Es - Eoffset_broadcast) / temperature_broadcast)

            # Generate random numbers on GPU
            rand_vals = torch.rand(accept_probs.shape, device=device)

            # Determine which spins to flip: either delta_E < Eoffset or rand < accept_prob
            acceptance = (delta_Es < Eoffset_broadcast) | (rand_vals < accept_probs)

            # Randomize flips to avoid deterministic patterns
            random_flip = torch.rand(acceptance.shape, device=device) < 0.5
            flip_mask = acceptance & random_flip  # Shape: [num_runs, N]

            # Apply spin flips: spins = spins * (1 - 2 * flip_mask)
            spins = spins * (1 - 2 * flip_mask.float())

            # Update Eoffset: reset to 0 where any spin was flipped, else increase by offset_increase_rate
            any_flips = flip_mask.any(dim=1)  # Shape: [num_runs]
            Eoffset = torch.where(any_flips, torch.zeros_like(Eoffset), Eoffset + offset_increase_rate)

            # Calculate energies: E = -0.5 * spins * J * spins
            energies = -0.5 * torch.sum(spins * torch.matmul(spins, J), dim=1)  # Shape: [num_runs]

            # Update best energies and spins
            improved = energies < best_energies
            best_energies = torch.where(improved, energies, best_energies)
            best_spins = torch.where(improved.unsqueeze(1), spins, best_spins)

        # After all iterations, find the best solution across all runs
        min_energy, min_idx = torch.min(best_energies, dim=0)
        optimal_spins = best_spins[min_idx].cpu().numpy()
        best_energy = min_energy.item()
        solving_time = time.time() - start_time

        return SpinGlassSolution(optimal_spins, best_energy, solving_time)