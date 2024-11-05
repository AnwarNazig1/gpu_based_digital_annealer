import os
import logging
from simulated_annealer_spin_glass_solver import SimulatedAnnealerSpinGlassSolver
from spin_glass_base import DistributionType
from digital_annealer_solver import DigitalAnnealerSpinGlassSolver, DigitalAnnealerSpinGlassSolver_OptimizedX,DigitalAnnealerSpinGlassSolver_OptimizedX2, DigitalAnnealerSpinGlassSolver_OptimizedX_test
from sk_spin_glass import SKSpinGlass
from two_d_spin_glass import TwoDSpinGlass
from gurobi_spin_glass_solver import GurobiSpinGlassSolver
from spin_glass_instance_generator import SpinGlassGenerator
from spin_glass_instance_manager import SpinGlassManager
# from gurobipy import Model, GRB, QuadExpr
import optuna
import csv
import matplotlib.pyplot as plt
import glob
import json
import pandas as pd
from typing import Dict, Any, Optional, List

def load_instance_files(n_value, distribution):
 
    # Find all files matching the pattern *_N_{n_value}_*
    pattern = f'solved_instances_experiment/SKSpinGlass_{distribution}_N_{n_value}_*_solution.json'
    instance_files = glob.glob(pattern)
    return instance_files

# Function to compute statistics for each N
def compute_statistics_and_plot_ordered(csv_pattern):
    # Dictionary to store results
    stats_by_N = {}

    # Loop through all files matching the pattern and sort them by N
    for csv_file in sorted(glob.glob(csv_pattern), key=lambda x: int(x.split('_')[3])):  # Sorting based on N
        # Load the CSV file
        df = pd.read_csv(csv_file)

        # Extract N from the file name
        N = int(csv_file.split('_')[3])  # Assuming the file name has N in the format SKSpinGlass_bimodal_N_<value>_gpu_annealer.csv

        # Compute mean and variance
        averages = df[['gurobi_energy', 'gurobi_time', 'annealer_energy', 'annealer_time']].mean()
        variances = df[['gurobi_energy', 'gurobi_time', 'annealer_energy', 'annealer_time']].var()

        # Store in dictionary
        stats_by_N[N] = {
            'mean_gurobi_energy': averages['gurobi_energy'],
            'mean_gurobi_time': averages['gurobi_time'],
            'mean_annealer_energy': averages['annealer_energy'],
            'mean_annealer_time': averages['annealer_time'],
            'var_gurobi_energy': variances['gurobi_energy'],
            'var_gurobi_time': variances['gurobi_time'],
            'var_annealer_energy': variances['annealer_energy'],
            'var_annealer_time': variances['annealer_time'],
            'efficiency_gurobi': averages['gurobi_energy'] / averages['gurobi_time'],
            'efficiency_annealer': averages['annealer_energy'] / averages['annealer_time'],
        }

    return stats_by_N
# Function to plot N vs time, energy, and efficiency
def plot_results_with_logscale(stats_by_N):
    # Convert dictionary to DataFrame
    df_stats = pd.DataFrame.from_dict(stats_by_N, orient='index')

    # Plot N vs average time (log scale)
    plt.figure(figsize=(10, 6))
    if 'mean_gurobi_time' in df_stats.columns:
        plt.plot(df_stats.index, df_stats['mean_gurobi_time'], label='Gurobi Time', marker='o')
    if 'mean_annealer_time' in df_stats.columns:
        plt.plot(df_stats.index, df_stats['mean_annealer_time'], label='Annealer Time', marker='o')
    plt.xlabel('N')
    plt.ylabel('Average Time (s)')
    plt.yscale('log')
    plt.title('N vs Average Time (Log Scale)')
    plt.legend()
    plt.grid(True)
    plt.show()

    # Plot N vs average energy (log scale)
    plt.figure(figsize=(10, 6))
    if 'mean_gurobi_energy' in df_stats.columns:
        plt.plot(df_stats.index, df_stats['mean_gurobi_energy'], label='Gurobi Energy', marker='o')
    if 'mean_annealer_energy' in df_stats.columns:
        plt.plot(df_stats.index, df_stats['mean_annealer_energy'], label='Annealer Energy', marker='o')
    plt.xlabel('N')
    plt.ylabel('Average Energy')
    plt.title('N vs Average Energy')
    plt.legend()
    plt.grid(True)
    plt.show()



def run_gpu_annealer(
    n_value: int = 64,
    distribution: str = "gaussian",
    optimized_params: Optional[Dict[str, Any]] = None,
    instance_type: Any = SKSpinGlass,
    num_iterations_per_instance: int = 20,
    output_csv_path: Optional[str] = None,
    skip_existing: bool = True
) -> None:
    """
    Runs the GPU annealer on SpinGlass instances and logs the results to a CSV file.

    Args:
        n_value (int): The size parameter for the SpinGlass instances.
        distribution (str): The distribution type for the instances (e.g., "gaussian").
        optimized_params (Dict[str, Any], optional): Parameters for the annealer. Defaults to predefined values.
        instance_type (Any): The type/class of the SpinGlass instance. Defaults to SKSpinGlass.
        num_iterations_per_instance (int): Number of annealing iterations per instance. Defaults to 20.
        output_csv_path (str, optional): Path to the output CSV file. Defaults to a formatted string based on parameters.
        skip_existing (bool): If True, skips writing headers if the CSV file already exists. Defaults to True.

    Raises:
        FileNotFoundError: If the instance files cannot be found.
        Exception: For any other exceptions that occur during processing.
    """
    
    import os

    # Set default optimized parameters if not provided
    if optimized_params is None:
        optimized_params = {
            'initial_temperature': 100,
            'final_temperature': 0.001,
            'alpha': 0.8,
            'num_iterations': 60,
            'offset_increase_rate': 10,
            'num_runs': 350
            # Add other default parameters as required
        }
    
    # Set default output CSV path if not provided
    if output_csv_path is None:
        output_csv_path = f"SKSpinGlass_{distribution}_N_{n_value}_gpu_annealer.csv"
    
    # Define CSV fieldnames
    fieldnames = [
        'instance_id',
        'N',
        'iteration_number',
        'gurobi_energy',
        'gurobi_time',
        'annealer_energy',
        'annealer_time',
        'energy_difference'
    ]
    
    try:
        # Load instance files based on n_value and distribution
        instances_files: List[str] = load_instance_files(n_value, distribution)
        if not instances_files:
            raise FileNotFoundError(f"No instance files found for N={n_value} and distribution='{distribution}'.")
        
        # Open the CSV file once and write headers if needed
        write_headers = not (skip_existing and os.path.exists(output_csv_path))
        with open(output_csv_path, mode='a', newline='') as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
            if write_headers:
                writer.writeheader()
            
            # Iterate over each instance file
            for filename in instances_files:
                try:
                    # Load the SpinGlass instance
                    instance = SpinGlassManager.load_instance(filename, instance_type)
                    
                    # Load instance metadata
                    with open(filename, 'r') as file:
                        instance_json = json.load(file)
                    
                    instance_id = instance_json.get('id', 'unknown_id')
                    gurobi_energy = instance_json.get('best_energy', None)
                    gurobi_time = instance_json.get('time_to_solve', None)
                    
                    if gurobi_energy is None or gurobi_time is None:
                        print(f"Warning: Missing Gurobi data in {filename}. Skipping this instance.")
                        continue
                    
                    # Perform multiple annealing iterations
                    for i in range(num_iterations_per_instance):
                        # Initialize the annealer with optimized parameters
                        digital_solver_optimized = DigitalAnnealerSpinGlassSolver_OptimizedX(
                            initial_temperature=optimized_params['initial_temperature'],
                            final_temperature=optimized_params['final_temperature'],
                            alpha=optimized_params['alpha'],
                            num_iterations=optimized_params['num_iterations'],
                            offset_increase_rate=optimized_params['offset_increase_rate'],
                            num_runs=optimized_params['num_runs']
                        )
                        
                        # Solve the instance
                        solution = digital_solver_optimized.solve(instance)
                        print(f"Solved instance {filename} on iteration {i+1}/{num_iterations_per_instance}.")
                        
                        gpu_annealer_energy = solution.energy
                        gpu_annealer_time = solution.solving_time
                        energy_difference = gpu_annealer_energy - gurobi_energy
                        
                        # Write the results to the CSV file
                        writer.writerow({
                            'instance_id': instance_id,
                            'N': n_value,
                            'iteration_number': i,
                            'gurobi_energy': gurobi_energy,
                            'gurobi_time': gurobi_time,
                            'annealer_energy': gpu_annealer_energy,
                            'annealer_time': gpu_annealer_time,
                            'energy_difference': energy_difference
                        })
                
                except Exception as e:
                    print(f"Error processing file {filename}: {e}")
                    continue
    
    except Exception as e:
        print(f"An error occurred during the annealing process: {e}")
        
        
def main():
    
    optimized_params = {
            'initial_temperature': 100,
            'final_temperature': 0.001,
            'alpha': 0.8,
            'num_iterations': 60,
            'offset_increase_rate': 10,
            'num_runs': 350
            # Add other default parameters as required
        }
    run_gpu_annealer(
    n_value=64,
    distribution="gaussian",
    optimized_params=optimized_params,
    )
    
    
    optimized_params = {
            'initial_temperature': 100,
            'final_temperature': 0.001,
            'alpha': 0.93,
            'num_iterations': 150,
            'offset_increase_rate': 10,
            'num_runs': 400
            # Add other default parameters as required
        }
    run_gpu_annealer(
    n_value=144,
    distribution="gaussian",
    optimized_params=optimized_params,
    )
    
    optimized_params = {
            'initial_temperature': 100,
            'final_temperature': 0.001,
            'alpha': 0.98,
            'num_iterations': 500,
            'offset_increase_rate': 5,
            'num_runs': 700
            # Add other default parameters as required
        }
    run_gpu_annealer(
    n_value=256,
    distribution="gaussian",
    optimized_params=optimized_params,
    )
    
    
    optimized_params = {
            'initial_temperature': 100,
            'final_temperature': 0.001,
            'alpha': 0.98,
            'num_iterations': 700,
            'offset_increase_rate': 5,
            'num_runs': 1000
            # Add other default parameters as required
        }
    run_gpu_annealer(
    n_value=400,
    distribution="gaussian",
    optimized_params=optimized_params,
    )
    
    
    
    optimized_params = {
            'initial_temperature': 150,
            'final_temperature': 0.001,
            'alpha': 0.98,
            'num_iterations': 800,
            'offset_increase_rate': 20,
            'num_runs': 1500
            # Add other default parameters as required
        }
    run_gpu_annealer(
    n_value=576,
    distribution="gaussian",
    optimized_params=optimized_params,
    )
    
    
    
    optimized_params = {
            'initial_temperature': 150,
            'final_temperature': 0.001,
            'alpha': 0.98,
            'num_iterations': 1200,
            'offset_increase_rate': 50,
            'num_runs': 1600
            # Add other default parameters as required
        }
    run_gpu_annealer(
    n_value=676,
    distribution="gaussian",
    optimized_params=optimized_params,
    )
    
    
    optimized_params = {
            'initial_temperature': 200,
            'final_temperature': 0.001,
            'alpha': 0.99,
            'num_iterations': 2100,
            'offset_increase_rate': 50,
            'num_runs': 1400
            # Add other default parameters as required
        }
    run_gpu_annealer(
    n_value=900,
    distribution="gaussian",
    optimized_params=optimized_params,
    )
    
    optimized_params = {
            'initial_temperature': 200,
            'final_temperature': 0.001,
            'alpha': 0.99,
            'num_iterations': 2100,
            'offset_increase_rate': 50,
            'num_runs': 2100
            # Add other default parameters as required
        }
    run_gpu_annealer(
    n_value=1024,
    distribution="gaussian",
    optimized_params=optimized_params,
    )
    
    
    
    # csv_pattern = "SKSpinGlass_bimodal_N_*_gpu_annealer.csv"
    # stats_by_N = compute_statistics_and_plot_ordered(csv_pattern)
    # plot_results_with_logscale(stats_by_N)
    # return 
    
    ##################################
    # n_value = 64
    # distribution ="gaussian"
    # instance_type = SKSpinGlass
    # instances_files = load_instance_files(n_value, distribution)
    # optimized_params = {
    #     'initial_temperature': 100,
    #     'final_temperature':0.001,
    #     'alpha': 0.8,
    #     'num_iterations': 60,
    #     'offset_increase_rate': 10,
    #     'num_runs': 350
    #     # Add other parameters as required
    # }
    
    # for filename in instances_files:
    #     with open(filename, 'r') as file:
    #         instance_json = json.load(file)
        
    #     reference_energy =  instance_json['best_energy']
    #     instance = SpinGlassManager.load_instance(filename, instance_type)
    #     digital_solver_optimized = DigitalAnnealerSpinGlassSolver_OptimizedX_test(initial_temperature=optimized_params['initial_temperature'], 
    #                                                                                     final_temperature=optimized_params['final_temperature'], 
    #                                                                                     alpha=optimized_params['alpha'], 
    #                                                                                     num_iterations=optimized_params['num_iterations'], 
    #                                                                                     offset_increase_rate=optimized_params['offset_increase_rate'],
    #                                                                                     num_runs=optimized_params['num_runs'],
    #                                                                                     refrence_energy=reference_energy
    #                                                                                     )
    #     solution = digital_solver_optimized.solve(instance)
    #     print('filename',filename)
    #     break
    # return
    ##############################àà
    # n_value = 64
    # distribution ="gaussian"
    # instance_type = SKSpinGlass
    # instances_files = load_instance_files(n_value, distribution)
    # optimized_params = {
    #     'initial_temperature': 100,
    #     'final_temperature':0.001,
    #     'alpha': 0.8,
    #     'num_iterations': 60,
    #     'offset_increase_rate': 10,
    #     'num_runs': 350
    #     # Add other parameters as required
    # }

    
    # output_csv_path = f"SKSpinGlass_{distribution}_N_{n_value}_gpu_annealer.csv"
    # with open(output_csv_path, mode='a', newline='') as csv_file:
    #     fieldnames = ['instance_id','N','iteration_number','gurobi_energy','gurobi_time', 'annealer_energy','annealer_time', 'energy_difference']
    #     writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
   
    
    # for filename in instances_files:
    #     instance = SpinGlassManager.load_instance(filename, instance_type)
    #     with open(filename, 'r') as file:
    #         instance_json = json.load(file)
        
        
    #     for i in range(20):
    #         digital_solver_optimized = DigitalAnnealerSpinGlassSolver_OptimizedX(initial_temperature=optimized_params['initial_temperature'], 
    #                                                                                 final_temperature=optimized_params['final_temperature'], 
    #                                                                                 alpha=optimized_params['alpha'], 
    #                                                                                 num_iterations=optimized_params['num_iterations'], 
    #                                                                                 offset_increase_rate=optimized_params['offset_increase_rate'],
    #                                                                                 num_runs=optimized_params['num_runs'])
    #         solution = digital_solver_optimized.solve(instance)
    #         print("i solved the instance", filename)
            
    #         gpu_annealer_energy = solution.energy
    #         gpu_annealer_time = solution.solving_time
    #         instance_id = instance_json['id']
    #         gurobi_energy = instance_json['best_energy']
    #         gurobi_time = instance_json['time_to_solve']
    #         energy_difference = gpu_annealer_energy - gurobi_energy
            
    #         with open(output_csv_path, mode='a', newline='') as csv_file:
    #             fieldnames = ['instance_id','N','iteration_number','gurobi_energy','gurobi_time', 'annealer_energy','annealer_time', 'energy_difference']
    #             writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
    #             writer.writerow({
    #                 'instance_id': instance_id,
    #                 'N': n_value,
    #                 'iteration_number': i,
    #                 'gurobi_energy': gurobi_energy,
    #                 'gurobi_time': gurobi_time,
    #                 'annealer_energy': gpu_annealer_energy,
    #                 'annealer_time': gpu_annealer_time,
    #                 'energy_difference': energy_difference
    #             })

if __name__ == "__main__":
    main()