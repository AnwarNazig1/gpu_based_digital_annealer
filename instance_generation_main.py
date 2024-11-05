import os
import logging
from simulated_annealer_spin_glass_solver import SimulatedAnnealerSpinGlassSolver
from spin_glass_base import DistributionType
from digital_annealer_solver import DigitalAnnealerSpinGlassSolver, DigitalAnnealerSpinGlassSolver_OptimizedX,DigitalAnnealerSpinGlassSolver_OptimizedX2
from sk_spin_glass import SKSpinGlass
from two_d_spin_glass import TwoDSpinGlass
from gurobi_spin_glass_solver import GurobiSpinGlassSolver
from spin_glass_instance_generator import SpinGlassGenerator
from spin_glass_instance_manager import SpinGlassManager
from gurobipy import Model, GRB, QuadExpr
import optuna
import csv
import matplotlib.pyplot as plt
import glob




# Save the results for post-optimization analysis
def save_results(trial, energy, time_taken):
    file_path = 'multi_objective_optimization_results_1024_bimodal.csv'
    file_exists = os.path.isfile(file_path)
    
    with open(file_path, mode='a', newline='') as file:
        writer = csv.writer(file)
        if not file_exists:
            # Write headers if the file is newly created
            writer.writerow(['Trial', 'Initial Temperature', 'Final Temperature', 'Alpha', 
                             'Num Iterations', 'Num Runs', 'Offset Increase Rate', 'Energy', 'Time'])
        
        # Append the results
        writer.writerow([trial.number, trial.params['initial_temperature'], trial.params['final_temperature'],
                         trial.params['alpha'], trial.params['num_iterations'], trial.params['num_runs'], 
                         trial.params['offset_increase_rate'], energy, time_taken])
        
def generate(N, instance_type = SKSpinGlass, distribution = DistributionType.BIMODAL, num_instances_per_size = 1, time_limit = 20):
    
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
     # Configuration
    config = {
        'output_folder': 'solved_instances_experiment',
        'problem_types': [f"{instance_type.__name__}:{distribution.value}"],#, 'SKSpinGlass:gaussian'], #'TwoDSpinGlass:bimodal', 'TwoDSpinGlass:gaussian',
        'N_values': [N],
        'num_instances_per_size': num_instances_per_size  # Adjust this number as needed
        
    }

    os.makedirs(config['output_folder'], exist_ok=True)
    solver = GurobiSpinGlassSolver(time_limit = time_limit)
    
    
    PROBLEM_TYPE_MAP = {
    'TwoDSpinGlass': TwoDSpinGlass,
    'SKSpinGlass': SKSpinGlass
    }
    
    distribution_map = {
    'gaussian': DistributionType.GAUSSIAN,
    'bimodal': DistributionType.BIMODAL
    }
       
    for problem_type in config['problem_types']:
        for N in config['N_values']:
            instance_type = PROBLEM_TYPE_MAP[instance_type.__name__]
            distribution = distribution_map[distribution.value]
            instances = SpinGlassGenerator.generate_instances(
                            num_instances=config['num_instances_per_size'],
                            N = N,
                            instance_type = instance_type,
                            distribution= distribution
                            )

            for idx, instance in enumerate(instances):
                try:
                    solution = solver.solve(instance)
                    filename = f"{instance_type.__name__}_{distribution.value}_N_{instance.N}_{idx}_solution.json"
                    output_path = os.path.join(config['output_folder'], filename)
                    SpinGlassManager.save_solution(instance, solution, output_path)
                    logger.info(f"Processed instance {idx+1}/{len(instances)}: {problem_type} N={instance.N}")
                except Exception as e:
                    logger.error(f"Error processing instance {idx+1}/{len(instances)}: {str(e)}")
    
def solve(instances_files , instance_type):
    i = 0
    for filename in instances_files:
        instance = SpinGlassManager.load_instance(filename, instance_type)
        gurobi_solver = GurobiSpinGlassSolver()
        digital_solver = DigitalAnnealerSpinGlassSolver()
        digital_solver_optimized = DigitalAnnealerSpinGlassSolver_OptimizedX()
        da_x2 = DigitalAnnealerSpinGlassSolver_OptimizedX2()
        saSolver = SimulatedAnnealerSpinGlassSolver()
        # gpu_annealear = GpuAnnealer()
        
        solution = saSolver.solve(instance)
        # filename = f"N_{instance.N}_#{i}_gb.json"
        filename = f"{filename}_saSolver.json"
        SpinGlassManager.save_solution(instance, solution, filename)
        i += 1
        print(i)


def load_instance_files(n_value, distribution):
    # Find all files matching the pattern *_N_{n_value}_*
    pattern = f'solved_instances_experiment/SKSpinGlass_{distribution}_N_{n_value}_*_solution.json'
    instance_files = glob.glob(pattern)
    return instance_files

# Optuna objective function
def objective(trial, n_value, distribution):
    # Parameter search space
    initial_temperature = trial.suggest_float('initial_temperature', 50.0, 200.0)
    final_temperature = trial.suggest_float('final_temperature', 0.0001, 0.01)
    alpha = trial.suggest_float('alpha', 0.9, 0.99)
    num_iterations = trial.suggest_int('num_iterations', 500, 3000)
    num_runs = trial.suggest_int('num_runs', 500, 3000)
    offset_increase_rate = trial.suggest_float('offset_increase_rate', 0.01, 500)
    
    # Load instance files dynamically based on N value (e.g., N = 128)
    instances_files = load_instance_files(n_value, distribution)


    # List of instance files to solve
    # instances_files = ['solved_instances_experiment/SKSpinGlass_bimodal_N_676_0_solution.json',
    #                    'solved_instances_experiment/SKSpinGlass_bimodal_N_676_1_solution.json',
    #                    'solved_instances_experiment/SKSpinGlass_bimodal_N_676_2_solution.json',
    #                    'solved_instances_experiment/SKSpinGlass_bimodal_N_676_4_solution.json',
    #                    'solved_instances_experiment/SKSpinGlass_bimodal_N_676_5_solution.json',
    #                    'solved_instances_experiment/SKSpinGlass_bimodal_N_676_6_solution.json',
    #                    'solved_instances_experiment/SKSpinGlass_bimodal_N_676_7_solution.json',
    #                    'solved_instances_experiment/SKSpinGlass_bimodal_N_676_8_solution.json',
    #                    'solved_instances_experiment/SKSpinGlass_bimodal_N_676_9_solution.json'
    #                    ]
    
    # Initialize the solver with the suggested parameters from Optuna
    solver = DigitalAnnealerSpinGlassSolver_OptimizedX(initial_temperature, final_temperature, alpha, num_iterations, num_runs, offset_increase_rate)

    # Solve for all instances and compute the average energy and time
    total_energy = 0
    total_time = 0
    num_instances = len(instances_files)

    for file in instances_files:
        # Load the instance
        instance = SpinGlassManager.load_instance(file, SKSpinGlass)

        # Solve using the digital annealer and measure time taken
        solution = solver.solve(instance)

        # Accumulate the energy and time for averaging later
        total_energy += solution.energy
        total_time += solution.solving_time

    # Compute the average energy and time over all instances
    average_energy = total_energy / num_instances
    average_time = total_time / num_instances

    # Save results (for further analysis)
    save_results(trial, average_energy, average_time)

    # Return both objectives (energy and time)
    return (average_energy, average_time)


# Run the optimization using Optuna
def optimize_digital_annealer(n_value, distribution):
    # Specify the SQLite database where Optuna will store the study
    storage = 'sqlite:///optuna_study.db'  # Save trials in a local SQLite database

    # Create a multi-objective Optuna study to minimize both energy and time
    study = optuna.create_study(
        directions=['minimize', 'minimize'],  # Multi-objective optimization
        storage=storage,                      # Store trials in the SQLite database
        study_name='digital_annealer_study',  # Name of the study
        load_if_exists=True                   # Load the study if it already exists
    )
    
    # Define a wrapper for the objective function to pass n_value and distribution
    def objective_wrapper(trial):
        return objective(trial, n_value, distribution)
    
    # Run optimization for a given number of trials
    study.optimize(objective_wrapper, n_trials=50)  # Run 50 trials

    # Return the study object for further analysis or plotting
    return study

def plot_pareto_front(study):
    energies = [trial.values[0] for trial in study.best_trials]
    times = [trial.values[1] for trial in study.best_trials]

    plt.figure(figsize=(8, 6))
    plt.scatter(energies, times, color='blue')
    plt.xlabel('Energy (Lower is Better)', fontsize=12)
    plt.ylabel('Time (Seconds, Lower is Better)', fontsize=12)
    plt.title('Pareto Front: Energy vs Time', fontsize=14)
    plt.grid(True)
    plt.show()
    

    
def main():
    # instance_files = ['solved_instances_experiment/SKSpinGlass_bimodal_N_144_0_solution.json',
    #                    'solved_instances_experiment/SKSpinGlass_bimodal_N_144_1_solution.json',
    #                    'solved_instances_experiment/SKSpinGlass_bimodal_N_144_2_solution.json',
    #                    'solved_instances_experiment/SKSpinGlass_bimodal_N_144_4_solution.json',
    #                    'solved_instances_experiment/SKSpinGlass_bimodal_N_144_5_solution.json',
    #                    'solved_instances_experiment/SKSpinGlass_bimodal_N_144_6_solution.json',
    #                    'solved_instances_experiment/SKSpinGlass_bimodal_N_144_7_solution.json',
    #                    'solved_instances_experiment/SKSpinGlass_bimodal_N_144_8_solution.json',
    #                    'solved_instances_experiment/SKSpinGlass_bimodal_N_144_9_solution.json'
    #                    ]
                      
    # # instance_files = load_instance_files(144, "bimodal")
    # optimized_params = {
    #     'initial_temperature': 150.0,
    #     'final_temperature':0.0001,
    #     'alpha': 0.99,
    #     'num_iterations': 2000,
    #     'offset_increase_rate': 5,
    #     'num_runs': 1000
    #     # Add other parameters as required
    # }
    # i = 0
    # # order by filename SKSpinGlass_bimodal_N_64_0_solution.json then SKSpinGlass_bimodal_N_64_1_solution.json and so on
    
    # for file in instance_files:
    #     instance = SpinGlassManager.load_instance(file, SKSpinGlass)
    #     solver = SimulatedAnnealerSpinGlassSolver(T_initial=optimized_params['initial_temperature'], T_final=optimized_params['final_temperature'], alpha=optimized_params['alpha'], num_iterations=optimized_params['num_iterations'], num_runs=optimized_params['num_runs'])
    #     # Solve using the digital annealer and measure time taken
    #     solution = solver.solve(instance)
    #     filename = f"{file}_saSolver.json"
    #     SpinGlassManager.save_solution(instance, solution, filename)
    #     i += 1
    #     print("i = ", i)
        


    # solve(instances_files , SKSpinGlass)
    
    # generate(1600, instance_type= SKSpinGlass, distribution= DistributionType.GAUSSIAN, num_instances_per_size= 50)
    generate(2048, instance_type= SKSpinGlass, distribution= DistributionType.GAUSSIAN, num_instances_per_size= 30, time_limit = 150)
    generate(3000, instance_type= SKSpinGlass, distribution= DistributionType.GAUSSIAN, num_instances_per_size= 30 = 200)
    
    # generate(64, instance_type= SKSpinGlass, distribution= DistributionType.GAUSSIAN, num_instances_per_size= 100)
    
    # # generate(576, instance_type= SKSpinGlass, distribution= DistributionType.BIMODAL, num_instances_per_size= 100)
    # generate(144, instance_type= SKSpinGlass, distribution= DistributionType.GAUSSIAN, num_instances_per_size= 100)
    
    # # generate(676, instance_type= SKSpinGlass, distribution= DistributionType.BIMODAL, num_instances_per_size= 100)
    # generate(256, instance_type= SKSpinGlass, distribution= DistributionType.GAUSSIAN, num_instances_per_size= 100)
    # generate(400, instance_type= SKSpinGlass, distribution= DistributionType.GAUSSIAN, num_instances_per_size= 100)
    # generate(576, instance_type= SKSpinGlass, distribution= DistributionType.GAUSSIAN, num_instances_per_size= 100)
    # generate(1024, instance_type= SKSpinGlass, distribution= DistributionType.BIMODAL, num_instances_per_size= 100)

    #13172
    
    #64 144, 256, 400, 576, 676, 784, 900, 1024, 2048
    
    # Run the multi-objective optimization process
    # study = optimize_digital_annealer(784, "gaussian")


    # Plot Pareto front
    # plot_pareto_front(study)

if __name__ == "__main__":
    main()
    
