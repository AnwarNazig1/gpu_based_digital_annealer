import os
import logging
from simulated_annealer_spin_glass_solver import SimulatedAnnealerSpinGlassSolver
from spin_glass_base import DistributionType
from digital_annealer_solver import DigitalAnnealerSpinGlassSolver, DigitalAnnealerSpinGlassSolver_OptimizedX
from sk_spin_glass import SKSpinGlass
from two_d_spin_glass import TwoDSpinGlass
from gurobi_spin_glass_solver import GurobiSpinGlassSolver
from spin_glass_instance_generator import SpinGlassGenerator
from spin_glass_instance_manager import SpinGlassManager
from gurobipy import Model, GRB, QuadExpr


def generate(N, instance_type = SKSpinGlass, distribution = DistributionType.BIMODAL):
    
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
     # Configuration
    config = {
        'output_folder': 'solved_instances_experiment',
        'problem_types': [f"{instance_type.__name__}:{distribution.value}"],#, 'SKSpinGlass:gaussian'], #'TwoDSpinGlass:bimodal', 'TwoDSpinGlass:gaussian',
        'N_values': [N],
        'num_instances_per_size': 10  # Adjust this number as needed
    }

    os.makedirs(config['output_folder'], exist_ok=True)
    solver = GurobiSpinGlassSolver()
    
    
    PROBLEM_TYPE_MAP = {
    'TwoDSpinGlass': TwoDSpinGlass,
    'SKSpinGlass': SKSpinGlass
    }
       
    for problem_type in config['problem_types']:
        for N in config['N_values']:
            instance_type = PROBLEM_TYPE_MAP[instance_type.__name__]
            distribution = DistributionType.BIMODAL
            instances = SpinGlassGenerator.generate_instances(
                            num_instances=config['num_instances_per_size'],
                            N = N,
                            instance_type = instance_type,
                            distribution= DistributionType.BIMODAL
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
        saSolver = SimulatedAnnealerSpinGlassSolver()
        
        solution = digital_solver_optimized.solve(instance)
        # filename = f"N_{instance.N}_#{i}_gb.json"
        filename = f"{filename}_digital_solver_optimized.json"
        SpinGlassManager.save_solution(instance, solution, filename)
        i += 1
        print(i)
    

def main():
    instances_files = ['solved_instances_experiment/SKSpinGlass_bimodal_N_512_0_solution.json',
                       'solved_instances_experiment/SKSpinGlass_bimodal_N_512_1_solution.json',
                       'solved_instances_experiment/SKSpinGlass_bimodal_N_512_2_solution.json',
                       'solved_instances_experiment/SKSpinGlass_bimodal_N_512_3_solution.json',
                       'solved_instances_experiment/SKSpinGlass_bimodal_N_512_4_solution.json',
                       'solved_instances_experiment/SKSpinGlass_bimodal_N_512_5_solution.json',
                       'solved_instances_experiment/SKSpinGlass_bimodal_N_512_6_solution.json',
                       'solved_instances_experiment/SKSpinGlass_bimodal_N_512_7_solution.json',
                       'solved_instances_experiment/SKSpinGlass_bimodal_N_512_8_solution.json',
                       'solved_instances_experiment/SKSpinGlass_bimodal_N_512_9_solution.json']
    # solve(instances_files , SKSpinGlass)
    generate(512, instance_type= SKSpinGlass, distribution= DistributionType.BIMODAL)

if __name__ == "__main__":
    main()
    
