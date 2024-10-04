# Spin Glass Solver Project

This project implements various solvers for Spin Glass problems, including instance generation, solving, and result management.

## File Descriptions

1. `instance_generation_main.py`:
   - Main script for generating and solving Spin Glass instances.
   - Contains functions for generating instances, solving them, and saving results.
   - Provides a command-line interface to run the generation and solving processes.

2. `simulated_annealer_spin_glass_solver.py`:
   - Implements the Simulated Annealing algorithm for solving Spin Glass problems.

3. `spin_glass_base.py`:
   - Contains base classes and enums for Spin Glass problems, including `DistributionType`.

4. `digital_annealer_solver.py`:
   - Implements the Digital Annealer algorithm for solving Spin Glass problems.
   - Includes an optimized version of the solver.

5. `sk_spin_glass.py`:
   - Implements the Sherrington-Kirkpatrick (SK) Spin Glass model.

6. `two_d_spin_glass.py`:
   - Implements the 2D Spin Glass on a torus model.

7. `gurobi_spin_glass_solver.py`:
   - Implements a Gurobi-based solver for Spin Glass problems.

8. `spin_glass_instance_generator.py`:
   - Contains the `SpinGlassGenerator` class for generating Spin Glass instances.

9. `spin_glass_instance_manager.py`:
   - Provides the `SpinGlassManager` class for loading, saving, and managing Spin Glass instances and solutions.

## Requirements

Create a `requirements.txt` file with the following contents:

```
gurobipy
numpy
scipy
```

## Installation Guide

1. Ensure you have Python 3.7+ installed on your system.
2. Clone the project repository to your local machine.
3. Navigate to the project directory in your terminal.
4. Create a virtual environment (optional but recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows, use: venv\Scripts\activate
   ```
5. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```
6. Ensure you have a valid Gurobi license installed on your system.

## Running the Project

1. Open a terminal and navigate to the project directory.
2. Activate the virtual environment if you created one:
   ```bash
   source venv/bin/activate  # On Windows, use: venv\Scripts\activate
   ```
3. Run the main script:
   ```bash
   python instance_generation_main.py
   ```
4. The script will generate Spin Glass instances, solve them using various solvers, and save the results in the `solved_instances_experiment` directory.

## Customizing the Run

To customize the generation and solving process:

1. Open `instance_generation_main.py` in a text editor.
2. Modify the `config` dictionary in the `generate` function to change:
   - `output_folder`: Where results are saved
   - `problem_types`: Types of Spin Glass problems to generate
   - `N_values`: Sizes of instances to generate
   - `num_instances_per_size`: Number of instances to generate for each size
3. In the `main` function, you can choose to either generate new instances or solve existing ones by uncommenting the appropriate function call:
   - Use `generate(512, instance_type=SKSpinGlass, distribution=DistributionType.BIMODAL)` to generate new instances
   - Use `solve(instances_files, SKSpinGlass)` to solve existing instances (update `instances_files` with your file paths)

4. Save your changes and run the script as described in the "Running the Project" section.

## Output

The script will create JSON files in the `solved_instances_experiment` directory (or the directory you specified in the config). These files contain the Spin Glass instances and their solutions.

