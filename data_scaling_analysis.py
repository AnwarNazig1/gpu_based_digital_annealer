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
import numpy as np
import math




def main():
    N = 1024
    distribution = "gaussian"
    df = pd.read_csv(f'SKSpinGlass_{distribution}_N_{N}_gpu_annealer.csv')
    # Get a list of unique N values
    N_values = df['N'].unique()

    # Desired cumulative success probability
    p_success_desired = 0.99

    # Initialize a list to store results
    results = []

    for N in N_values:
        # Filter data for current N
        df_N = df[df['N'] == N]
        
        # Total number of runs for this N
        total_runs = len(df_N)
        
        # Number of successful runs
        successful_runs = df_N[df_N['energy_difference'] <= 0.1]
        num_successful_runs = len(successful_runs)
        
        # Empirical success probability
        P_success = num_successful_runs / total_runs
        
        # Average runtime per run
        avg_runtime = df_N['annealer_time'].mean()
        
        # Check if P_success is 0 or 1 to avoid division by zero or log(0)
        if P_success == 0:
            TTS = np.inf  # Infinite time, since no successful runs
        elif P_success == 1:
            TTS = avg_runtime  # Only one run needed
        else:
            # Compute TTS
            TTS = avg_runtime * (math.log(1 - p_success_desired) / math.log(1 - P_success))
        
        # Collect TTS data for different p_success values (optional)
        # For example, TTS50 (p_success=0.5) and TTS80 (p_success=0.8)
        TTS50 = avg_runtime * (math.log(1 - 0.5) / math.log(1 - P_success)) if P_success != 1 else avg_runtime
        TTS80 = avg_runtime * (math.log(1 - 0.8) / math.log(1 - P_success)) if P_success != 1 else avg_runtime
        
        # Compute percentiles of TTS distribution (optional)
        # First, compute TTS for each run
        # Note: Since runtime per run is constant in your data, TTS per run isn't meaningful unless P_success varies per run.
        # If you have runtime per run and success per run, you can compute TTS per run.
        
        # Append results
        results.append({
            'N': N,
            'Total Runs': total_runs,
            'Successful Runs': num_successful_runs,
            'P_success': P_success,
            'Avg Runtime': avg_runtime,
            'TTS_99%': TTS,
            'TTS_50%': TTS50,
            'TTS_80%': TTS80
        })


    results_df = pd.DataFrame(results)
    print(results_df)
    results_df.to_csv(f'tts_results_{N}_gauss.csv', index=False)
    
    #open the file tts_results_64.csv and write the result
    
    
    




if __name__ == "__main__":
    main()