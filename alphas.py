import math

def calculate_iterations_to_final_temp(initial_temp, final_temp, alpha):
    """
    Calculate the number of iterations needed to reach the final temperature.
    
    Args:
    initial_temp (float): The starting temperature.
    final_temp (float): The target final temperature.
    alpha (float): The cooling rate (0 < alpha < 1).
    
    Returns:
    int: The number of iterations needed to reach the final temperature.
    """
    if alpha <= 0 or alpha >= 1:
        raise ValueError("Alpha must be between 0 and 1 (exclusive).")
    if final_temp >= initial_temp:
        raise ValueError("Final temperature must be less than initial temperature.")
    if final_temp <= 0 or initial_temp <= 0:
        raise ValueError("Temperatures must be positive.")
    
    # The formula is derived from: final_temp = initial_temp * (alpha^iterations)
    # Taking log of both sides and solving for iterations:
    iterations = math.log(final_temp / initial_temp) / math.log(alpha)
    
    # We round up to the nearest integer because we can't have fractional iterations
    return math.ceil(iterations)

def __main__():
    # Example usage:
    initial_temp = 100
    final_temp = 0.0001
    alphas = [0.9, 0.93, 0.95, 0.97, 0.99]

    for alpha in alphas:
        iterations = calculate_iterations_to_final_temp(initial_temp, final_temp, alpha)
        print(f"Alpha: {alpha:.2f}, Iterations to reach {final_temp}: {iterations}")
        
    
    import torch
    # Check if multiple GPUs are available
    if torch.cuda.device_count() > 1:
        print(f"Number of available GPUs: {torch.cuda.device_count()}")

        # Select the desired GPU (e.g., GPU 1)
        desired_gpu = 1
        torch.cuda.set_device(desired_gpu)
        print(f"Using GPU: {desired_gpu}")
    else:
        print("Using single GPU or no GPU available")

    # Check the name of the current device
    print(f"Current GPU: {torch.cuda.get_device_name(torch.cuda.current_device())}")

if __name__ == "__main__":
    __main__()