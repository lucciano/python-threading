import time
import math
import multiprocessing
from concurrent.futures import ProcessPoolExecutor

def cpu_intensive_task(n=1000000):
    """
    A CPU-intensive function that calculates prime numbers up to n.
    This function will consume significant CPU resources.
    """
    print(f"Starting CPU-intensive task: finding primes up to {n}")
    start_time = time.time()
    
    primes = []
    for num in range(2, n):
        is_prime = True
        # Check if num is prime by testing divisibility up to sqrt(num)
        for i in range(2, int(math.sqrt(num)) + 1):
            if num % i == 0:
                is_prime = False
                break
        if is_prime:
            primes.append(num)
    
    end_time = time.time()
    print(f"Found {len(primes)} prime numbers in {end_time - start_time:.2f} seconds")
    return primes

def matrix_multiplication_cpu_intensive(size=1000):
    """
    Another CPU-intensive function that performs matrix multiplication.
    This creates large matrices and multiplies them, consuming lots of CPU.
    """
    print(f"Starting matrix multiplication with {size}x{size} matrices")
    start_time = time.time()
    
    # Create two large matrices filled with random-like values
    matrix_a = [[(i * j) % 100 for j in range(size)] for i in range(size)]
    matrix_b = [[(i + j) % 100 for j in range(size)] for i in range(size)]
    result = [[0 for _ in range(size)] for _ in range(size)]
    
    # Perform matrix multiplication (O(n³) complexity)
    for i in range(size):
        for j in range(size):
            for k in range(size):
                result[i][j] += matrix_a[i][k] * matrix_b[k][j]
    
    end_time = time.time()
    print(f"Matrix multiplication completed in {end_time - start_time:.2f} seconds")
    return result

def fibonacci_cpu_intensive(n=40):
    """
    Calculate Fibonacci numbers using the inefficient recursive approach.
    This is CPU-intensive due to exponential time complexity.
    """
    print(f"Calculating Fibonacci numbers up to F({n})")
    start_time = time.time()
    
    def fib(n):
        if n <= 1:
            return n
        return fib(n-1) + fib(n-2)
    
    results = [fib(i) for i in range(n)]
    end_time = time.time()
    print(f"Fibonacci calculation completed in {end_time - start_time:.2f} seconds")
    return results

def run_cpu_tasks_parallel(num_cores=None):
    """
    Run multiple CPU-intensive tasks in parallel using multiprocessing.
    This will utilize multiple CPU cores simultaneously.
    """
    print("Starting parallel CPU-intensive tasks...")
    start_time = time.time()
    
    # Define the tasks to run in parallel
    tasks = [50000, 50001, 50002, 50003, 50004, 50005, 50006, 50007, 50008, 50009, 50010]
    
    # Get the number of CPU cores available
    if num_cores is None:
        num_cores = multiprocessing.cpu_count()
    else:
        num_cores = int(num_cores)
    print(f"Running {len(tasks)} tasks in parallel using {num_cores} CPU cores")
    
    # Use ProcessPoolExecutor to run tasks in parallel
    with ProcessPoolExecutor(max_workers=num_cores) as executor:
        # Submit all tasks to the executor
        futures = [executor.submit(cpu_intensive_task, n) for n in tasks]
        
        # Wait for all tasks to complete and collect results
        results = []
        for i, future in enumerate(futures):
            try:
                result = future.result()
                results.append(result)
                print(f"Task {i+1} completed")
            except Exception as e:
                print(f"Task {i+1} failed: {e}")
    
    end_time = time.time()
    print(f"All parallel tasks completed in {end_time - start_time:.2f} seconds")
    return results

def main():
    print("Hello from python-threading!")

    
    # Uncomment one of these to run a CPU-intensive task:
    
    # Option 1: Prime number calculation using multiprocessing (parallel CPU usage)
    run_cpu_tasks_parallel(6)
    
    # Option 2: Matrix multiplication (high CPU usage)
    # matrix_multiplication_cpu_intensive(500)
    
    # Option 3: Fibonacci calculation (very high CPU usage for large n)
    # fibonacci_cpu_intensive(35)
    


if __name__ == "__main__":
    main()
