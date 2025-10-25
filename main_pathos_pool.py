import math
from pathos.multiprocessing import Pool
from multiprocessing import cpu_count
import sys
import time
import threading

def beep_every_2_seconds():
    """
    A method that prints 'beep' every 20 seconds.
    This runs in a separate thread so it doesn't block other operations.
    """
    while True:
        time.sleep(2)
        print("beep")

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

if __name__ == '__main__':
    num_cores = int(sys.argv[1]) if len(sys.argv) > 1 else cpu_count()
    
    # Start the beep thread in the background
    beep_thread = threading.Thread(target=beep_every_2_seconds, daemon=True)
    beep_thread.start()
    print("Beep thread started - will print 'beep' every 20 seconds")
    
    with Pool(num_cores) as p:
        print(p.map(cpu_intensive_task, [1000000, 1000001, 1000002, 1000003, 1000004, 1000005, 1000006, 1000007, 1000008, 1000009, 1000010]))