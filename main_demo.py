"""
Comprehensive Python Concurrency Demo

This is the main demonstration file that showcases all Python concurrency approaches:
- Threading
- Multiprocessing  
- Async/Await
- Concurrent.Futures
- Performance Comparison

Run this file to see all examples in action.
"""

import sys
import time
import logging
from typing import List

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def print_banner(title: str):
    """Print a formatted banner."""
    print("\n" + "="*80)
    print(f" {title}")
    print("="*80)


def run_threading_examples():
    """Run threading examples."""
    print_banner("THREADING EXAMPLES")
    
    try:
        from threading_examples import ThreadingExamples
        examples = ThreadingExamples()
        examples.run_all_examples()
    except ImportError as e:
        logger.error(f"Failed to import threading examples: {e}")
    except Exception as e:
        logger.error(f"Error running threading examples: {e}")


def run_multiprocessing_examples():
    """Run multiprocessing examples."""
    print_banner("MULTIPROCESSING EXAMPLES")
    
    try:
        from multiprocessing_examples import MultiprocessingExamples
        examples = MultiprocessingExamples()
        examples.run_all_examples()
    except ImportError as e:
        logger.error(f"Failed to import multiprocessing examples: {e}")
    except Exception as e:
        logger.error(f"Error running multiprocessing examples: {e}")


def run_async_examples():
    """Run async examples."""
    print_banner("ASYNC/AWAIT EXAMPLES")
    
    try:
        from async_examples import run_async_examples
        run_async_examples()
    except ImportError as e:
        logger.error(f"Failed to import async examples: {e}")
    except Exception as e:
        logger.error(f"Error running async examples: {e}")


def run_concurrent_futures_examples():
    """Run concurrent.futures examples."""
    print_banner("CONCURRENT.FUTURES EXAMPLES")
    
    try:
        from concurrent_futures_examples import ConcurrentFuturesExamples
        examples = ConcurrentFuturesExamples()
        examples.run_all_examples()
    except ImportError as e:
        logger.error(f"Failed to import concurrent.futures examples: {e}")
    except Exception as e:
        logger.error(f"Error running concurrent.futures examples: {e}")


def run_comparison_demo():
    """Run performance comparison demo."""
    print_banner("PERFORMANCE COMPARISON")
    
    try:
        from concurrency_comparison import ConcurrencyComparison
        comparison = ConcurrencyComparison()
        comparison.run_comprehensive_comparison()
    except ImportError as e:
        logger.error(f"Failed to import comparison demo: {e}")
    except Exception as e:
        logger.error(f"Error running comparison demo: {e}")


def run_quick_demo():
    """Run a quick demo of all approaches."""
    print_banner("QUICK CONCURRENCY DEMO")
    
    import threading
    import multiprocessing
    import asyncio
    from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
    import time
    import random
    
    def cpu_task(n):
        """CPU-intensive task."""
        return sum(i * i for i in range(n))
    
    def io_task(duration):
        """I/O-intensive task."""
        time.sleep(duration)
        return f"Completed after {duration}s"
    
    # Test data
    cpu_numbers = [10000, 20000, 30000]
    io_durations = [0.5, 1.0, 1.5]
    
    print("\n1. Sequential Execution:")
    start = time.time()
    sequential_results = [cpu_task(n) for n in cpu_numbers]
    sequential_time = time.time() - start
    print(f"   Time: {sequential_time:.2f}s")
    
    print("\n2. Threading:")
    start = time.time()
    with ThreadPoolExecutor(max_workers=3) as executor:
        thread_results = list(executor.map(cpu_task, cpu_numbers))
    thread_time = time.time() - start
    print(f"   Time: {thread_time:.2f}s")
    
    print("\n3. Multiprocessing:")
    start = time.time()
    with ProcessPoolExecutor(max_workers=3) as executor:
        process_results = list(executor.map(cpu_task, cpu_numbers))
    process_time = time.time() - start
    print(f"   Time: {process_time:.2f}s")
    
    print("\n4. Async/Await:")
    async def async_cpu_task(n):
        return cpu_task(n)
    
    async def run_async():
        return await asyncio.gather(*[async_cpu_task(n) for n in cpu_numbers])
    
    start = time.time()
    async_results = asyncio.run(run_async())
    async_time = time.time() - start
    print(f"   Time: {async_time:.2f}s")
    
    print(f"\nSpeedup comparison:")
    print(f"   Threading: {sequential_time/thread_time:.2f}x")
    print(f"   Multiprocessing: {sequential_time/process_time:.2f}x")
    print(f"   Async: {sequential_time/async_time:.2f}x")


def show_menu():
    """Show the main menu."""
    print("\n" + "="*80)
    print(" PYTHON CONCURRENCY EXAMPLES - MAIN MENU")
    print("="*80)
    print("1. Run Threading Examples")
    print("2. Run Multiprocessing Examples") 
    print("3. Run Async/Await Examples")
    print("4. Run Concurrent.Futures Examples")
    print("5. Run Performance Comparison")
    print("6. Run Quick Demo")
    print("7. Run All Examples")
    print("8. Exit")
    print("="*80)


def main():
    """Main function with interactive menu."""
    if len(sys.argv) > 1:
        # Command line mode
        choice = sys.argv[1]
    else:
        # Interactive mode
        show_menu()
        choice = input("\nEnter your choice (1-8): ").strip()
    
    if choice == "1":
        run_threading_examples()
    elif choice == "2":
        run_multiprocessing_examples()
    elif choice == "3":
        run_async_examples()
    elif choice == "4":
        run_concurrent_futures_examples()
    elif choice == "5":
        run_comparison_demo()
    elif choice == "6":
        run_quick_demo()
    elif choice == "7":
        print_banner("RUNNING ALL EXAMPLES")
        run_threading_examples()
        run_multiprocessing_examples()
        run_async_examples()
        run_concurrent_futures_examples()
        run_comparison_demo()
    elif choice == "8":
        print("Goodbye!")
        return
    else:
        print("Invalid choice. Please run the script again.")
        return
    
    print("\n" + "="*80)
    print(" DEMO COMPLETED")
    print("="*80)


if __name__ == "__main__":
    main()
