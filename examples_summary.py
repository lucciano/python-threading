"""
Python Concurrency Examples Summary

This file provides a quick overview of all the concurrency examples
available in this repository with simple, runnable code snippets.
"""

import time
import threading
import multiprocessing
import asyncio
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import random


def print_section(title):
    """Print a section header."""
    print(f"\n{'='*60}")
    print(f" {title}")
    print('='*60)


# ============================================================================
# 1. BASIC THREADING EXAMPLES
# ============================================================================

def basic_threading_example():
    """Basic threading example."""
    print_section("BASIC THREADING")
    
    def worker(name, duration):
        print(f"Thread {name} starting")
        time.sleep(duration)
        print(f"Thread {name} finished")
    
    # Create and start threads
    threads = []
    for i in range(3):
        thread = threading.Thread(target=worker, args=(f"Worker-{i}", random.uniform(1, 3)))
        threads.append(thread)
        thread.start()
    
    # Wait for all threads
    for thread in threads:
        thread.join()
    
    print("All threads completed!")


def threading_with_synchronization():
    """Threading with synchronization."""
    print_section("THREADING WITH SYNCHRONIZATION")
    
    shared_data = 0
    lock = threading.Lock()
    
    def increment_data(thread_name, iterations):
        nonlocal shared_data
        for _ in range(iterations):
            with lock:
                old_value = shared_data
                time.sleep(0.001)  # Simulate work
                shared_data = old_value + 1
                print(f"{thread_name}: shared_data = {shared_data}")
    
    # Create threads
    threads = []
    for i in range(3):
        thread = threading.Thread(target=increment_data, args=(f"Thread-{i}", 5))
        threads.append(thread)
        thread.start()
    
    # Wait for completion
    for thread in threads:
        thread.join()
    
    print(f"Final shared_data: {shared_data}")


# ============================================================================
# 2. BASIC MULTIPROCESSING EXAMPLES
# ============================================================================

def basic_multiprocessing_example():
    """Basic multiprocessing example."""
    print_section("BASIC MULTIPROCESSING")
    
    def cpu_task(n):
        print(f"Processing {n}")
        return sum(i * i for i in range(n))
    
    # Use ProcessPoolExecutor
    with ProcessPoolExecutor(max_workers=multiprocessing.cpu_count()) as executor:
        numbers = [10000, 20000, 30000]
        results = list(executor.map(cpu_task, numbers))
        print(f"Results: {results}")


def multiprocessing_with_communication():
    """Multiprocessing with communication."""
    print_section("MULTIPROCESSING WITH COMMUNICATION")
    
    def producer(queue, items):
        for item in items:
            print(f"Producing: {item}")
            queue.put(item)
            time.sleep(0.5)
        queue.put(None)  # Sentinel
    
    def consumer(queue, consumer_id):
        while True:
            item = queue.get()
            if item is None:
                break
            print(f"Consumer {consumer_id}: {item}")
            time.sleep(0.3)
    
    # Create queue and processes
    queue = multiprocessing.Queue()
    items = [f"Item-{i}" for i in range(5)]
    
    producer_process = multiprocessing.Process(target=producer, args=(queue, items))
    consumer_process = multiprocessing.Process(target=consumer, args=(queue, "C1"))
    
    producer_process.start()
    consumer_process.start()
    
    producer_process.join()
    consumer_process.join()


# ============================================================================
# 3. BASIC ASYNC/AWAIT EXAMPLES
# ============================================================================

def basic_async_example():
    """Basic async/await example."""
    print_section("BASIC ASYNC/AWAIT")
    
    async def async_worker(name, duration):
        print(f"Async worker {name} starting")
        await asyncio.sleep(duration)
        print(f"Async worker {name} finished")
        return f"Result from {name}"
    
    async def main():
        # Create tasks
        tasks = [
            async_worker(f"Worker-{i}", random.uniform(1, 3))
            for i in range(3)
        ]
        
        # Run all tasks concurrently
        results = await asyncio.gather(*tasks)
        print(f"All tasks completed: {results}")
    
    # Run async main
    asyncio.run(main())


def async_with_synchronization():
    """Async with synchronization."""
    print_section("ASYNC WITH SYNCHRONIZATION")
    
    async def async_worker_with_lock(worker_id, iterations):
        shared_data = 0
        lock = asyncio.Lock()
        
        for _ in range(iterations):
            async with lock:
                old_value = shared_data
                await asyncio.sleep(0.01)
                shared_data = old_value + 1
                print(f"Worker {worker_id}: shared_data = {shared_data}")
    
    async def main():
        tasks = [
            async_worker_with_lock(i, 3)
            for i in range(3)
        ]
        await asyncio.gather(*tasks)
    
    asyncio.run(main())


# ============================================================================
# 4. BASIC CONCURRENT.FUTURES EXAMPLES
# ============================================================================

def basic_concurrent_futures_example():
    """Basic concurrent.futures example."""
    print_section("BASIC CONCURRENT.FUTURES")
    
    def io_task(duration):
        print(f"Processing I/O task for {duration}s")
        time.sleep(duration)
        return f"Completed after {duration}s"
    
    def cpu_task(n):
        print(f"Processing CPU task {n}")
        return sum(i * i for i in range(n))
    
    # I/O-bound tasks with ThreadPoolExecutor
    print("I/O-bound tasks:")
    with ThreadPoolExecutor(max_workers=3) as executor:
        io_durations = [0.5, 1.0, 1.5]
        io_results = list(executor.map(io_task, io_durations))
        print(f"I/O results: {io_results}")
    
    # CPU-bound tasks with ProcessPoolExecutor
    print("\nCPU-bound tasks:")
    with ProcessPoolExecutor(max_workers=3) as executor:
        cpu_numbers = [10000, 20000, 30000]
        cpu_results = list(executor.map(cpu_task, cpu_numbers))
        print(f"CPU results: {cpu_results}")


# ============================================================================
# 5. PERFORMANCE COMPARISON
# ============================================================================

def performance_comparison():
    """Simple performance comparison."""
    print_section("PERFORMANCE COMPARISON")
    
    def cpu_task(n):
        return sum(i * i for i in range(n))
    
    def io_task(duration):
        time.sleep(duration)
        return f"Completed after {duration}s"
    
    # Test data
    cpu_numbers = [10000, 20000, 30000]
    io_durations = [0.5, 1.0, 1.5]
    
    # Sequential execution
    print("Sequential execution:")
    start = time.time()
    sequential_cpu = [cpu_task(n) for n in cpu_numbers]
    sequential_io = [io_task(d) for d in io_durations]
    sequential_time = time.time() - start
    print(f"  Time: {sequential_time:.2f}s")
    
    # Threading
    print("\nThreading:")
    start = time.time()
    with ThreadPoolExecutor(max_workers=3) as executor:
        thread_cpu = list(executor.map(cpu_task, cpu_numbers))
        thread_io = list(executor.map(io_task, io_durations))
    thread_time = time.time() - start
    print(f"  Time: {thread_time:.2f}s")
    print(f"  Speedup: {sequential_time/thread_time:.2f}x")
    
    # Multiprocessing
    print("\nMultiprocessing:")
    start = time.time()
    with ProcessPoolExecutor(max_workers=3) as executor:
        process_cpu = list(executor.map(cpu_task, cpu_numbers))
        process_io = list(executor.map(io_task, io_durations))
    process_time = time.time() - start
    print(f"  Time: {process_time:.2f}s")
    print(f"  Speedup: {sequential_time/process_time:.2f}x")
    
    # Async
    print("\nAsync/Await:")
    async def async_cpu_task(n):
        return cpu_task(n)
    
    async def async_io_task(d):
        await asyncio.sleep(d)
        return f"Completed after {d}s"
    
    async def run_async():
        cpu_tasks = [async_cpu_task(n) for n in cpu_numbers]
        io_tasks = [async_io_task(d) for d in io_durations]
        return await asyncio.gather(*cpu_tasks, *io_tasks)
    
    start = time.time()
    async_results = asyncio.run(run_async())
    async_time = time.time() - start
    print(f"  Time: {async_time:.2f}s")
    print(f"  Speedup: {sequential_time/async_time:.2f}x")


# ============================================================================
# 6. REAL-WORLD EXAMPLES
# ============================================================================

def real_world_threading_example():
    """Real-world threading example."""
    print_section("REAL-WORLD THREADING EXAMPLE")
    
    class WebScraper:
        def __init__(self, max_workers=3):
            self.max_workers = max_workers
        
        def fetch_url(self, url):
            print(f"Fetching {url}")
            time.sleep(random.uniform(0.5, 2))  # Simulate network delay
            return f"Content from {url}"
        
        def scrape_urls(self, urls):
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                results = list(executor.map(self.fetch_url, urls))
            return results
    
    # Example usage
    scraper = WebScraper(max_workers=3)
    urls = [f"https://example.com/page/{i}" for i in range(5)]
    
    start_time = time.time()
    results = scraper.scrape_urls(urls)
    scraping_time = time.time() - start_time
    
    print(f"Scraped {len(results)} URLs in {scraping_time:.2f}s")
    print(f"Results: {results}")


def real_world_multiprocessing_example():
    """Real-world multiprocessing example."""
    print_section("REAL-WORLD MULTIPROCESSING EXAMPLE")
    
    def process_data(data):
        print(f"Processing data: {data}")
        # Simulate CPU-intensive processing
        result = sum(i * i for i in range(data))
        time.sleep(0.5)  # Simulate additional work
        return result
    
    # Example usage
    data_items = [1000, 2000, 3000, 4000, 5000]
    
    with ProcessPoolExecutor(max_workers=multiprocessing.cpu_count()) as executor:
        start_time = time.time()
        results = list(executor.map(process_data, data_items))
        processing_time = time.time() - start_time
    
    print(f"Processed {len(results)} items in {processing_time:.2f}s")
    print(f"Results: {results}")


def real_world_async_example():
    """Real-world async example."""
    print_section("REAL-WORLD ASYNC EXAMPLE")
    
    async def fetch_data(url):
        print(f"Fetching {url}")
        await asyncio.sleep(random.uniform(0.5, 2))  # Simulate network delay
        return f"Data from {url}"
    
    async def process_data_async(urls):
        tasks = [fetch_data(url) for url in urls]
        return await asyncio.gather(*tasks)
    
    # Example usage
    urls = [f"https://api.example.com/data/{i}" for i in range(5)]
    
    start_time = time.time()
    results = asyncio.run(process_data_async(urls))
    processing_time = time.time() - start_time
    
    print(f"Fetched {len(results)} data items in {processing_time:.2f}s")
    print(f"Results: {results}")


# ============================================================================
# MAIN FUNCTION
# ============================================================================

def main():
    """Run all examples."""
    print("Python Concurrency Examples Summary")
    print("This file demonstrates basic concurrency patterns in Python.")
    
    # Basic examples
    basic_threading_example()
    threading_with_synchronization()
    basic_multiprocessing_example()
    multiprocessing_with_communication()
    basic_async_example()
    async_with_synchronization()
    basic_concurrent_futures_example()
    
    # Performance comparison
    performance_comparison()
    
    # Real-world examples
    real_world_threading_example()
    real_world_multiprocessing_example()
    real_world_async_example()
    
    print("\n" + "="*60)
    print(" ALL EXAMPLES COMPLETED")
    print("="*60)
    print("\nFor more comprehensive examples, run:")
    print("  python main_demo.py")
    print("\nOr run specific example files:")
    print("  python threading_examples.py")
    print("  python multiprocessing_examples.py")
    print("  python async_examples.py")
    print("  python concurrent_futures_examples.py")
    print("  python concurrency_comparison.py")


if __name__ == "__main__":
    main()
