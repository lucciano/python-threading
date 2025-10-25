"""
Comprehensive Python Concurrent.Futures Examples

This module demonstrates various concurrent.futures approaches in Python:
1. ThreadPoolExecutor for I/O-bound tasks
2. ProcessPoolExecutor for CPU-bound tasks
3. Future objects and callbacks
4. Exception handling in concurrent execution
5. Timeout and cancellation
6. Performance comparison between executors
7. Real-world concurrent patterns
8. Advanced concurrent programming techniques
"""

import time
import random
import math
import requests
import logging
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from concurrent.futures import Future, Executor
from typing import List, Dict, Any, Callable, Optional
import threading
import multiprocessing
from functools import partial
import json

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(threadName)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ConcurrentFuturesExamples:
    """Comprehensive concurrent.futures examples and demonstrations."""
    
    def __init__(self):
        self.results = []
        self.shared_data = 0
        self.lock = threading.Lock()
    
    def basic_thread_pool_example(self):
        """Example 1: Basic ThreadPoolExecutor usage."""
        print("\n=== Basic ThreadPoolExecutor Example ===")
        
        def io_bound_task(url: str) -> str:
            """Simulate an I/O-bound task."""
            logger.info(f"Processing {url}")
            time.sleep(random.uniform(0.5, 2))  # Simulate I/O delay
            return f"Result from {url}"
        
        # URLs to process
        urls = [
            "https://httpbin.org/delay/1",
            "https://httpbin.org/delay/2",
            "https://httpbin.org/delay/1",
            "https://httpbin.org/delay/3",
            "https://httpbin.org/delay/1"
        ]
        
        # Use ThreadPoolExecutor for I/O-bound tasks
        with ThreadPoolExecutor(max_workers=3) as executor:
            # Submit all tasks
            futures = [executor.submit(io_bound_task, url) for url in urls]
            
            # Collect results as they complete
            results = []
            for future in as_completed(futures):
                try:
                    result = future.result()
                    results.append(result)
                    logger.info(f"Completed: {result}")
                except Exception as e:
                    logger.error(f"Task failed: {e}")
        
        print(f"All tasks completed: {results}")
    
    def basic_process_pool_example(self):
        """Example 2: Basic ProcessPoolExecutor usage."""
        print("\n=== Basic ProcessPoolExecutor Example ===")
        
        def cpu_bound_task(n: int) -> int:
            """CPU-intensive task."""
            logger.info(f"Processing task {n}")
            result = sum(i * i for i in range(n))
            return result
        
        # Numbers to process
        numbers = [100000, 200000, 300000, 400000, 500000]
        
        # Use ProcessPoolExecutor for CPU-bound tasks
        with ProcessPoolExecutor(max_workers=multiprocessing.cpu_count()) as executor:
            # Submit all tasks
            futures = [executor.submit(cpu_bound_task, n) for n in numbers]
            
            # Collect results
            results = []
            for future in as_completed(futures):
                try:
                    result = future.result()
                    results.append(result)
                    logger.info(f"Completed: {result}")
                except Exception as e:
                    logger.error(f"Task failed: {e}")
        
        print(f"All tasks completed: {results}")
    
    def future_callbacks_example(self):
        """Example 3: Future callbacks and state management."""
        print("\n=== Future Callbacks Example ===")
        
        def task_with_callback(task_id: int) -> str:
            """Task that returns a result."""
            time.sleep(random.uniform(1, 3))
            return f"Task {task_id} completed"
        
        def success_callback(future: Future):
            """Callback for successful completion."""
            result = future.result()
            logger.info(f"Success callback: {result}")
        
        def failure_callback(future: Future):
            """Callback for task failure."""
            try:
                future.result()
            except Exception as e:
                logger.error(f"Failure callback: {e}")
        
        # Create executor and submit tasks
        with ThreadPoolExecutor(max_workers=2) as executor:
            futures = []
            
            for i in range(5):
                future = executor.submit(task_with_callback, i)
                
                # Add callbacks
                future.add_done_callback(success_callback)
                future.add_done_callback(failure_callback)
                
                futures.append(future)
            
            # Wait for all tasks
            for future in futures:
                future.result()
    
    def exception_handling_example(self):
        """Example 4: Exception handling in concurrent execution."""
        print("\n=== Exception Handling Example ===")
        
        def risky_task(task_id: int) -> str:
            """Task that might fail."""
            time.sleep(random.uniform(0.5, 1.5))
            
            if random.random() < 0.3:  # 30% chance of failure
                raise ValueError(f"Task {task_id} failed randomly")
            
            return f"Task {task_id} succeeded"
        
        def handle_task_with_retry(task_id: int, max_retries: int = 3) -> str:
            """Task with retry logic."""
            for attempt in range(max_retries):
                try:
                    return risky_task(task_id)
                except ValueError as e:
                    if attempt == max_retries - 1:
                        raise e
                    logger.info(f"Task {task_id} failed, retrying... (attempt {attempt + 1})")
                    time.sleep(0.5)
        
        # Submit tasks with exception handling
        with ThreadPoolExecutor(max_workers=3) as executor:
            futures = []
            
            # Submit regular tasks
            for i in range(5):
                future = executor.submit(risky_task, i)
                futures.append(future)
            
            # Submit tasks with retry
            for i in range(5, 8):
                future = executor.submit(handle_task_with_retry, i)
                futures.append(future)
            
            # Handle results
            successful_results = []
            failed_results = []
            
            for future in as_completed(futures):
                try:
                    result = future.result()
                    successful_results.append(result)
                except Exception as e:
                    failed_results.append(str(e))
            
            print(f"Successful: {successful_results}")
            print(f"Failed: {failed_results}")
    
    def timeout_and_cancellation_example(self):
        """Example 5: Timeout and cancellation."""
        print("\n=== Timeout and Cancellation Example ===")
        
        def long_running_task(task_id: int, duration: float) -> str:
            """Task that takes a long time."""
            logger.info(f"Long task {task_id} starting")
            time.sleep(duration)
            return f"Long task {task_id} completed"
        
        def timeout_example():
            """Example of timeout handling."""
            with ThreadPoolExecutor(max_workers=2) as executor:
                # Submit a task that will timeout
                future = executor.submit(long_running_task, 1, 5)  # 5 second task
                
                try:
                    result = future.result(timeout=2)  # 2 second timeout
                    return result
                except TimeoutError:
                    logger.info("Task timed out")
                    return "Timeout"
        
        def cancellation_example():
            """Example of task cancellation."""
            with ThreadPoolExecutor(max_workers=3) as executor:
                # Submit multiple long-running tasks
                futures = [
                    executor.submit(long_running_task, i, 3)
                    for i in range(5)
                ]
                
                # Cancel some tasks after 1 second
                time.sleep(1)
                for i, future in enumerate(futures):
                    if i % 2 == 0:  # Cancel even-indexed tasks
                        future.cancel()
                        logger.info(f"Cancelled task {i}")
                
                # Collect results
                results = []
                for i, future in enumerate(futures):
                    try:
                        result = future.result()
                        results.append(f"Task {i}: {result}")
                    except Exception as e:
                        results.append(f"Task {i}: {type(e).__name__}")
                
                return results
        
        # Test timeout
        timeout_result = timeout_example()
        print(f"Timeout result: {timeout_result}")
        
        # Test cancellation
        cancellation_results = cancellation_example()
        print(f"Cancellation results: {cancellation_results}")
    
    def performance_comparison_example(self):
        """Example 6: Performance comparison between executors."""
        print("\n=== Performance Comparison Example ===")
        
        def cpu_intensive_task(n: int) -> int:
            """CPU-intensive task."""
            return sum(i * i for i in range(n))
        
        def io_intensive_task(duration: float) -> str:
            """I/O-intensive task."""
            time.sleep(duration)
            return f"Completed after {duration}s"
        
        def mixed_task(n: int, duration: float) -> tuple:
            """Mixed CPU and I/O task."""
            time.sleep(duration)  # I/O part
            result = sum(i * i for i in range(n))  # CPU part
            return result, duration
        
        # Test parameters
        cpu_tasks = [10000, 20000, 30000, 40000]
        io_tasks = [0.5, 1.0, 1.5, 2.0]
        mixed_tasks = [(1000, 0.2), (2000, 0.3), (3000, 0.4), (4000, 0.5)]
        
        # Test CPU-bound tasks
        print("CPU-bound tasks:")
        
        # Sequential execution
        start_time = time.time()
        sequential_results = [cpu_intensive_task(n) for n in cpu_tasks]
        sequential_time = time.time() - start_time
        print(f"Sequential time: {sequential_time:.2f}s")
        
        # ThreadPoolExecutor
        start_time = time.time()
        with ThreadPoolExecutor(max_workers=4) as executor:
            thread_results = list(executor.map(cpu_intensive_task, cpu_tasks))
        thread_time = time.time() - start_time
        print(f"ThreadPoolExecutor time: {thread_time:.2f}s")
        
        # ProcessPoolExecutor
        start_time = time.time()
        with ProcessPoolExecutor(max_workers=4) as executor:
            process_results = list(executor.map(cpu_intensive_task, cpu_tasks))
        process_time = time.time() - start_time
        print(f"ProcessPoolExecutor time: {process_time:.2f}s")
        
        print(f"Thread speedup: {sequential_time/thread_time:.2f}x")
        print(f"Process speedup: {sequential_time/process_time:.2f}x")
        
        # Test I/O-bound tasks
        print("\nI/O-bound tasks:")
        
        # Sequential execution
        start_time = time.time()
        sequential_results = [io_intensive_task(d) for d in io_tasks]
        sequential_time = time.time() - start_time
        print(f"Sequential time: {sequential_time:.2f}s")
        
        # ThreadPoolExecutor
        start_time = time.time()
        with ThreadPoolExecutor(max_workers=4) as executor:
            thread_results = list(executor.map(io_intensive_task, io_tasks))
        thread_time = time.time() - start_time
        print(f"ThreadPoolExecutor time: {thread_time:.2f}s")
        
        print(f"Thread speedup: {sequential_time/thread_time:.2f}x")
    
    def advanced_patterns_example(self):
        """Example 7: Advanced concurrent patterns."""
        print("\n=== Advanced Patterns Example ===")
        
        class ConcurrentDataProcessor:
            """Example of advanced concurrent data processing."""
            
            def __init__(self, max_workers: int = 4):
                self.max_workers = max_workers
                self.processed_count = 0
                self.lock = threading.Lock()
            
            def process_item(self, item: Dict[str, Any]) -> Dict[str, Any]:
                """Process a single item."""
                time.sleep(random.uniform(0.1, 0.5))  # Simulate processing
                
                with self.lock:
                    self.processed_count += 1
                
                return {
                    "id": item["id"],
                    "processed": True,
                    "result": item["value"] * 2
                }
            
            def batch_process(self, items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
                """Process multiple items concurrently."""
                with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                    futures = [executor.submit(self.process_item, item) for item in items]
                    return [future.result() for future in as_completed(futures)]
            
            def get_processed_count(self) -> int:
                """Get the number of processed items."""
                with self.lock:
                    return self.processed_count
        
        # Create processor and sample data
        processor = ConcurrentDataProcessor(max_workers=3)
        sample_data = [
            {"id": i, "value": random.randint(1, 100)}
            for i in range(20)
        ]
        
        # Process data
        start_time = time.time()
        results = processor.batch_process(sample_data)
        processing_time = time.time() - start_time
        
        print(f"Processed {len(results)} items in {processing_time:.2f}s")
        print(f"Total processed count: {processor.get_processed_count()}")
    
    def real_world_example(self):
        """Example 8: Real-world concurrent application."""
        print("\n=== Real-World Example ===")
        
        class WebScraper:
            """Example web scraper using concurrent.futures."""
            
            def __init__(self, max_workers: int = 5):
                self.max_workers = max_workers
            
            def fetch_url(self, url: str) -> Dict[str, Any]:
                """Fetch a single URL (simulated)."""
                logger.info(f"Fetching {url}")
                time.sleep(random.uniform(0.5, 2))  # Simulate network delay
                
                return {
                    "url": url,
                    "status": "success",
                    "content_length": random.randint(1000, 10000),
                    "timestamp": time.time()
                }
            
            def scrape_urls(self, urls: List[str]) -> List[Dict[str, Any]]:
                """Scrape multiple URLs concurrently."""
                with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                    futures = [executor.submit(self.fetch_url, url) for url in urls]
                    
                    results = []
                    for future in as_completed(futures):
                        try:
                            result = future.result()
                            results.append(result)
                        except Exception as e:
                            logger.error(f"Failed to fetch URL: {e}")
                    
                    return results
        
        # Create scraper and sample URLs
        scraper = WebScraper(max_workers=3)
        urls = [
            f"https://example.com/page/{i}"
            for i in range(10)
        ]
        
        # Scrape URLs
        start_time = time.time()
        results = scraper.scrape_urls(urls)
        scraping_time = time.time() - start_time
        
        print(f"Scraped {len(results)} URLs in {scraping_time:.2f}s")
        print(f"Average time per URL: {scraping_time/len(results):.2f}s")
    
    def custom_executor_example(self):
        """Example 9: Custom executor implementation."""
        print("\n=== Custom Executor Example ===")
        
        class CustomExecutor(Executor):
            """Custom executor that limits concurrent execution."""
            
            def __init__(self, max_workers: int = 2):
                self.max_workers = max_workers
                self.semaphore = threading.Semaphore(max_workers)
                self.threads = []
            
            def submit(self, fn, *args, **kwargs):
                """Submit a task to the executor."""
                future = Future()
                
                def worker():
                    with self.semaphore:
                        try:
                            result = fn(*args, **kwargs)
                            future.set_result(result)
                        except Exception as e:
                            future.set_exception(e)
                
                thread = threading.Thread(target=worker)
                thread.start()
                self.threads.append(thread)
                
                return future
            
            def shutdown(self, wait=True):
                """Shutdown the executor."""
                if wait:
                    for thread in self.threads:
                        thread.join()
        
        def custom_task(task_id: int) -> str:
            """Task for custom executor."""
            logger.info(f"Custom task {task_id} starting")
            time.sleep(random.uniform(1, 3))
            return f"Custom task {task_id} completed"
        
        # Use custom executor
        with CustomExecutor(max_workers=2) as executor:
            futures = [executor.submit(custom_task, i) for i in range(5)]
            results = [future.result() for future in futures]
        
        print(f"Custom executor results: {results}")
    
    def run_all_examples(self):
        """Run all concurrent.futures examples."""
        print("Starting comprehensive concurrent.futures examples...")
        
        self.basic_thread_pool_example()
        self.basic_process_pool_example()
        self.future_callbacks_example()
        self.exception_handling_example()
        self.timeout_and_cancellation_example()
        self.performance_comparison_example()
        self.advanced_patterns_example()
        self.real_world_example()
        self.custom_executor_example()
        
        print("\nAll concurrent.futures examples completed!")


def main():
    """Main function to run concurrent.futures examples."""
    examples = ConcurrentFuturesExamples()
    examples.run_all_examples()


if __name__ == "__main__":
    main()
