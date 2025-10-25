"""
Comprehensive Python Threading Examples

This module demonstrates various threading approaches in Python:
1. Basic threading with Thread class
2. Thread synchronization (Lock, Semaphore, Event)
3. Thread communication (Queue)
4. ThreadPoolExecutor for managed thread pools
5. Thread-safe data structures
"""

import threading
import time
import random
import queue
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock, Semaphore, Event, Condition
import logging

# Configure logging to see thread activity
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(threadName)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ThreadingExamples:
    """Comprehensive threading examples and demonstrations."""
    
    def __init__(self):
        self.shared_data = 0
        self.lock = Lock()
        self.semaphore = Semaphore(3)  # Allow max 3 threads
        self.event = Event()
        self.condition = Condition()
        self.queue = queue.Queue()
        self.results = []
    
    def basic_threading_example(self):
        """Example 1: Basic threading with Thread class."""
        print("\n=== Basic Threading Example ===")
        
        def worker(name, duration):
            """Worker function that simulates some work."""
            logger.info(f"Thread {name} starting")
            time.sleep(duration)
            logger.info(f"Thread {name} finished after {duration}s")
            return f"Result from {name}"
        
        # Create and start threads
        threads = []
        for i in range(5):
            thread = threading.Thread(
                target=worker, 
                args=(f"Worker-{i}", random.uniform(1, 3))
            )
            threads.append(thread)
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
        
        print("All threads completed!")
    
    def thread_synchronization_example(self):
        """Example 2: Thread synchronization with Lock."""
        print("\n=== Thread Synchronization Example ===")
        
        def increment_shared_data(thread_name, iterations):
            """Function that safely increments shared data."""
            for _ in range(iterations):
                with self.lock:  # Acquire lock
                    old_value = self.shared_data
                    time.sleep(0.001)  # Simulate some processing
                    self.shared_data = old_value + 1
                    logger.info(f"{thread_name}: shared_data = {self.shared_data}")
        
        # Reset shared data
        self.shared_data = 0
        
        # Create threads that will modify shared data
        threads = []
        for i in range(3):
            thread = threading.Thread(
                target=increment_shared_data,
                args=(f"Thread-{i}", 5)
            )
            threads.append(thread)
            thread.start()
        
        # Wait for completion
        for thread in threads:
            thread.join()
        
        print(f"Final shared_data value: {self.shared_data}")
    
    def semaphore_example(self):
        """Example 3: Semaphore to limit concurrent access."""
        print("\n=== Semaphore Example ===")
        
        def limited_resource_access(thread_name):
            """Function that accesses a limited resource."""
            logger.info(f"{thread_name} waiting for resource access")
            
            with self.semaphore:  # Acquire semaphore (max 3 concurrent)
                logger.info(f"{thread_name} acquired resource")
                time.sleep(2)  # Simulate resource usage
                logger.info(f"{thread_name} released resource")
        
        # Create more threads than semaphore allows
        threads = []
        for i in range(8):
            thread = threading.Thread(
                target=limited_resource_access,
                args=(f"Thread-{i}",)
            )
            threads.append(thread)
            thread.start()
        
        for thread in threads:
            thread.join()
    
    def event_coordination_example(self):
        """Example 4: Event for thread coordination."""
        print("\n=== Event Coordination Example ===")
        
        def waiter_thread():
            """Thread that waits for an event."""
            logger.info("Waiter thread: waiting for event...")
            self.event.wait()  # Block until event is set
            logger.info("Waiter thread: event received!")
        
        def setter_thread():
            """Thread that sets the event after some time."""
            time.sleep(3)
            logger.info("Setter thread: setting event...")
            self.event.set()
        
        # Create waiter and setter threads
        waiter = threading.Thread(target=waiter_thread)
        setter = threading.Thread(target=setter_thread)
        
        waiter.start()
        setter.start()
        
        waiter.join()
        setter.join()
    
    def condition_variable_example(self):
        """Example 5: Condition variable for producer-consumer pattern."""
        print("\n=== Condition Variable Example ===")
        
        def producer():
            """Producer thread that adds items to queue."""
            for i in range(5):
                with self.condition:
                    self.queue.put(f"Item-{i}")
                    logger.info(f"Produced: Item-{i}")
                    self.condition.notify()  # Notify waiting consumers
                time.sleep(1)
        
        def consumer():
            """Consumer thread that processes items from queue."""
            for _ in range(5):
                with self.condition:
                    while self.queue.empty():
                        logger.info("Consumer: waiting for items...")
                        self.condition.wait()  # Wait for notification
                    
                    item = self.queue.get()
                    logger.info(f"Consumed: {item}")
        
        # Create producer and consumer threads
        producer_thread = threading.Thread(target=producer)
        consumer_thread = threading.Thread(target=consumer)
        
        producer_thread.start()
        consumer_thread.start()
        
        producer_thread.join()
        consumer_thread.join()
    
    def thread_pool_executor_example(self):
        """Example 6: ThreadPoolExecutor for managed thread pools."""
        print("\n=== ThreadPoolExecutor Example ===")
        
        def cpu_bound_task(n):
            """Simulate a CPU-bound task."""
            logger.info(f"Processing task {n}")
            time.sleep(random.uniform(1, 3))
            return f"Result-{n}"
        
        def io_bound_task(url):
            """Simulate an I/O-bound task."""
            logger.info(f"Fetching {url}")
            time.sleep(random.uniform(0.5, 2))
            return f"Data from {url}"
        
        # Example 1: CPU-bound tasks
        print("CPU-bound tasks:")
        with ThreadPoolExecutor(max_workers=3) as executor:
            futures = [executor.submit(cpu_bound_task, i) for i in range(5)]
            for future in as_completed(futures):
                result = future.result()
                logger.info(f"Completed: {result}")
        
        # Example 2: I/O-bound tasks
        print("\nI/O-bound tasks:")
        urls = [f"http://example.com/{i}" for i in range(5)]
        with ThreadPoolExecutor(max_workers=5) as executor:
            future_to_url = {executor.submit(io_bound_task, url): url for url in urls}
            for future in as_completed(future_to_url):
                url = future_to_url[future]
                result = future.result()
                logger.info(f"Completed {url}: {result}")
    
    def thread_safe_counter_example(self):
        """Example 7: Thread-safe counter implementation."""
        print("\n=== Thread-Safe Counter Example ===")
        
        class ThreadSafeCounter:
            """Thread-safe counter using locks."""
            
            def __init__(self):
                self._value = 0
                self._lock = Lock()
            
            def increment(self):
                with self._lock:
                    self._value += 1
                    return self._value
            
            def decrement(self):
                with self._lock:
                    self._value -= 1
                    return self._value
            
            def value(self):
                with self._lock:
                    return self._value
        
        counter = ThreadSafeCounter()
        
        def worker(counter, operations):
            """Worker that performs counter operations."""
            for _ in range(operations):
                if random.choice([True, False]):
                    counter.increment()
                else:
                    counter.decrement()
                time.sleep(0.001)
        
        # Create multiple threads that modify the counter
        threads = []
        for i in range(5):
            thread = threading.Thread(
                target=worker,
                args=(counter, 100)
            )
            threads.append(thread)
            thread.start()
        
        for thread in threads:
            thread.join()
        
        print(f"Final counter value: {counter.value()}")
    
    def daemon_thread_example(self):
        """Example 8: Daemon threads."""
        print("\n=== Daemon Thread Example ===")
        
        def background_task():
            """Background task that runs continuously."""
            while True:
                logger.info("Background task running...")
                time.sleep(1)
        
        def main_task():
            """Main task that runs for a limited time."""
            logger.info("Main task starting...")
            time.sleep(5)
            logger.info("Main task completed!")
        
        # Create daemon thread (will terminate when main thread ends)
        daemon_thread = threading.Thread(target=background_task, daemon=True)
        daemon_thread.start()
        
        # Run main task
        main_task()
        
        print("Main thread ending - daemon thread will be terminated")
    
    def run_all_examples(self):
        """Run all threading examples."""
        print("Starting comprehensive threading examples...")
        
        self.basic_threading_example()
        self.thread_synchronization_example()
        self.semaphore_example()
        self.event_coordination_example()
        self.condition_variable_example()
        self.thread_pool_executor_example()
        self.thread_safe_counter_example()
        self.daemon_thread_example()
        
        print("\nAll threading examples completed!")


def main():
    """Main function to run threading examples."""
    examples = ThreadingExamples()
    examples.run_all_examples()


if __name__ == "__main__":
    main()
