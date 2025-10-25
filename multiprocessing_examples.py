"""
Comprehensive Python Multiprocessing Examples

This module demonstrates various multiprocessing approaches in Python:
1. Basic Process creation and management
2. Process communication (Queue, Pipe, Manager)
3. Process synchronization (Lock, Semaphore, Event)
4. Process pools for parallel execution
5. Shared memory and data sharing
6. Process vs Thread performance comparison
"""

import multiprocessing
import time
import random
import math
import os
from multiprocessing import Process, Queue, Pipe, Manager, Lock, Semaphore, Event, Value, Array
from multiprocessing import Pool, cpu_count
from concurrent.futures import ProcessPoolExecutor, as_completed
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(processName)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class MultiprocessingExamples:
    """Comprehensive multiprocessing examples and demonstrations."""
    
    def __init__(self):
        self.shared_value = None
        self.shared_array = None
        self.manager = None
    
    def basic_process_example(self):
        """Example 1: Basic process creation and management."""
        print("\n=== Basic Process Example ===")
        
        def worker_process(name, duration):
            """Worker function that runs in a separate process."""
            logger.info(f"Process {name} (PID: {os.getpid()}) starting")
            time.sleep(duration)
            logger.info(f"Process {name} (PID: {os.getpid()}) finished")
            return f"Result from {name}"
        
        # Create and start processes
        processes = []
        for i in range(3):
            process = Process(
                target=worker_process,
                args=(f"Worker-{i}", random.uniform(2, 4))
            )
            processes.append(process)
            process.start()
        
        # Wait for all processes to complete
        for process in processes:
            process.join()
        
        print("All processes completed!")
    
    def process_communication_queue(self):
        """Example 2: Process communication using Queue."""
        print("\n=== Process Communication with Queue ===")
        
        def producer(queue, items):
            """Producer process that puts items in queue."""
            for item in items:
                logger.info(f"Producing: {item}")
                queue.put(item)
                time.sleep(0.5)
            queue.put(None)  # Sentinel value to signal end
        
        def consumer(queue, consumer_id):
            """Consumer process that gets items from queue."""
            while True:
                item = queue.get()
                if item is None:  # Check for sentinel
                    break
                logger.info(f"Consumer {consumer_id} consumed: {item}")
                time.sleep(0.3)
        
        # Create queue for communication
        queue = Queue()
        
        # Create producer and consumer processes
        items = [f"Item-{i}" for i in range(5)]
        producer_process = Process(target=producer, args=(queue, items))
        consumer_process = Process(target=consumer, args=(queue, "C1"))
        
        producer_process.start()
        consumer_process.start()
        
        producer_process.join()
        consumer_process.join()
    
    def process_communication_pipe(self):
        """Example 3: Process communication using Pipe."""
        print("\n=== Process Communication with Pipe ===")
        
        def sender(conn):
            """Process that sends data through pipe."""
            for i in range(5):
                message = f"Message-{i}"
                logger.info(f"Sending: {message}")
                conn.send(message)
                time.sleep(0.5)
            conn.close()
        
        def receiver(conn):
            """Process that receives data through pipe."""
            while True:
                try:
                    message = conn.recv()
                    logger.info(f"Received: {message}")
                except EOFError:
                    break
            conn.close()
        
        # Create pipe
        parent_conn, child_conn = Pipe()
        
        # Create sender and receiver processes
        sender_process = Process(target=sender, args=(child_conn,))
        receiver_process = Process(target=receiver, args=(parent_conn,))
        
        sender_process.start()
        receiver_process.start()
        
        sender_process.join()
        receiver_process.join()
    
    def shared_memory_example(self):
        """Example 4: Shared memory using Value and Array."""
        print("\n=== Shared Memory Example ===")
        
        def worker_with_shared_memory(shared_value, shared_array, worker_id):
            """Worker that modifies shared memory."""
            logger.info(f"Worker {worker_id} starting")
            
            # Modify shared value
            with shared_value.get_lock():
                shared_value.value += 1
                logger.info(f"Worker {worker_id}: shared_value = {shared_value.value}")
            
            # Modify shared array
            for i in range(len(shared_array)):
                shared_array[i] += worker_id
                time.sleep(0.1)
            
            logger.info(f"Worker {worker_id} finished")
        
        # Create shared memory objects
        shared_value = Value('i', 0)  # Integer value
        shared_array = Array('i', [0, 0, 0, 0, 0])  # Integer array
        
        # Create processes that modify shared memory
        processes = []
        for i in range(3):
            process = Process(
                target=worker_with_shared_memory,
                args=(shared_value, shared_array, i)
            )
            processes.append(process)
            process.start()
        
        # Wait for all processes
        for process in processes:
            process.join()
        
        print(f"Final shared_value: {shared_value.value}")
        print(f"Final shared_array: {list(shared_array)}")
    
    def manager_example(self):
        """Example 5: Manager for complex shared objects."""
        print("\n=== Manager Example ===")
        
        def worker_with_manager(shared_dict, shared_list, worker_id):
            """Worker that modifies manager objects."""
            logger.info(f"Worker {worker_id} starting")
            
            # Modify shared dictionary
            shared_dict[f"worker_{worker_id}"] = f"data_{worker_id}"
            
            # Modify shared list
            shared_list.append(f"item_{worker_id}")
            
            time.sleep(1)
            logger.info(f"Worker {worker_id} finished")
        
        # Create manager
        with Manager() as manager:
            shared_dict = manager.dict()
            shared_list = manager.list()
            
            # Create processes
            processes = []
            for i in range(3):
                process = Process(
                    target=worker_with_manager,
                    args=(shared_dict, shared_list, i)
                )
                processes.append(process)
                process.start()
            
            # Wait for all processes
            for process in processes:
                process.join()
            
            print(f"Final shared_dict: {dict(shared_dict)}")
            print(f"Final shared_list: {list(shared_list)}")
    
    def process_synchronization_example(self):
        """Example 6: Process synchronization with Lock and Semaphore."""
        print("\n=== Process Synchronization Example ===")
        
        def synchronized_worker(lock, semaphore, worker_id):
            """Worker that uses synchronization primitives."""
            logger.info(f"Worker {worker_id} waiting for semaphore")
            
            with semaphore:  # Acquire semaphore
                logger.info(f"Worker {worker_id} acquired semaphore")
                
                with lock:  # Acquire lock for critical section
                    logger.info(f"Worker {worker_id} in critical section")
                    time.sleep(1)
                    logger.info(f"Worker {worker_id} leaving critical section")
                
                logger.info(f"Worker {worker_id} released semaphore")
        
        # Create synchronization objects
        lock = Lock()
        semaphore = Semaphore(2)  # Allow max 2 processes
        
        # Create processes
        processes = []
        for i in range(5):
            process = Process(
                target=synchronized_worker,
                args=(lock, semaphore, i)
            )
            processes.append(process)
            process.start()
        
        for process in processes:
            process.join()
    
    def process_pool_example(self):
        """Example 7: Process pool for parallel execution."""
        print("\n=== Process Pool Example ===")
        
        def cpu_intensive_task(n):
            """CPU-intensive task for parallel execution."""
            logger.info(f"Processing task {n}")
            result = sum(i * i for i in range(n))
            time.sleep(1)  # Simulate work
            return result
        
        # Using multiprocessing.Pool
        print("Using multiprocessing.Pool:")
        with Pool(processes=cpu_count()) as pool:
            tasks = [10000, 20000, 30000, 40000]
            results = pool.map(cpu_intensive_task, tasks)
            print(f"Results: {results}")
        
        # Using ProcessPoolExecutor
        print("\nUsing ProcessPoolExecutor:")
        with ProcessPoolExecutor(max_workers=cpu_count()) as executor:
            futures = [executor.submit(cpu_intensive_task, task) for task in tasks]
            results = [future.result() for future in futures]
            print(f"Results: {results}")
    
    def process_vs_thread_performance(self):
        """Example 8: Performance comparison between processes and threads."""
        print("\n=== Process vs Thread Performance ===")
        
        def cpu_bound_task(n):
            """CPU-bound task for performance testing."""
            return sum(i * i for i in range(n))
        
        def io_bound_task(duration):
            """I/O-bound task for performance testing."""
            time.sleep(duration)
            return f"Completed after {duration}s"
        
        # Test parameters
        task_count = 4
        cpu_task_size = 100000
        io_duration = 1
        
        print("CPU-bound tasks (processes should be faster):")
        
        # Test with processes
        start_time = time.time()
        with ProcessPoolExecutor(max_workers=cpu_count()) as executor:
            futures = [executor.submit(cpu_bound_task, cpu_task_size) for _ in range(task_count)]
            process_results = [future.result() for future in futures]
        process_time = time.time() - start_time
        print(f"Process time: {process_time:.2f}s")
        
        # Test with threads
        start_time = time.time()
        with ThreadPoolExecutor(max_workers=cpu_count()) as executor:
            futures = [executor.submit(cpu_bound_task, cpu_task_size) for _ in range(task_count)]
            thread_results = [future.result() for future in futures]
        thread_time = time.time() - start_time
        print(f"Thread time: {thread_time:.2f}s")
        
        print(f"\nProcess speedup: {thread_time/process_time:.2f}x")
        
        print("\nI/O-bound tasks (threads should be faster):")
        
        # Test I/O-bound with processes
        start_time = time.time()
        with ProcessPoolExecutor(max_workers=task_count) as executor:
            futures = [executor.submit(io_bound_task, io_duration) for _ in range(task_count)]
            process_results = [future.result() for future in futures]
        process_time = time.time() - start_time
        print(f"Process time: {process_time:.2f}s")
        
        # Test I/O-bound with threads
        start_time = time.time()
        with ThreadPoolExecutor(max_workers=task_count) as executor:
            futures = [executor.submit(io_bound_task, io_duration) for _ in range(task_count)]
            thread_results = [future.result() for future in futures]
        thread_time = time.time() - start_time
        print(f"Thread time: {thread_time:.2f}s")
    
    def process_monitoring_example(self):
        """Example 9: Process monitoring and management."""
        print("\n=== Process Monitoring Example ===")
        
        def monitored_worker(worker_id, duration):
            """Worker that can be monitored."""
            logger.info(f"Worker {worker_id} (PID: {os.getpid()}) starting")
            start_time = time.time()
            
            while time.time() - start_time < duration:
                logger.info(f"Worker {worker_id} still working...")
                time.sleep(0.5)
            
            logger.info(f"Worker {worker_id} completed")
            return f"Result from worker {worker_id}"
        
        # Create processes
        processes = []
        for i in range(3):
            process = Process(
                target=monitored_worker,
                args=(i, 3)
            )
            processes.append(process)
            process.start()
        
        # Monitor processes
        while any(p.is_alive() for p in processes):
            alive_processes = [p for p in processes if p.is_alive()]
            logger.info(f"Alive processes: {len(alive_processes)}")
            time.sleep(1)
        
        # Wait for all processes to complete
        for process in processes:
            process.join()
        
        print("All processes completed!")
    
    def run_all_examples(self):
        """Run all multiprocessing examples."""
        print("Starting comprehensive multiprocessing examples...")
        
        self.basic_process_example()
        self.process_communication_queue()
        self.process_communication_pipe()
        self.shared_memory_example()
        self.manager_example()
        self.process_synchronization_example()
        self.process_pool_example()
        self.process_vs_thread_performance()
        self.process_monitoring_example()
        
        print("\nAll multiprocessing examples completed!")


def main():
    """Main function to run multiprocessing examples."""
    # Required for multiprocessing on Windows
    multiprocessing.freeze_support()
    
    examples = MultiprocessingExamples()
    examples.run_all_examples()


if __name__ == "__main__":
    main()
