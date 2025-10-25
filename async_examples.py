"""
Comprehensive Python Async/Await Examples

This module demonstrates various async/await approaches in Python:
1. Basic async/await syntax
2. Async context managers and iterators
3. Async generators
4. Async synchronization primitives
5. Async I/O operations
6. Async task management and cancellation
7. Async performance optimization
8. Real-world async patterns
"""

import asyncio
import aiohttp
import aiofiles
import time
import random
import logging
from typing import AsyncGenerator, List, Dict, Any
from asyncio import Lock, Semaphore, Event, Condition, Queue
from asyncio import create_task, gather, wait, as_completed
from asyncio import sleep, run, get_event_loop
import json

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class AsyncExamples:
    """Comprehensive async/await examples and demonstrations."""
    
    def __init__(self):
        self.shared_data = 0
        self.lock = Lock()
        self.semaphore = Semaphore(3)
        self.event = Event()
        self.condition = Condition()
        self.queue = Queue()
    
    async def basic_async_example(self):
        """Example 1: Basic async/await syntax."""
        print("\n=== Basic Async/Await Example ===")
        
        async def async_worker(name: str, duration: float) -> str:
            """Async worker function."""
            logger.info(f"Async worker {name} starting")
            await asyncio.sleep(duration)
            logger.info(f"Async worker {name} finished")
            return f"Result from {name}"
        
        # Create and run async tasks
        tasks = [
            async_worker(f"Worker-{i}", random.uniform(1, 3))
            for i in range(5)
        ]
        
        # Run all tasks concurrently
        results = await asyncio.gather(*tasks)
        print(f"All tasks completed: {results}")
    
    async def async_context_manager_example(self):
        """Example 2: Async context managers."""
        print("\n=== Async Context Manager Example ===")
        
        class AsyncResource:
            """Example async context manager."""
            
            def __init__(self, name: str):
                self.name = name
                self.is_open = False
            
            async def __aenter__(self):
                logger.info(f"Opening resource {self.name}")
                await asyncio.sleep(0.5)  # Simulate async setup
                self.is_open = True
                return self
            
            async def __aexit__(self, exc_type, exc_val, exc_tb):
                logger.info(f"Closing resource {self.name}")
                await asyncio.sleep(0.2)  # Simulate async cleanup
                self.is_open = False
        
        # Use async context manager
        async with AsyncResource("Database") as db:
            logger.info(f"Using resource: {db.name}")
            await asyncio.sleep(1)
            logger.info("Resource operation completed")
        
        print("Context manager cleanup completed")
    
    async def async_generator_example(self):
        """Example 3: Async generators."""
        print("\n=== Async Generator Example ===")
        
        async def async_number_generator(n: int) -> AsyncGenerator[int, None]:
            """Async generator that yields numbers."""
            for i in range(n):
                await asyncio.sleep(0.1)  # Simulate async work
                yield i * i
        
        async def async_fibonacci(n: int) -> AsyncGenerator[int, None]:
            """Async generator for Fibonacci numbers."""
            a, b = 0, 1
            for _ in range(n):
                yield a
                await asyncio.sleep(0.05)  # Simulate async work
                a, b = b, a + b
        
        # Use async generators
        print("Squares:")
        async for square in async_number_generator(5):
            print(f"Square: {square}")
        
        print("\nFibonacci:")
        async for fib in async_fibonacci(8):
            print(f"Fibonacci: {fib}")
    
    async def async_synchronization_example(self):
        """Example 4: Async synchronization primitives."""
        print("\n=== Async Synchronization Example ===")
        
        async def async_worker_with_lock(worker_id: int, iterations: int):
            """Worker that uses async lock."""
            for _ in range(iterations):
                async with self.lock:
                    old_value = self.shared_data
                    await asyncio.sleep(0.01)  # Simulate async work
                    self.shared_data = old_value + 1
                    logger.info(f"Worker {worker_id}: shared_data = {self.shared_data}")
        
        async def async_worker_with_semaphore(worker_id: int):
            """Worker that uses async semaphore."""
            logger.info(f"Worker {worker_id} waiting for semaphore")
            
            async with self.semaphore:
                logger.info(f"Worker {worker_id} acquired semaphore")
                await asyncio.sleep(1)  # Simulate work
                logger.info(f"Worker {worker_id} released semaphore")
        
        # Reset shared data
        self.shared_data = 0
        
        # Test async lock
        print("Testing async lock:")
        tasks = [
            async_worker_with_lock(i, 3)
            for i in range(3)
        ]
        await asyncio.gather(*tasks)
        print(f"Final shared_data: {self.shared_data}")
        
        # Test async semaphore
        print("\nTesting async semaphore:")
        tasks = [
            async_worker_with_semaphore(i)
            for i in range(8)
        ]
        await asyncio.gather(*tasks)
    
    async def async_event_example(self):
        """Example 5: Async event coordination."""
        print("\n=== Async Event Example ===")
        
        async def async_waiter():
            """Async function that waits for an event."""
            logger.info("Async waiter: waiting for event...")
            await self.event.wait()
            logger.info("Async waiter: event received!")
        
        async def async_setter():
            """Async function that sets the event."""
            await asyncio.sleep(2)
            logger.info("Async setter: setting event...")
            self.event.set()
        
        # Create waiter and setter tasks
        waiter_task = asyncio.create_task(async_waiter())
        setter_task = asyncio.create_task(async_setter())
        
        # Wait for both tasks
        await asyncio.gather(waiter_task, setter_task)
    
    async def async_io_operations_example(self):
        """Example 6: Async I/O operations."""
        print("\n=== Async I/O Operations Example ===")
        
        async def async_file_operations():
            """Example of async file operations."""
            # Simulate async file writing
            content = "Hello, Async World!\n" * 100
            
            # Simulate async file write
            logger.info("Writing file asynchronously...")
            await asyncio.sleep(0.5)  # Simulate I/O delay
            
            # Simulate async file read
            logger.info("Reading file asynchronously...")
            await asyncio.sleep(0.3)  # Simulate I/O delay
            
            return f"Processed {len(content)} characters"
        
        async def async_network_operations():
            """Example of async network operations."""
            # Simulate async HTTP requests
            urls = [
                "https://httpbin.org/delay/1",
                "https://httpbin.org/delay/2",
                "https://httpbin.org/delay/1"
            ]
            
            async def fetch_url(session, url):
                logger.info(f"Fetching {url}")
                # Simulate network delay
                await asyncio.sleep(random.uniform(0.5, 2))
                return f"Response from {url}"
            
            # Simulate concurrent requests
            results = []
            for url in urls:
                result = await fetch_url(None, url)  # Simplified for demo
                results.append(result)
            
            return results
        
        # Run async I/O operations
        file_result = await async_file_operations()
        network_results = await async_network_operations()
        
        print(f"File operation: {file_result}")
        print(f"Network operations: {network_results}")
    
    async def async_task_management_example(self):
        """Example 7: Async task management and cancellation."""
        print("\n=== Async Task Management Example ===")
        
        async def cancellable_worker(worker_id: int, duration: float):
            """Worker that can be cancelled."""
            try:
                logger.info(f"Worker {worker_id} starting")
                await asyncio.sleep(duration)
                logger.info(f"Worker {worker_id} completed")
                return f"Result from worker {worker_id}"
            except asyncio.CancelledError:
                logger.info(f"Worker {worker_id} was cancelled")
                raise
        
        async def timeout_example():
            """Example of task timeout."""
            try:
                result = await asyncio.wait_for(
                    cancellable_worker(1, 5),  # 5 second task
                    timeout=2  # 2 second timeout
                )
                return result
            except asyncio.TimeoutError:
                logger.info("Task timed out")
                return "Timeout"
        
        async def cancellation_example():
            """Example of task cancellation."""
            # Create long-running tasks
            tasks = [
                asyncio.create_task(cancellable_worker(i, 3))
                for i in range(3)
            ]
            
            # Cancel one task after 1 second
            await asyncio.sleep(1)
            tasks[1].cancel()
            
            # Wait for remaining tasks
            results = []
            for task in tasks:
                try:
                    result = await task
                    results.append(result)
                except asyncio.CancelledError:
                    results.append("Cancelled")
            
            return results
        
        # Test timeout
        timeout_result = await timeout_example()
        print(f"Timeout result: {timeout_result}")
        
        # Test cancellation
        cancellation_results = await cancellation_example()
        print(f"Cancellation results: {cancellation_results}")
    
    async def async_producer_consumer_example(self):
        """Example 8: Async producer-consumer pattern."""
        print("\n=== Async Producer-Consumer Example ===")
        
        async def async_producer(queue: Queue, item_count: int):
            """Async producer that adds items to queue."""
            for i in range(item_count):
                item = f"Item-{i}"
                await queue.put(item)
                logger.info(f"Produced: {item}")
                await asyncio.sleep(0.2)
            
            # Signal end of production
            await queue.put(None)
        
        async def async_consumer(queue: Queue, consumer_id: str):
            """Async consumer that processes items from queue."""
            while True:
                item = await queue.get()
                if item is None:  # Check for end signal
                    break
                logger.info(f"Consumer {consumer_id} consumed: {item}")
                await asyncio.sleep(0.3)  # Simulate processing
                queue.task_done()
        
        # Create queue and tasks
        queue = Queue()
        
        # Create producer and consumer tasks
        producer_task = asyncio.create_task(async_producer(queue, 5))
        consumer_tasks = [
            asyncio.create_task(async_consumer(queue, f"C{i}"))
            for i in range(2)
        ]
        
        # Wait for producer to finish
        await producer_task
        
        # Wait for all items to be processed
        await queue.join()
        
        # Cancel consumer tasks
        for task in consumer_tasks:
            task.cancel()
        
        await asyncio.gather(*consumer_tasks, return_exceptions=True)
    
    async def async_performance_optimization_example(self):
        """Example 9: Async performance optimization."""
        print("\n=== Async Performance Optimization Example ===")
        
        async def cpu_bound_task(n: int) -> int:
            """CPU-bound task (not ideal for async)."""
            return sum(i * i for i in range(n))
        
        async def io_bound_task(duration: float) -> str:
            """I/O-bound task (ideal for async)."""
            await asyncio.sleep(duration)
            return f"Completed after {duration}s"
        
        async def mixed_task(n: int, duration: float) -> tuple:
            """Task that mixes CPU and I/O work."""
            # I/O work (good for async)
            await asyncio.sleep(duration)
            
            # CPU work (not ideal for async)
            result = sum(i * i for i in range(n))
            
            return result, duration
        
        # Test I/O-bound tasks (should be fast with async)
        print("I/O-bound tasks:")
        start_time = time.time()
        tasks = [io_bound_task(0.5) for _ in range(10)]
        results = await asyncio.gather(*tasks)
        io_time = time.time() - start_time
        print(f"Async I/O time: {io_time:.2f}s")
        
        # Test mixed tasks
        print("\nMixed tasks:")
        start_time = time.time()
        tasks = [mixed_task(1000, 0.1) for _ in range(5)]
        results = await asyncio.gather(*tasks)
        mixed_time = time.time() - start_time
        print(f"Async mixed time: {mixed_time:.2f}s")
    
    async def async_real_world_example(self):
        """Example 10: Real-world async patterns."""
        print("\n=== Real-World Async Example ===")
        
        class AsyncDataProcessor:
            """Example async data processor."""
            
            def __init__(self):
                self.processed_count = 0
                self.lock = Lock()
            
            async def process_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
                """Process data asynchronously."""
                # Simulate async data processing
                await asyncio.sleep(random.uniform(0.1, 0.5))
                
                # Update counter safely
                async with self.lock:
                    self.processed_count += 1
                
                return {
                    "id": data["id"],
                    "processed": True,
                    "timestamp": time.time()
                }
            
            async def batch_process(self, data_list: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
                """Process multiple data items concurrently."""
                tasks = [self.process_data(data) for data in data_list]
                return await asyncio.gather(*tasks)
        
        # Create processor and sample data
        processor = AsyncDataProcessor()
        sample_data = [
            {"id": i, "value": f"data_{i}"}
            for i in range(10)
        ]
        
        # Process data concurrently
        start_time = time.time()
        results = await processor.batch_process(sample_data)
        processing_time = time.time() - start_time
        
        print(f"Processed {len(results)} items in {processing_time:.2f}s")
        print(f"Total processed count: {processor.processed_count}")
    
    async def run_all_examples(self):
        """Run all async examples."""
        print("Starting comprehensive async examples...")
        
        await self.basic_async_example()
        await self.async_context_manager_example()
        await self.async_generator_example()
        await self.async_synchronization_example()
        await self.async_event_example()
        await self.async_io_operations_example()
        await self.async_task_management_example()
        await self.async_producer_consumer_example()
        await self.async_performance_optimization_example()
        await self.async_real_world_example()
        
        print("\nAll async examples completed!")


async def main():
    """Main async function to run examples."""
    examples = AsyncExamples()
    await examples.run_all_examples()


def run_async_examples():
    """Synchronous wrapper to run async examples."""
    asyncio.run(main())


if __name__ == "__main__":
    run_async_examples()
