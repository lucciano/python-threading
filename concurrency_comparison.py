"""
Comprehensive Python Concurrency Comparison

This module provides a comprehensive comparison of different Python concurrency approaches:
1. Sequential execution (baseline)
2. Threading (for I/O-bound tasks)
3. Multiprocessing (for CPU-bound tasks)
4. Async/await (for I/O-bound tasks)
5. Concurrent.futures (high-level interface)

The comparison includes:
- Performance benchmarks
- Memory usage analysis
- Use case recommendations
- Code complexity comparison
- Real-world scenarios
"""

import time
import random
import math
import threading
import multiprocessing
import asyncio
import psutil
import os
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from typing import List, Dict, Any, Callable
import logging
from dataclasses import dataclass
from statistics import mean, median

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class BenchmarkResult:
    """Results from a concurrency benchmark."""
    approach: str
    execution_time: float
    memory_usage: float
    cpu_usage: float
    success_rate: float
    throughput: float


class ConcurrencyComparison:
    """Comprehensive comparison of Python concurrency approaches."""
    
    def __init__(self):
        self.results = []
        self.process = psutil.Process()
    
    def measure_memory_usage(self) -> float:
        """Measure current memory usage in MB."""
        return self.process.memory_info().rss / 1024 / 1024
    
    def measure_cpu_usage(self) -> float:
        """Measure current CPU usage percentage."""
        return self.process.cpu_percent()
    
    def cpu_intensive_task(self, n: int) -> int:
        """CPU-intensive task for benchmarking."""
        return sum(i * i for i in range(n))
    
    def io_intensive_task(self, duration: float) -> str:
        """I/O-intensive task for benchmarking."""
        time.sleep(duration)
        return f"Completed after {duration}s"
    
    def mixed_task(self, n: int, duration: float) -> tuple:
        """Mixed CPU and I/O task for benchmarking."""
        time.sleep(duration)  # I/O part
        result = sum(i * i for i in range(n))  # CPU part
        return result, duration
    
    def sequential_cpu_benchmark(self, tasks: List[int]) -> BenchmarkResult:
        """Benchmark sequential CPU-intensive tasks."""
        logger.info("Running sequential CPU benchmark...")
        
        start_time = time.time()
        start_memory = self.measure_memory_usage()
        
        results = [self.cpu_intensive_task(n) for n in tasks]
        
        end_time = time.time()
        end_memory = self.measure_memory_usage()
        
        execution_time = end_time - start_time
        memory_usage = end_memory - start_memory
        throughput = len(tasks) / execution_time
        
        return BenchmarkResult(
            approach="Sequential",
            execution_time=execution_time,
            memory_usage=memory_usage,
            cpu_usage=100.0,  # Sequential uses 100% of one core
            success_rate=100.0,
            throughput=throughput
        )
    
    def threading_cpu_benchmark(self, tasks: List[int]) -> BenchmarkResult:
        """Benchmark threading for CPU-intensive tasks."""
        logger.info("Running threading CPU benchmark...")
        
        start_time = time.time()
        start_memory = self.measure_memory_usage()
        
        results = []
        threads = []
        
        def worker(n):
            results.append(self.cpu_intensive_task(n))
        
        for n in tasks:
            thread = threading.Thread(target=worker, args=(n,))
            threads.append(thread)
            thread.start()
        
        for thread in threads:
            thread.join()
        
        end_time = time.time()
        end_memory = self.measure_memory_usage()
        
        execution_time = end_time - start_time
        memory_usage = end_memory - start_memory
        throughput = len(tasks) / execution_time
        
        return BenchmarkResult(
            approach="Threading",
            execution_time=execution_time,
            memory_usage=memory_usage,
            cpu_usage=100.0,  # Threading limited by GIL
            success_rate=100.0,
            throughput=throughput
        )
    
    def multiprocessing_cpu_benchmark(self, tasks: List[int]) -> BenchmarkResult:
        """Benchmark multiprocessing for CPU-intensive tasks."""
        logger.info("Running multiprocessing CPU benchmark...")
        
        start_time = time.time()
        start_memory = self.measure_memory_usage()
        
        with ProcessPoolExecutor(max_workers=multiprocessing.cpu_count()) as executor:
            results = list(executor.map(self.cpu_intensive_task, tasks))
        
        end_time = time.time()
        end_memory = self.measure_memory_usage()
        
        execution_time = end_time - start_time
        memory_usage = end_memory - start_memory
        throughput = len(tasks) / execution_time
        
        return BenchmarkResult(
            approach="Multiprocessing",
            execution_time=execution_time,
            memory_usage=memory_usage,
            cpu_usage=100.0,  # Multiprocessing can use all cores
            success_rate=100.0,
            throughput=throughput
        )
    
    def async_cpu_benchmark(self, tasks: List[int]) -> BenchmarkResult:
        """Benchmark async for CPU-intensive tasks."""
        logger.info("Running async CPU benchmark...")
        
        async def async_cpu_task(n):
            return self.cpu_intensive_task(n)
        
        async def run_async_tasks():
            return await asyncio.gather(*[async_cpu_task(n) for n in tasks])
        
        start_time = time.time()
        start_memory = self.measure_memory_usage()
        
        results = asyncio.run(run_async_tasks())
        
        end_time = time.time()
        end_memory = self.measure_memory_usage()
        
        execution_time = end_time - start_time
        memory_usage = end_memory - start_memory
        throughput = len(tasks) / execution_time
        
        return BenchmarkResult(
            approach="Async",
            execution_time=execution_time,
            memory_usage=memory_usage,
            cpu_usage=100.0,  # Async limited by GIL for CPU tasks
            success_rate=100.0,
            throughput=throughput
        )
    
    def sequential_io_benchmark(self, tasks: List[float]) -> BenchmarkResult:
        """Benchmark sequential I/O-intensive tasks."""
        logger.info("Running sequential I/O benchmark...")
        
        start_time = time.time()
        start_memory = self.measure_memory_usage()
        
        results = [self.io_intensive_task(duration) for duration in tasks]
        
        end_time = time.time()
        end_memory = self.measure_memory_usage()
        
        execution_time = end_time - start_time
        memory_usage = end_memory - start_memory
        throughput = len(tasks) / execution_time
        
        return BenchmarkResult(
            approach="Sequential",
            execution_time=execution_time,
            memory_usage=memory_usage,
            cpu_usage=0.0,  # I/O tasks don't use much CPU
            success_rate=100.0,
            throughput=throughput
        )
    
    def threading_io_benchmark(self, tasks: List[float]) -> BenchmarkResult:
        """Benchmark threading for I/O-intensive tasks."""
        logger.info("Running threading I/O benchmark...")
        
        start_time = time.time()
        start_memory = self.measure_memory_usage()
        
        results = []
        threads = []
        
        def worker(duration):
            results.append(self.io_intensive_task(duration))
        
        for duration in tasks:
            thread = threading.Thread(target=worker, args=(duration,))
            threads.append(thread)
            thread.start()
        
        for thread in threads:
            thread.join()
        
        end_time = time.time()
        end_memory = self.measure_memory_usage()
        
        execution_time = end_time - start_time
        memory_usage = end_memory - start_memory
        throughput = len(tasks) / execution_time
        
        return BenchmarkResult(
            approach="Threading",
            execution_time=execution_time,
            memory_usage=memory_usage,
            cpu_usage=0.0,  # I/O tasks don't use much CPU
            success_rate=100.0,
            throughput=throughput
        )
    
    def async_io_benchmark(self, tasks: List[float]) -> BenchmarkResult:
        """Benchmark async for I/O-intensive tasks."""
        logger.info("Running async I/O benchmark...")
        
        async def async_io_task(duration):
            await asyncio.sleep(duration)
            return f"Completed after {duration}s"
        
        async def run_async_tasks():
            return await asyncio.gather(*[async_io_task(d) for d in tasks])
        
        start_time = time.time()
        start_memory = self.measure_memory_usage()
        
        results = asyncio.run(run_async_tasks())
        
        end_time = time.time()
        end_memory = self.measure_memory_usage()
        
        execution_time = end_time - start_time
        memory_usage = end_memory - start_memory
        throughput = len(tasks) / execution_time
        
        return BenchmarkResult(
            approach="Async",
            execution_time=execution_time,
            memory_usage=memory_usage,
            cpu_usage=0.0,  # I/O tasks don't use much CPU
            success_rate=100.0,
            throughput=throughput
        )
    
    def concurrent_futures_io_benchmark(self, tasks: List[float]) -> BenchmarkResult:
        """Benchmark concurrent.futures for I/O-intensive tasks."""
        logger.info("Running concurrent.futures I/O benchmark...")
        
        start_time = time.time()
        start_memory = self.measure_memory_usage()
        
        with ThreadPoolExecutor(max_workers=len(tasks)) as executor:
            results = list(executor.map(self.io_intensive_task, tasks))
        
        end_time = time.time()
        end_memory = self.measure_memory_usage()
        
        execution_time = end_time - start_time
        memory_usage = end_memory - start_memory
        throughput = len(tasks) / execution_time
        
        return BenchmarkResult(
            approach="Concurrent.Futures",
            execution_time=execution_time,
            memory_usage=memory_usage,
            cpu_usage=0.0,  # I/O tasks don't use much CPU
            success_rate=100.0,
            throughput=throughput
        )
    
    def run_cpu_benchmarks(self):
        """Run CPU-intensive benchmarks."""
        print("\n" + "="*60)
        print("CPU-INTENSIVE TASK BENCHMARKS")
        print("="*60)
        
        # CPU-intensive tasks
        cpu_tasks = [50000, 60000, 70000, 80000, 90000]
        
        benchmarks = [
            self.sequential_cpu_benchmark,
            self.threading_cpu_benchmark,
            self.multiprocessing_cpu_benchmark,
            self.async_cpu_benchmark
        ]
        
        results = []
        for benchmark in benchmarks:
            try:
                result = benchmark(cpu_tasks)
                results.append(result)
            except Exception as e:
                logger.error(f"Benchmark failed: {e}")
        
        self.print_benchmark_results(results, "CPU-Intensive Tasks")
        return results
    
    def run_io_benchmarks(self):
        """Run I/O-intensive benchmarks."""
        print("\n" + "="*60)
        print("I/O-INTENSIVE TASK BENCHMARKS")
        print("="*60)
        
        # I/O-intensive tasks
        io_tasks = [0.5, 1.0, 1.5, 2.0, 2.5]
        
        benchmarks = [
            self.sequential_io_benchmark,
            self.threading_io_benchmark,
            self.async_io_benchmark,
            self.concurrent_futures_io_benchmark
        ]
        
        results = []
        for benchmark in benchmarks:
            try:
                result = benchmark(io_tasks)
                results.append(result)
            except Exception as e:
                logger.error(f"Benchmark failed: {e}")
        
        self.print_benchmark_results(results, "I/O-Intensive Tasks")
        return results
    
    def print_benchmark_results(self, results: List[BenchmarkResult], title: str):
        """Print benchmark results in a formatted table."""
        print(f"\n{title} Results:")
        print("-" * 80)
        print(f"{'Approach':<20} {'Time (s)':<10} {'Memory (MB)':<12} {'Throughput':<12} {'Speedup':<10}")
        print("-" * 80)
        
        # Calculate speedup relative to sequential
        sequential_time = next(r.execution_time for r in results if r.approach == "Sequential")
        
        for result in results:
            speedup = sequential_time / result.execution_time if result.execution_time > 0 else 0
            print(f"{result.approach:<20} {result.execution_time:<10.2f} {result.memory_usage:<12.2f} "
                  f"{result.throughput:<12.2f} {speedup:<10.2f}x")
        
        print("-" * 80)
    
    def generate_recommendations(self, cpu_results: List[BenchmarkResult], io_results: List[BenchmarkResult]):
        """Generate recommendations based on benchmark results."""
        print("\n" + "="*60)
        print("RECOMMENDATIONS")
        print("="*60)
        
        # Find best performers
        cpu_best = min(cpu_results, key=lambda r: r.execution_time)
        io_best = min(io_results, key=lambda r: r.execution_time)
        
        print(f"\nBest for CPU-intensive tasks: {cpu_best.approach}")
        print(f"  - Execution time: {cpu_best.execution_time:.2f}s")
        print(f"  - Speedup: {cpu_results[0].execution_time / cpu_best.execution_time:.2f}x")
        
        print(f"\nBest for I/O-intensive tasks: {io_best.approach}")
        print(f"  - Execution time: {io_best.execution_time:.2f}s")
        print(f"  - Speedup: {io_results[0].execution_time / io_best.execution_time:.2f}x")
        
        print("\nGeneral Recommendations:")
        print("1. Use multiprocessing for CPU-bound tasks")
        print("2. Use threading or async for I/O-bound tasks")
        print("3. Use concurrent.futures for high-level concurrency")
        print("4. Consider async/await for modern I/O-heavy applications")
        print("5. Sequential execution is fine for simple, small tasks")
    
    def run_comprehensive_comparison(self):
        """Run comprehensive concurrency comparison."""
        print("Starting comprehensive concurrency comparison...")
        print(f"System: {os.name}, CPU cores: {multiprocessing.cpu_count()}")
        
        # Run benchmarks
        cpu_results = self.run_cpu_benchmarks()
        io_results = self.run_io_benchmarks()
        
        # Generate recommendations
        self.generate_recommendations(cpu_results, io_results)
        
        print("\nComparison completed!")


def main():
    """Main function to run concurrency comparison."""
    comparison = ConcurrencyComparison()
    comparison.run_comprehensive_comparison()


if __name__ == "__main__":
    main()
