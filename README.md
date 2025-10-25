# Python Concurrency Examples

A comprehensive collection of Python concurrency examples covering threading, multiprocessing, async/await, and concurrent.futures.

## Overview

This repository demonstrates various Python concurrency approaches with practical examples, performance comparisons, and real-world use cases.

## Features

- **Threading Examples**: Basic threading, synchronization, communication, and ThreadPoolExecutor
- **Multiprocessing Examples**: Process creation, communication, shared memory, and process pools
- **Async/Await Examples**: Async functions, context managers, generators, and I/O operations
- **Concurrent.Futures Examples**: High-level concurrency with ThreadPoolExecutor and ProcessPoolExecutor
- **Performance Comparison**: Benchmarks and recommendations for different use cases
- **Real-world Examples**: Practical patterns and applications

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd python-threading
```

2. Install dependencies:
```bash
pip install -e .
```

Or using uv:
```bash
uv sync
```

## Quick Start

Run the main demo to see all examples:

```bash
python main_demo.py
```

Or run specific examples:

```bash
# Threading examples
python threading_examples.py

# Multiprocessing examples
python multiprocessing_examples.py

# Async examples
python async_examples.py

# Concurrent.futures examples
python concurrent_futures_examples.py

# Performance comparison
python concurrency_comparison.py
```

## Examples Overview

### 1. Threading Examples (`threading_examples.py`)

- Basic threading with `Thread` class
- Thread synchronization (Lock, Semaphore, Event, Condition)
- Thread communication with Queue
- ThreadPoolExecutor for managed thread pools
- Thread-safe data structures
- Daemon threads

### 2. Multiprocessing Examples (`multiprocessing_examples.py`)

- Process creation and management
- Process communication (Queue, Pipe, Manager)
- Shared memory (Value, Array)
- Process synchronization (Lock, Semaphore, Event)
- Process pools for parallel execution
- Performance comparison with threading

### 3. Async/Await Examples (`async_examples.py`)

- Basic async/await syntax
- Async context managers and iterators
- Async generators
- Async synchronization primitives
- Async I/O operations
- Task management and cancellation
- Producer-consumer patterns

### 4. Concurrent.Futures Examples (`concurrent_futures_examples.py`)

- ThreadPoolExecutor for I/O-bound tasks
- ProcessPoolExecutor for CPU-bound tasks
- Future objects and callbacks
- Exception handling
- Timeout and cancellation
- Performance optimization
- Real-world patterns

### 5. Performance Comparison (`concurrency_comparison.py`)

- CPU-intensive task benchmarks
- I/O-intensive task benchmarks
- Memory usage analysis
- Performance recommendations
- Use case guidance

## Key Concepts

### When to Use Each Approach

| Approach | Best For | Pros | Cons |
|----------|----------|------|------|
| **Threading** | I/O-bound tasks | Simple, shared memory | GIL limitation |
| **Multiprocessing** | CPU-bound tasks | True parallelism | Memory overhead |
| **Async/Await** | I/O-bound tasks | Efficient, modern | Single-threaded |
| **Concurrent.Futures** | High-level concurrency | Easy to use | Less control |

### Performance Guidelines

- **CPU-bound tasks**: Use multiprocessing
- **I/O-bound tasks**: Use threading or async/await
- **Mixed workloads**: Use concurrent.futures
- **Simple tasks**: Sequential execution may be sufficient

## Dependencies

- Python 3.12+
- pathos (for advanced multiprocessing)
- aiohttp (for async HTTP examples)
- aiofiles (for async file operations)
- psutil (for system monitoring)
- requests (for HTTP examples)

## File Structure

```
python-threading/
├── main_demo.py                 # Main demonstration script
├── threading_examples.py        # Threading examples
├── multiprocessing_examples.py   # Multiprocessing examples
├── async_examples.py           # Async/await examples
├── concurrent_futures_examples.py # Concurrent.futures examples
├── concurrency_comparison.py   # Performance comparison
├── main_multiprocessing.py     # Original multiprocessing example
├── main_multiprocessing_pool.py # Original pool example
├── main_pathos_pool.py         # Original pathos example
├── pyproject.toml              # Project configuration
├── uv.lock                     # Dependency lock file
└── README.md                   # This file
```

## Usage Examples

### Basic Threading

```python
import threading
import time

def worker(name):
    print(f"Worker {name} starting")
    time.sleep(2)
    print(f"Worker {name} finished")

# Create and start threads
threads = []
for i in range(3):
    thread = threading.Thread(target=worker, args=(f"Worker-{i}",))
    threads.append(thread)
    thread.start()

# Wait for all threads
for thread in threads:
    thread.join()
```

### Basic Multiprocessing

```python
import multiprocessing
from concurrent.futures import ProcessPoolExecutor

def cpu_task(n):
    return sum(i * i for i in range(n))

# Use ProcessPoolExecutor
with ProcessPoolExecutor(max_workers=4) as executor:
    results = list(executor.map(cpu_task, [10000, 20000, 30000]))
```

### Basic Async/Await

```python
import asyncio

async def async_task(name):
    print(f"Task {name} starting")
    await asyncio.sleep(2)
    print(f"Task {name} finished")

async def main():
    tasks = [async_task(f"Task-{i}") for i in range(3)]
    await asyncio.gather(*tasks)

# Run async main
asyncio.run(main())
```

## Performance Tips

1. **Profile before optimizing**: Measure actual performance bottlenecks
2. **Choose the right tool**: Match the approach to your workload
3. **Consider overhead**: Concurrency has setup costs
4. **Monitor resources**: Watch CPU, memory, and I/O usage
5. **Test with realistic data**: Use production-like workloads

## Common Pitfalls

1. **GIL limitations**: Threading won't help CPU-bound tasks
2. **Memory overhead**: Multiprocessing uses more memory
3. **Deadlocks**: Be careful with synchronization primitives
4. **Exception handling**: Concurrent code needs robust error handling
5. **Resource cleanup**: Always clean up resources properly

## Contributing

1. Fork the repository
2. Create a feature branch
3. Add your examples or improvements
4. Test thoroughly
5. Submit a pull request

## License

This project is open source and available under the MIT License.

## Resources

- [Python Threading Documentation](https://docs.python.org/3/library/threading.html)
- [Python Multiprocessing Documentation](https://docs.python.org/3/library/multiprocessing.html)
- [Python AsyncIO Documentation](https://docs.python.org/3/library/asyncio.html)
- [Concurrent.Futures Documentation](https://docs.python.org/3/library/concurrent.futures.html)
- [Real Python Concurrency Guide](https://realpython.com/python-concurrency/)

## Support

For questions or issues, please open an issue on GitHub or contact the maintainers.