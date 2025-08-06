import functools
import os
import logging as log
import psutil
import time

logger = log.getLogger(__name__)

def process_memory():
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    return mem_info.rss

def measure_performance_decorator(measure: bool):
    """
    Measure and report the execution time and CPU usage of a function.
    It conditionally performs these measurements based on a boolean parameter measure.

    Args:
        measure (bool):  determines whether to measure performance metrics.

    Returns:

    """
    def decorator_if(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            if measure:
                initial_cpu_times = psutil.cpu_times()
                start_time = time.time()
                mem_before = process_memory()
                # -----------------------------
                result = func(*args, **kwargs)
                # -----------------------------
                mem_after = process_memory()

                end_time = time.time()
                final_cpu_times = psutil.cpu_times()

                # Calculate the CPU usage
                cpu_usage = {
                    "user": final_cpu_times.user - initial_cpu_times.user,
                    "system": final_cpu_times.system - initial_cpu_times.system,
                    "idle": final_cpu_times.idle - initial_cpu_times.idle,
                    "cpu_num":  len(psutil.Process().cpu_affinity()),
                    "memory_used": mem_after - mem_before
                }
                execution_time = end_time - start_time

                print(f"Execution time: {execution_time} seconds")
                print(f"CPU usage: {cpu_usage}")
            else:
                result = func(*args, **kwargs)
            return result
        return wrapper
    return decorator_if

def validate_file_path(file_path: str):
    """
    Validates the provided file path and raises an exception if the path is not valid.

    Parameters:
    file_path (str): The file path to be validated.

    Raises:
    ValueError: If the file path is invalid or does not exist.
    """
    # Check if the path is empty
    if not file_path:
        raise ValueError("The file path is empty.")

    # Check if the path is a string
    if not isinstance(file_path, str):
        raise ValueError("The file path must be a string.")

    # Check if the path exists
    if not os.path.exists(file_path):
        raise ValueError(f"The file path '{file_path}' does not exist.")

    # Check if the path is a file (not a directory)
    if not os.path.isfile(file_path):
        raise ValueError(f"The file path '{file_path}' is not a file.")

    # Optionally, check if the file has the correct permissions
    if not os.access(file_path, os.R_OK):
        raise ValueError(f"The file path '{file_path}' is not readable.")

    if not os.access(file_path, os.W_OK):
        raise ValueError(f"The file path '{file_path}' is not writable.")

    # If all checks pass, the file path is valid
    logger.info(f"The file path '{file_path}' is valid.")
