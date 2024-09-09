import time
from functools import wraps

def timer(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"| SYS: Time taken for {func.__name__}: {end_time - start_time:.4f} seconds")
        return result
    return wrapper