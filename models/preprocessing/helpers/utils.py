from contextlib import contextmanager
from pathlib import Path
import random
import time
import numpy as np

def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)

def ensure_dirs(paths):
    for path in paths:
        ensure_dir(path)

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)

@contextmanager
def timer(name):
    start = time.time()
    yield
    elapsed = time.time() - start
    print(f"{name}: {elapsed:.2f}s")


def log(message):
    print(f"[preprocessing] {message}")