"""多种子实验与并行计算。"""

from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context

import torch

SEEDS = list(range(37, 47))


def single_thread():
    torch.set_num_threads(1)


def parallel_map(function, jobs, workers):
    if workers == 1:
        single_thread()
        return [function(job) for job in jobs]
    with ProcessPoolExecutor(
        max_workers=workers, mp_context=get_context("spawn"), initializer=single_thread,
    ) as pool:
        results = []
        for index, result in enumerate(pool.map(function, jobs), 1):
            results.append(result)
            print(f"{function.__name__}: {index}/{len(jobs)}", flush=True)
    return results
