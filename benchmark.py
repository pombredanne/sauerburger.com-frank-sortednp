#!/usr/bin/env python3

import timeit
import numpy as np
from functools import reduce
import argparse
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import sortednp as snp

np.random.seed(2853)

def np_kway_intersect(*arrays, assume_sorted):
    """
    Intersect all arrays iteratively.
    """
    return reduce(np.intersect1d, arrays)

def np_kway_merge(*arrays, assume_sorted):
    """
    Intersect all arrays iteratively.
    """
    all = np.concatenate(arrays)
    all.sort()
    return all

def get_time(func, *args, **kwds):
    """
    Returns the execution time of the function when called with the given
    arguments and keywords.
    """
    start_time = timeit.default_timer()
    func(*args, **kwds)
    return timeit.default_timer() - start_time

def get_random_array(size, density=1):
    """
    Return an random array of the given size. The density paramter describes
    how likely it is to have duplicated items. Higher values produce more
    duplicates.
    """
    p = 1.5
    pool = np.arange(0, size * p).astype('float64')
    np.random.shuffle(pool)
    a = pool[:int(size / p)]
    return a

def benchmark(algo, array_size, n_arrays, assume_sorted):
    """
    Run a generic benchmark test. The given algorithm must be callable. This
    must be a calculate the k-way union or intersection for numpy of sortednp. The
    benchmark will use n_arrays arrays with each array_size entries. If the
    flag assume sorted is True, the algorithm will be called with the same
    argument and the benchmark sorts the arrays before the stop watch is
    started. If assume_sorted is False, the time it takes to sort the arrays
    is included in the execution time of the algorithm.
    """
    arrays = [get_random_array(array_size) for i in range(n_arrays)]

    if assume_sorted:
        for array in arrays:
            array.sort()

    return get_time(algo, *arrays, assume_sorted=assume_sorted) 

def plot_intersect_benchmark(assume_sorted, n_average):
    """
    Create the plot for the intersection benchmark.
    """
    plt.figure(figsize=(6, 4))
    plt.subplot(111)

    sizes = [1e7, 5e6, 2e6, 1e6, 5e5, 2e5, 1e5, 5e4, 2e4, 1e4, 5e3, 2e3, 1e3]
    colors = ['#8c564b', '#9467bd', '#d62728', '#2ca02c', '#ff7f0e', '#1f77b4']
    def next_color():
        return colors.pop()

    for n in [15, 10, 5, 2]:
        snp_timing = np.zeros(len(sizes))
        np_timing = np.zeros(len(sizes))

        for i in range(n_average):
            snp_timing += np.array([
                benchmark(snp.kway_intersect, s, n, assume_sorted)
                for s in sizes])
            np_timing += np.array([
                benchmark(np_kway_intersect, s, n, assume_sorted)
                for s in sizes])

        c = next_color()
        plt.plot(sizes, snp_timing / np_timing, label="intersect %d arrays" % n, color=c)

    plt.xscale('log')
    plt.xlabel("array size")
    plt.ylabel("duration sortednp / numpy")
    plt.xlim([min(sizes), max(sizes)])
    plt.legend()
    plt.tight_layout()
    suffix = "_assume_sorted" if assume_sorted else ""
    plt.savefig("bm_intersect%s.png" % suffix, dpi=300)

def plot_merge_benchmark(assume_sorted, n_average):
    """
    Create the plot for the union benchmark.
    """
    plt.figure(figsize=(6, 4))
    plt.subplot(111)

    sizes = [1e7, 5e6, 2e6, 1e6, 5e5, 2e5, 1e5, 5e4, 2e4, 1e4, 5e3, 2e3, 1e3]
    colors = ['#8c564b', '#9467bd', '#d62728', '#2ca02c', '#ff7f0e', '#1f77b4']
    def next_color():
        return colors.pop()

    for n in [15, 10, 5, 2]:
        snp_timing = np.zeros(len(sizes))
        np_timing = np.zeros(len(sizes))

        for i in range(n_average):
            snp_timing += np.array([
                benchmark(snp.kway_merge, s, n, assume_sorted)
                for s in sizes])
            np_timing += np.array([
                benchmark(np_kway_merge, s, n, assume_sorted)
                for s in sizes])

        c = next_color()
        plt.plot(sizes, snp_timing / np_timing, label="merge %d arrays" % n, color=c)

    plt.xscale('log')
    plt.xlabel("array size")
    plt.ylabel("duration sortednp / numpy")
    plt.xlim([min(sizes), max(sizes)])
    plt.legend()
    plt.tight_layout()
    suffix = "_assume_sorted" if assume_sorted else ""
    plt.savefig("bm_merge%s.png" % suffix, dpi=300)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create benchmark plots to "
        "compare the sortednp package to the default numpy methods.")

    parser.add_argument("-n", "--average", default=5, type=int, dest='n',
        help="Repeat each operation n times and take the average, "
        "default is 5.")

    parser.add_argument("--quick", action="store_const", dest='n', const=1,
        help="Perform each test only once.")

    args = parser.parse_args()

    print("The benchmark needs about 2GB of memory.")

    print("Benchmark: merge, assume_sorted=False")
    plot_merge_benchmark(assume_sorted=False, n_average=args.n)

    print("Benchmark: merge, assume_sorted=True")
    plot_merge_benchmark(assume_sorted=True, n_average=args.n)

    print("Benchmark: intersect, assume_sorted=False")
    plot_intersect_benchmark(assume_sorted=False, n_average=args.n)

    print("Benchmark: intersect, assume_sorted=True")
    plot_intersect_benchmark(assume_sorted=True, n_average=args.n)

