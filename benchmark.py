#!/usr/bin/env python3

"""
Run benchmarks to compare sortednp to numpy and show the result in plots. The
benchmark tests both intersecting and merging. The benchmarks are executed
twice, once taking the time to sort the arrays into account, and once using
presorted arrays.
"""

import timeit
from functools import reduce
import argparse
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import sortednp as snp

matplotlib.pyplot.switch_backend('Agg')

np.random.seed(2853)

def np_kway_intersect(*arrays, assume_sorted, **_kwds):
    """
    Intersect all arrays iteratively.
    """
    if assume_sorted:
        # This infomraiton cannot be used
        pass

    return reduce(np.intersect1d, arrays)

def np_kway_merge(*arrays, assume_sorted, **_kwds):
    """
    Intersect all arrays iteratively.
    """
    if assume_sorted:
        # This infomraiton cannot be used
        pass

    concatenation = np.concatenate(arrays)
    concatenation.sort()
    return concatenation

def get_time(func, *args, **kwds):
    """
    Returns the execution time of the function when called with the given
    arguments and keywords.
    """
    start_time = timeit.default_timer()
    func(*args, **kwds)
    return timeit.default_timer() - start_time

def get_random_array(size, sparseness=2):
    """
    Return an random array of the given size. The density paramter describes
    how likely it is to have duplicated items. Higher values produce more
    duplicates.
    """
    if sparseness > 2:
        enlarge = 2
        factor = sparseness / 2
    else:
        enlarge = sparseness
        factor = 1

    pool = np.arange(0, size * enlarge).astype('float32')
    np.random.shuffle(pool)
    pool *= factor
    return pool[:int(size)]

def benchmark(func, arrays, assume_sorted, algorithm=None):
    """
    Run a generic benchmark test. The given algorithm must be callable. This
    must be a calculate the k-way union or intersection for numpy of sortednp. The
    benchmark will use n_arrays arrays with each array_size entries. If the
    flag assume sorted is True, the algorithm will be called with the same
    argument and the benchmark sorts the arrays before the stop watch is
    started. If assume_sorted is False, the time it takes to sort the arrays
    is included in the execution time of the algorithm.
    """
    if isinstance(arrays, tuple):
        array_size, n_arrays = arrays
        arrays = [get_random_array(array_size) for i in range(n_arrays)]

    if assume_sorted:
        for array in arrays:
            array.sort()

    kwds = {"assume_sorted": assume_sorted}
    if algorithm is not None:
        kwds['algorithm'] = algorithm

    return get_time(func, *arrays, **kwds)

def plot_intersect_benchmark(assume_sorted, n_average, algorithm=None):
    """
    Create the plot for the intersection benchmark.
    """
    plt.figure(figsize=(6, 4))
    plt.subplot(111)

    sizes = [1e7, 5e6, 2e6, 1e6, 5e5, 2e5, 1e5, 5e4, 2e4, 1e4, 5e3, 2e3, 1e3]
    colors = ['#8c564b', '#9467bd', '#d62728', '#2ca02c', '#ff7f0e', '#1f77b4']

    for n_arrays in [15, 10, 5, 2]:
        snp_timing = np.zeros(len(sizes))
        np_timing = np.zeros(len(sizes))

        for _i in range(n_average):
            snp_timing += np.array([
                benchmark(snp.kway_intersect, (s, n_arrays), assume_sorted,
                          algorithm=algorithm)
                for s in sizes])
            np_timing += np.array([
                benchmark(np_kway_intersect, (s, n_arrays), assume_sorted)
                for s in sizes])

        color = colors.pop()
        plt.plot(sizes, snp_timing / np_timing,
                 label="intersect %d arrays" % n_arrays, color=color)

    plt.xscale('log')
    plt.xlabel("array size")
    plt.ylabel("duration sortednp / numpy")
    plt.xlim([min(sizes), max(sizes)])
    plt.legend()
    plt.tight_layout()
    suffix = "_assume_sorted" if assume_sorted else ""
    if algorithm is not None:
        if algorithm == snp.SIMPLE_SEARCH:
            suffix += "_simple"
        elif algorithm == snp.BINARY_SEARCH:
            suffix += "_binary"
        elif algorithm == snp.GALLOPING_SEARCH:
            suffix += "_galloping"
        else:
            suffix += "_invalid"
    plt.savefig("bm_intersect%s.png" % suffix, dpi=300)

def plot_merge_benchmark(assume_sorted, n_average):
    """
    Create the plot for the union benchmark.
    """
    plt.figure(figsize=(6, 4))
    plt.subplot(111)

    sizes = [1e7, 5e6, 2e6, 1e6, 5e5, 2e5, 1e5, 5e4, 2e4, 1e4, 5e3, 2e3, 1e3]
    colors = ['#8c564b', '#9467bd', '#d62728', '#2ca02c', '#ff7f0e', '#1f77b4']

    for n_arrays in [15, 10, 5, 2]:
        snp_timing = np.zeros(len(sizes))
        np_timing = np.zeros(len(sizes))

        for _i in range(n_average):
            snp_timing += np.array([
                benchmark(snp.kway_merge, (s, n_arrays), assume_sorted)
                for s in sizes])
            np_timing += np.array([
                benchmark(np_kway_merge, (s, n_arrays), assume_sorted)
                for s in sizes])

        color = colors.pop()
        plt.plot(sizes, snp_timing / np_timing,
                 label="merge %d arrays" % n_arrays, color=color)

    plt.xscale('log')
    plt.xlabel("array size")
    plt.ylabel("duration sortednp / numpy")
    plt.xlim([min(sizes), max(sizes)])
    plt.legend()
    plt.tight_layout()
    suffix = "_assume_sorted" if assume_sorted else ""
    plt.savefig("bm_merge%s.png" % suffix, dpi=300)

def plot_intersect_sparseness(assume_sorted, n_average, algorithm=None):
    """
    Merge n = (2, 5, 10, 15) arrays, where all except one array contains all
    integers from 0 to its, and the other array contains sparse integers.
    """
    plt.figure(figsize=(6, 4))
    plt.subplot(111)

    sparsenesses = [1, 1.3, 2, 5, 10, 20, 50, 100, 200, 500, 1000]
    colors = ['#8c564b', '#9467bd', '#d62728', '#2ca02c', '#ff7f0e', '#1f77b4']

    array_size = 1e6

    for n_arrays in [15, 10, 5, 2]:
        snp_timing = np.zeros(len(sparsenesses))
        np_timing = np.zeros(len(sparsenesses))


        for _i in range(n_average):
            for _j, sparseness in enumerate(sparsenesses):
                arrays = [get_random_array(array_size, sparseness)] \
                     + [np.arange(array_size).astype('float32') \
                        for i in range(n_arrays - 1)]
                snp_timing[_j] += \
                    benchmark(snp.kway_intersect, arrays, assume_sorted,
                              algorithm=algorithm)

                arrays = [get_random_array(array_size, sparseness)] \
                     + [np.arange(array_size).astype('float32') \
                        for i in range(n_arrays - 1)]
                np_timing[_j] += \
                    benchmark(np_kway_intersect, arrays, assume_sorted)

        color = colors.pop()
        plt.plot(sparsenesses, snp_timing / np_timing,
                 label="intersect %d arrays" % n_arrays, color=color)

    plt.xscale('log')
    plt.xlabel("Sparseness of first array")
    plt.ylabel("Duration sortednp / numpy")
    plt.xscale('log')
    plt.legend()
    plt.tight_layout()
    suffix = "_assume_sorted" if assume_sorted else ""
    if algorithm is not None:
        if algorithm == snp.SIMPLE_SEARCH:
            suffix += "_simple"
        elif algorithm == snp.BINARY_SEARCH:
            suffix += "_binary"
        elif algorithm == snp.GALLOPING_SEARCH:
            suffix += "_galloping"
        else:
            suffix += "_invalid"


    plt.savefig("bm_intersect_sparse%s.png" % suffix, dpi=300)

def main():
    """
    Parse command line arguments and run the benchmark.
    """

    parser = argparse.ArgumentParser(
        description="Create benchmark plots to compare the sortednp package"
                    " to the default numpy methods.")

    parser.add_argument("-n", "--average", default=5, type=int, dest='n',
                        help="Repeat each operation n times and take the"
                        " average, default is 5.")

    parser.add_argument("--quick", action="store_const", dest='n', const=1,
                        help="Perform each test only once.")

    parser.add_argument("--merge", action="store_true",
                        help="Benchmark merge operation.")

    parser.add_argument("--intersect", action="store_true",
                        help="Benchmark intersect operation using default"
                        " algorithm.")

    parser.add_argument("--intersect-simple", action="store_true",
                        help="Benchmark intersect operation using simple"
                        " search algorithm.")

    parser.add_argument("--intersect-binary", action="store_true",
                        help="Benchmark intersect operation using binary"
                        " search algorithm.")

    parser.add_argument("--intersect-galloping", action="store_true",
                        help="Benchmark intersect operation using galloping"
                        " search algorithm.")

    cli_args = parser.parse_args()

    print("The benchmark needs about 2GB of memory.")

    if cli_args.merge:
        run_merge(cli_args)

    if cli_args.intersect:
        run_intersect_default(cli_args)

    if cli_args.intersect_simple:
        run_intersect_simple(cli_args)

    if cli_args.intersect_binary:
        run_intersect_galloping(cli_args)

    if cli_args.intersect_galloping:
        run_intersect_galloping(cli_args)

def run_merge(cli_args):
    """
    Run merge benchmark.
    """
    print("Benchmark: merge, assume_sorted=False")
    plot_merge_benchmark(assume_sorted=False, n_average=cli_args.n)

    print("Benchmark: merge, assume_sorted=True")
    plot_merge_benchmark(assume_sorted=True, n_average=cli_args.n)

def run_intersect_default(cli_args):
    """
    Run intersection benchmark using default search.
    """
    print("Benchmark: intersect, assume_sorted=False")
    plot_intersect_benchmark(assume_sorted=False, n_average=cli_args.n)

    print("Benchmark: intersect, assume_sorted=True")
    plot_intersect_benchmark(assume_sorted=True, n_average=cli_args.n)

    print("Benchmark: intersect sparseness, assume_sorted=False")
    plot_intersect_sparseness(assume_sorted=False, n_average=cli_args.n)

    print("Benchmark: intersect sparseness, assume_sorted=True")
    plot_intersect_sparseness(assume_sorted=True, n_average=cli_args.n)

def run_intersect_simple(cli_args):
    """
    Run intersection benchmark using simple search.
    """
    print("Benchmark: intersect (simple), assume_sorted=False")
    plot_intersect_benchmark(assume_sorted=False, n_average=cli_args.n,
                             algorithm=snp.SIMPLE_SEARCH)

    print("Benchmark: intersect (simple), assume_sorted=True")
    plot_intersect_benchmark(assume_sorted=True, n_average=cli_args.n,
                             algorithm=snp.SIMPLE_SEARCH)

    print("Benchmark: intersect sparseness (simple), assume_sorted=False")
    plot_intersect_sparseness(assume_sorted=False, n_average=cli_args.n,
                              algorithm=snp.SIMPLE_SEARCH)

    print("Benchmark: intersect sparseness (simple), assume_sorted=True")
    plot_intersect_sparseness(assume_sorted=True, n_average=cli_args.n,
                              algorithm=snp.SIMPLE_SEARCH)


def run_intersect_binary(cli_args):
    """
    Run intersection benchmark using binary search.
    """
    print("Benchmark: intersect (binary), assume_sorted=False")
    plot_intersect_benchmark(assume_sorted=False, n_average=cli_args.n,
                             algorithm=snp.BINARY_SEARCH)

    print("Benchmark: intersect (binary), assume_sorted=True")
    plot_intersect_benchmark(assume_sorted=True, n_average=cli_args.n,
                             algorithm=snp.BINARY_SEARCH)

    print("Benchmark: intersect sparseness (binary), assume_sorted=False")
    plot_intersect_sparseness(assume_sorted=False, n_average=cli_args.n,
                              algorithm=snp.BINARY_SEARCH)

    print("Benchmark: intersect sparseness (binary), assume_sorted=True")
    plot_intersect_sparseness(assume_sorted=True, n_average=cli_args.n,
                              algorithm=snp.BINARY_SEARCH)


def run_intersect_galloping(cli_args):
    """
    Run intersection benchmark using galloping search.
    """
    print("Benchmark: intersect (galloping), assume_sorted=False")
    plot_intersect_benchmark(assume_sorted=False, n_average=cli_args.n,
                             algorithm=snp.GALLOPING_SEARCH)

    print("Benchmark: intersect (galloping), assume_sorted=True")
    plot_intersect_benchmark(assume_sorted=True, n_average=cli_args.n,
                             algorithm=snp.GALLOPING_SEARCH)

    print("Benchmark: intersect sparseness (galloping), assume_sorted=False")
    plot_intersect_sparseness(assume_sorted=False, n_average=cli_args.n,
                              algorithm=snp.GALLOPING_SEARCH)

    print("Benchmark: intersect sparseness (galloping), assume_sorted=True")
    plot_intersect_sparseness(assume_sorted=True, n_average=cli_args.n,
                              algorithm=snp.GALLOPING_SEARCH)

if __name__ == "__main__":
    main()
