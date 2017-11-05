#!/usr/bin/env python3

import timeit
import numpy as np
import sortednp as snp

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

np.random.seed(2853)

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
    a.sort()
    return a

def get_snp_intersect_time(size, n=2):
    """
    Calculate the intersection for the given array sizes using sorted numpy
    and return the duration. The parameter n determines the number of arrays.
    """
    a = [get_random_array(size) for i in range(n)]

    def func():
        x = a.pop()
        for A in a:
            x = snp.intersect(A, x)

        return x
    
    return get_time(func)

def get_snp_merge_time(size, n=2):
    """
    Calculate the union for the given array sizes using sorted numpy
    and return the duration. The parameter n determines the number of arrays.
    """
    a = [get_random_array(size) for i in range(n)]

    def func():
        x = a.pop()
        for A in a:
            x = snp.merge(A, x)

        return x
    
    return get_time(func)

def get_np_intersect_time(size, n=2):
    """
    Calculate the intersection for the given array sizes using standard numpy
    and return the duration. The parameter n determines the number of arrays.
    """
    a = [get_random_array(size) for i in range(n)]

    def func():
        x = a.pop()
        for A in a:
            x = np.intersect1d(A, x)
        return x
    
    return get_time(func)

def get_np_merge_time(size, n=2):
    """
    Calculate the union for the given array sizes using standard numpy
    and return the duration. The parameter n determines the number of arrays.
    """
    a = [get_random_array(size) for i in range(n)]

    def func():
        x = a.pop()
        for A in a:
            x = np.concatenate((A, x))
        x.sort()
        return x
    
    return get_time(func)


def bm_intersect():
    """
    Create the plot for the intersection benchmark.
    """
    plt.figure(figsize=(6, 4))
    plt.subplot(111)

    sizes = [1e4, 2e4, 5e4, 1e5, 2e5, 5e5, 1e6]
    colors = list(reversed(['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728',
        '#9467bd', '#8c564b', '#e377c2', '#7f7f7f',
        '#bcbd22', '#17becf']))
    def next_color():
        return colors.pop()

    for n in [2, 5, 10, 15]:
        snp_timing = np.zeros(len(sizes))
        np_timing = np.zeros(len(sizes))

        for i in range(5):
            snp_timing += np.array([get_snp_intersect_time(s, n) for s in sizes])
            np_timing += np.array([get_np_intersect_time(s, n) for s in sizes])

        c = next_color()
        plt.plot(sizes, snp_timing / np_timing, label="intersect %d arrays" % n, color=c)

    plt.xscale('log')
    plt.xlabel("array size")
    plt.ylabel("duration sortednp / numpy")
    plt.xlim([min(sizes), max(sizes)])
    plt.legend()
    plt.tight_layout()
    plt.savefig("bm_intersect.png", dpi=300)

def bm_merge():
    """
    Create the plot for the union benchmark.
    """
    plt.figure(figsize=(6, 4))
    plt.subplot(111)

    sizes = [1e3, 2e3, 5e3, 1e4, 2e4, 5e4]
    colors = list(reversed(['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728',
        '#9467bd', '#8c564b', '#e377c2', '#7f7f7f',
        '#bcbd22', '#17becf']))
    def next_color():
        return colors.pop()

    for n in [2, 5, 10, 15]:
        snp_timing = np.zeros(len(sizes))
        np_timing = np.zeros(len(sizes))

        for i in range(1):
            snp_timing += np.array([get_snp_merge_time(s, n) for s in sizes])
            np_timing += np.array([get_np_merge_time(s, n) for s in sizes])

        c = next_color()
        plt.plot(sizes, snp_timing / np_timing, label="merge %d arrays" % n, color=c)

    plt.xscale('log')
    plt.xlabel("array size")
    plt.ylabel("duration sortednp / numpy")
    plt.xlim([min(sizes), max(sizes)])
    plt.legend()
    plt.tight_layout()
    plt.savefig("bm_merge.png", dpi=300)

if __name__ == "__main__":
    bm_merge()
    bm_intersect()

