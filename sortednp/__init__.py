
from _sortednp import merge, intersect

def resolve(o):
    """
    Helper function.

    Check whether the given object is callable. If yes, return its return
    value, otherwise return the object itself.
    """
    return o() if callable(o) else o

def kway_merge(*arrays, assume_sorted=True):
    """
    Merge all given arrays and return the result. Depending on the optional
    flag assume_sorted, the function sorts the arrays before merging.

    The method raises a TypeError, if the array count is zero.

    The arguments can load the arrays on the fly. If an argument is callable,
    its return value is used as an array instead of the argument itself. This
    make it possible to load one array after another to avoid having all
    arrays in memory at the same time..

    Note on the performance: The function merges the arrays one-by-one. This
    not the most performant implementation. Use the module heapq for more
    efficient ways to merge sorted arrays.
    """
    if len(arrays) == 0:
        raise TypeError("Merge expects at least one array.")

    arrays = list(arrays)
    m = arrays.pop()
    m = resolve(m)
    if not assume_sorted:
        m.sort()
    for a in arrays:
        a = resolve(a)
        if not assume_sorted:
            a.sort()
        m = merge(m, a)
    return m

def kway_intersect(*arrays, assume_sorted=True):
    """
    Intersect all given arrays and return the result. Depending on the
    optional flag assume_sorted, the function sort sorts the arrays prior to
    intersecting.

    The method raises a TypeError, if the array count is zero.

    The arguments can load the arrays on the fly. If an argument is callable,
    its return value is used as an array instead of the argument itself. This
    make it possible to load one array after another to avoid having all
    arrays in memory at the same time..

    Note on the performance: The function intersects the arrays one-by-one.
    This is not the most performant implementation. 
    """

    if len(arrays) == 0:
        raise TypeError("Merge expects at least one array.")

    # start with smallest non-callable
    inf = float('inf')
    len_array = [(inf if callable(a) else len(a), a) for a in arrays]
    len_array = sorted(len_array, key=lambda x: x[0])
    arrays = [a for l, a in len_array]

    i = arrays.pop()
    i = resolve(i)
    if not assume_sorted:
        i.sort()
    for a in arrays:
        a = resolve(a)
        if not assume_sorted:
            a.sort()
        i = intersect(i, a)
    return i
