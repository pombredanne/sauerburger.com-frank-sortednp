Numpy and Numpy arrays are a really great tool. However, intersecting and
merging multiple numpy arrays is rather less performant. The current numpy
implementation concatenates the two arrays and sorts the combination. If you
want to merge or intersect multiple numpy arrays, there is a much faster way,
by using the property, that the resulting array is sorted.

Sortednp (sorted numpy) operates on sorted numpy arrays to calculate the
intersection or the union of two numpy arrays in an efficient way. The
resulting array is again a sorted numpy array, which can be merged or
intersected with the next array. The intended use case is that sorted numpy
arrays are sorted as the basic data structure and merged or intersected at
request. Typical applications include information retrieval and search engines
in particular.

It is also possible to implement a k-way merging or intersecting algorithm,
which operates on an arbitrary number of arrays at the same time. This package
is intended to deal with arrays with $`10^6`$ or $`10^{10}`$ items. Usually, these
arrays are too large to keep more than two of them in memory at the same
time. This package implements methods to merge and intersect multiple arrays,
which can be loaded on-demand.

# Installation
There are two different methods to install `sortednp`.

## Using `pip` (recommended)

You can install the package directly from PyPI using `pip` (here `pip3`). There are
pre-compiled wheels for `linux` 32- and 64bit.

```bash
$ pip3 install sortednp
```

## Using `setuptools`

Alternatively, you can clone the git repository and run the
setup script.

```bash
$ git clone https://gitlab.sauerburger.com/frank/sortednp.git
$ cd sortednp
$ python3 setup.py install
```
## Numpy Dependency
The installation fails in some cases, because of a build-time dependency on
numpy. Usually, the problem can be solved by manually installing a recent numpy
version via `pip3 install -U numpy`.

# Usage

The package provides two different kinds of methods. The first class is intended
to operate on two arrays. The second class operates on two or more arrays and
calls the first class of methods internally.

## Two-way methods

Two numpy sorted arrays can be merged with the `merge` method, which takes two
numpy arrays and returns the sorted union of the two arrays.

<!-- write merge.py -->
```python
# merge.py
import numpy as np
import sortednp as snp

a = np.array([0, 3, 4, 6, 7])
b = np.array([1, 2, 3, 5, 7, 9])

m = snp.merge(a, b)
print(m)
```

If you run this, you should see the union of both arrays as a sorted numpy
array.
<!-- console_output -->
```python
$ python3 merge.py
[0 1 2 3 3 4 5 6 7 7 9]
```

Two sorted numpy arrays can be intersected with the `intersect` method, which takes two
numpy arrays and returns the sorted intersection of the two arrays.

<!-- write intersect.py -->
```python
# intersect.py
import numpy as np
import sortednp as snp

a = np.array([0, 3, 4, 6, 7])
b = np.array([1, 2, 3, 5, 7, 9])

i = snp.intersect(a, b)
print(i)
```

If you run this, you should see the intersection of both arrays as a sorted numpy
array.
<!-- console_output -->
```python
$ python3 intersect.py
[3 7]
```
## k-way methods
Similarly, the k-way intersect and merge methods take two or more arrays and
perform the merge or intersect operation on its arguments.

<!-- write kway_intersect.py -->
```python
# kway_intersect.py
import numpy as np
import sortednp as snp

a = np.array([0, 3, 4, 6, 7])
b = np.array([0, 3, 5, 7, 9])
c = np.array([1, 2, 3, 5, 7, 9])
d = np.array([2, 3, 6, 7, 8])

i = snp.kway_intersect(a, b, c, d)
print(i)
```

If you run this, you should see the intersection of all four arrays as a sorted numpy
array.
<!-- console_output -->
```python
$ python3 kway_intersect.py
[3 7]
```

The k-way merger `sortednp.kway_merge` works analogously. However, the native
`numpy` implementation is faster compared to the merge provided by this package.
The k-way merger has been added for completeness. The package `heapq` provides
efficient methods to merge multiple arrays simultaneously.

The methods `kway_merge` and `kway_intersect` accept the optional keyword
argument `assume_sorted`. By default, it is set to `True`. If it is set to `False`,
the method calls `sort()` on the input arrays before performing the operation.
The default should be kept if the arrays are already sorted to save the time it
takes to sort the arrays.

Since the arrays might be too large to keep all of them in memory at the same
time, it is possible to pass a `callable` instead of an array to the methods.
The `callable` is expected to return the actual array. It is called immediately
before the array is required. This reduces the memory consumption.

## Algorithms
Intersections are calculated by iterating both arrays. For a given element in
one array, the method needs to search the other and check if the element is
contained. In order to make this more efficient, we can use the fact that the
arrays are sorted. There are three search methods, which can be selected via the
optional keyword argument `algorithm`.

 * `sortednp.SIMPLE_SEARCH`: Search for an element by linearly iterating over the
   array element-by-element.
   [More Information](https://en.wikipedia.org/wiki/Linear_search).
 * `sortednp.BINARY_SEARCH`: Slice the remainder of the array in halves and
   repeat the procedure on the slice which contains the searched element.
   [More Information](https://en.wikipedia.org/wiki/Binary_search_algorithm).
 * `sortednp.GALLOPING_SEARCH`: First, search for an element linearly, doubling
   the step size after each step. If a step goes beyond the search element,
   perform a binary search between the last two positions.
   [More Information](https://en.wikipedia.org/wiki/Exponential_search).

The default is `sortednp.GALLOPING_SEARCH`.
<!-- END-OF-RST -->
The performance of all three algorithms is compared in the next section.

# Performance
The performance of the package can be compared with the default implementation
of numpy. The ratio of the execution time between sortednp and numpy is
shown for various different benchmark tests.

The merge or intersect time can be estimated under two different assumptions. If
the arrays, which are merged or intersected, are already sorted, one should not
consider the time it takes to sort the random arrays in the benchmark. On the
other hand, if one considers a scenario in which the arrays are not sorted, one
should take the sorting time into account.

## Intersect

The performance of the intersection operation depends on the sparseness of the
two arrays. For example, if the first element of one of the arrays is larger
than all elements in the other array, only the other array has to be searched
(linearly, binarily, or exponentially). Similarly, if the common elements are
far apart in the arrays (sparseness), large chunks of the arrays can be skipped.
The arrays in the benchmark contain random (unique) integers. The sparseness is
defined as the average difference between two consecutive elements in one array.

The first set of tests studies the performance dependence on the size of the
arrays. The second set of tests studies the dependence on the sparseness of the
array.

### Assume sorted arrays
The following table summarizes the performance compared to numpy if one ignores
the time it takes to sort the initial arrays.
<table>
  <tr>
    <th>Test</th>
    <th>Simple Search</th>
    <th>Binary Search</th>
    <th>Galloping Search</th>
  </tr>
  <tr>
    <th>Intersect</th>
    <td> <img src="https://gitlab.sauerburger.com/frank/sortednp/-/jobs/artifacts/master/raw/bm_intersect_assume_sorted_simple.png?job=benchmark" /> </td>
    <td> <img src="https://gitlab.sauerburger.com/frank/sortednp/-/jobs/artifacts/master/raw/bm_intersect_assume_sorted_binary.png?job=benchmark" /> </td>
    <td> <img src="https://gitlab.sauerburger.com/frank/sortednp/-/jobs/artifacts/master/raw/bm_intersect_assume_sorted_galloping.png?job=benchmark" /> </td>
  </tr>
  <tr>
    <th>Intersect Sparseness</th>
    <td> <img src="https://gitlab.sauerburger.com/frank/sortednp/-/jobs/artifacts/master/raw/bm_intersect_sparse_assume_sorted_simple.png?job=benchmark" /> </td>
    <td> <img src="https://gitlab.sauerburger.com/frank/sortednp/-/jobs/artifacts/master/raw/bm_intersect_sparse_assume_sorted_binary.png?job=benchmark" /> </td>
    <td> <img src="https://gitlab.sauerburger.com/frank/sortednp/-/jobs/artifacts/master/raw/bm_intersect_sparse_assume_sorted_galloping.png?job=benchmark" /> </td>
  </tr>
</table>

### Include sorting time
The following table summarizes the performance compared to numpy if one takes
the time it takes to sort the initial arrays into account.
<table>
  <tr>
    <th>Test</th>
    <th>Simple Search</th>
    <th>Binary Search</th>
    <th>Galloping Search</th>
  </tr>
  <tr>
    <th>Intersect</th>
    <td> <img src="https://gitlab.sauerburger.com/frank/sortednp/-/jobs/artifacts/master/raw/bm_intersect_simple.png?job=benchmark" /> </td>
    <td> <img src="https://gitlab.sauerburger.com/frank/sortednp/-/jobs/artifacts/master/raw/bm_intersect_binary.png?job=benchmark" /> </td>
    <td> <img src="https://gitlab.sauerburger.com/frank/sortednp/-/jobs/artifacts/master/raw/bm_intersect_galloping.png?job=benchmark" /> </td>
  </tr>
  <tr>
    <th>Intersect Sparseness</th>
    <td> <img src="https://gitlab.sauerburger.com/frank/sortednp/-/jobs/artifacts/master/raw/bm_intersect_sparse_simple.png?job=benchmark" /> </td>
    <td> <img src="https://gitlab.sauerburger.com/frank/sortednp/-/jobs/artifacts/master/raw/bm_intersect_sparse_binary.png?job=benchmark" /> </td>
    <td> <img src="https://gitlab.sauerburger.com/frank/sortednp/-/jobs/artifacts/master/raw/bm_intersect_sparse_galloping.png?job=benchmark" /> </td>
  </tr>
</table>

## Merge
<table>
  <tr>
    <th>Assume sorted</th>
    <th>Include sorting time</th>
  </tr>
  <tr>
    <td> <img src="https://gitlab.sauerburger.com/frank/sortednp/-/jobs/artifacts/master/raw/bm_merge_assume_sorted.png?job=benchmark" /> </td>
    <td> <img src="https://gitlab.sauerburger.com/frank/sortednp/-/jobs/artifacts/master/raw/bm_merge.png?job=benchmark" /> </td>
  </tr>
  </tr>
</table>
