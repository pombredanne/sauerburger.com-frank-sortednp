WARNING: The current implementation is still in the research-and-development
phase.

# About
Numpy and Numpy arrays are a really great tool. However intersecting and
merging multiple numpy arrays rather less performant. The current numpy
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
is intended to deal with arrays with $`10^6`$ or $`10^10`$ items. Usually these
arrays are too large to keep more then two of them in memory at the same
time, which reduces the need for a k-way merger or intersecter.

# Installation

You can install the package by cloning the git repository and running the
setup script.

<!-- console
$ git clone https://gitlab.sauerburger.com/frank/sortednp.git sortednp
$ cd sortednp && python3 setup.py install
-->

```bash
$ git clone git@gitlab.sauerburger.com:frank/sortednp.git
$ python3 setup.py install
```

# Usage

Two numpy sorted arrays can be merged wih the `merge` method, which takes two
numpy arrays, as shown in the code snipped `merge.py`.

<!-- write merge.py -->
```python
import numpy as np
import sortednp as snp

a = np.array([0, 3, 4, 6, 7], dtype='float')
b = np.array([1, 2, 3, 5, 7, 9], dtype='float')

m = snp.merge(a, b)
print(m)
```

If you run this you should see the union of both arrays as a sorted numpy
array.
<!-- console_output -->
```python
$ python3 merge.py
[ 0.  1.  2.  3.  3.  4.  5.  6.  7.  7.  9.]
```

Two numpy sorted arrays can be intersected wih the `intersect` method, which takes two
numpy arrays, as shown in the code snipped `intersect.py`.

<!-- write intersect.py -->
```python
import numpy as np
import sortednp as snp

a = np.array([0, 3, 4, 6, 7], dtype='float')
b = np.array([1, 2, 3, 5, 7, 9], dtype='float')

i = snp.intersect(a, b)
print(i)
```

If you run this you should see the union of both arrays as a sorted numpy
array.
<!-- console_output -->
```python
$ python3 intersect.py
[ 3.  7.]
```
