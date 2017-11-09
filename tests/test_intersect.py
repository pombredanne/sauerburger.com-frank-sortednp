
from abc import ABCMeta, abstractmethod
import sys
import weakref
import unittest
import numpy as np

import sortednp as snp

class IntersectBase(metaclass=ABCMeta):
    """
    Define general test cases for the intersect method. Sub-classes need to
    implement have to overwrite the dtype method.
    """

    @abstractmethod
    def get_dtype(self):
        """
        Returns the numpy data type, which should be used for all tests.
        """
        pass
    
    def test_simple_middle(self):
        """
        Check that intersect returns only element, which are present in both
        non-empty input arrays. The common elements are neither at the
        beginning nor at the end of the arrays.
        """
        a = np.array([2, 3, 6, 7, 9], dtype=self.get_dtype())
        b = np.array([1, 3, 7, 8, 10], dtype=self.get_dtype())

        i = snp.intersect(a, b)

        self.assertEqual(list(i), [3, 7])
        self.assertEqual(i.dtype, self.get_dtype())

    def test_simple_end_single(self):
        """
        Check that intersect returns only element, which are present in both
        non-empty input arrays. One common element is at the end of one array.
        """
        a = np.array([2, 3, 6, 7, 9], dtype=self.get_dtype())
        b = np.array([1, 3, 7, 9, 10], dtype=self.get_dtype())

        i = snp.intersect(a, b)

        self.assertEqual(list(i), [3, 7, 9])
        self.assertEqual(i.dtype, self.get_dtype())


    def test_simple_end_both(self):
        """
        Check that intersect returns only element, which are present in both
        non-empty input arrays. One common element is at the end of both
        arrays.
        """
        a = np.array([2, 3, 6, 7, 9], dtype=self.get_dtype())
        b = np.array([1, 3, 7, 9], dtype=self.get_dtype())

        i = snp.intersect(a, b)

        self.assertEqual(list(i), [3, 7, 9])
        self.assertEqual(i.dtype, self.get_dtype())

    def test_simple_begining_single(self):
        """
        Check that intersect returns only element, which are present in both
        non-empty input arrays. One common element is at the begining of one array.
        """
        a = np.array([2, 3, 6, 7, 8], dtype=self.get_dtype())
        b = np.array([1, 2, 3, 7, 9, 10], dtype=self.get_dtype())

        i = snp.intersect(a, b)

        self.assertEqual(list(i), [2, 3, 7])
        self.assertEqual(i.dtype, self.get_dtype())

    def test_simple_begining_both(self):
        """
        Check that intersect returns only element, which are present in both
        non-empty input arrays. One common element is at the end of both
        arrays.
        """
        a = np.array([2, 3, 6, 7, 8], dtype=self.get_dtype())
        b = np.array([2, 3, 7, 9, 10], dtype=self.get_dtype())

        i = snp.intersect(a, b)

        self.assertEqual(list(i), [2, 3, 7])
        self.assertEqual(i.dtype, self.get_dtype())

    def test_empty_intersect(self):
        """
        Check that intersect returns an empty array, if the non-empty inputs
        do not have any elements in common.
        """
        a = np.array([1, 3, 5, 10], dtype=self.get_dtype())
        b = np.array([2, 4, 7, 8, 20], dtype=self.get_dtype())

        i = snp.intersect(a, b)

        self.assertEqual(list(i), [])
        self.assertEqual(len(i), 0)
        self.assertEqual(i.dtype, self.get_dtype())

    def test_empty_input_single(self):
        """
        Check that intersect returns an empty array, if one of the input arrays
        is empty.
        """
        a = np.array([], dtype=self.get_dtype())
        b = np.array([2, 4, 7, 8, 20], dtype=self.get_dtype())

        i = snp.intersect(a, b)
        self.assertEqual(list(i), [])
        self.assertEqual(len(i), 0)
        self.assertEqual(i.dtype, self.get_dtype())

        i = snp.intersect(b, a)
        self.assertEqual(list(i), [])
        self.assertEqual(len(i), 0)
        self.assertEqual(i.dtype, self.get_dtype())

    def test_empty_input_both(self):
        """
        Check that intersect returns an empty array, both  input arrays are
        empty.
        """
        a = np.array([], dtype=self.get_dtype())
        b = np.array([], dtype=self.get_dtype())

        i = snp.intersect(a, b)
        self.assertEqual(list(i), [])
        self.assertEqual(len(i), 0)
        self.assertEqual(i.dtype, self.get_dtype())

    def test_contained(self):
        """
        Check that intersect returns the common elements, if one array is
        contained in the other.
        """
        a = np.array([1, 2, 3, 4, 5, 6, 7, 8], dtype=self.get_dtype())
        b = np.array([4, 5, 7], dtype=self.get_dtype())

        i = snp.intersect(a, b)
        self.assertEqual(list(i), [4, 5, 7])
        self.assertEqual(i.dtype, self.get_dtype())

        i = snp.intersect(b, a)
        self.assertEqual(list(i), [4, 5, 7])
        self.assertEqual(i.dtype, self.get_dtype())

    def test_identical(self):
        """
        Check that intersect returns a copy, if the same array is passed in
        twice.
        """
        a = np.array([3, 4, 6, 8], dtype=self.get_dtype())

        i = snp.intersect(a, a)
        self.assertEqual(list(i), [3, 4, 6, 8])
        self.assertEqual(list(a), [3, 4, 6, 8])
        self.assertEqual(i.dtype, self.get_dtype())

        i[0] = 1
        self.assertEqual(list(a), [3, 4, 6, 8])

    def test_separated(self):
        """
        Check that intersect returns an empty array, if all elements are
        greater than all elements in the other.
        """
        a = np.array([1, 2, 3], dtype=self.get_dtype())
        b = np.array([4, 6, 8], dtype=self.get_dtype())

        i = snp.intersect(a, b)
        self.assertEqual(list(i), [])
        self.assertEqual(len(i), 0)
        self.assertEqual(i.dtype, self.get_dtype())

    def test_duplicates_same(self):
        """
        Check that duplicates in the same array are dropped if not present in
        the other array.
        """
        a = np.array([1, 2, 2, 3], dtype=self.get_dtype())
        b = np.array([1, 6, 8], dtype=self.get_dtype())

        i = snp.intersect(a, b)
        self.assertEqual(list(i), [1])
        self.assertEqual(i.dtype, self.get_dtype())

        a = np.array([1, 2, 2, 3], dtype=self.get_dtype())
        b = np.array([1, 2, 4, 6, 8], dtype=self.get_dtype())

        i = snp.intersect(a, b)
        self.assertEqual(list(i), [1, 2])
        self.assertEqual(i.dtype, self.get_dtype())


    def test_duplicates_both(self):
        """
        Check that duplicates in the same array are kept if they are also
        duplicated in the other array.
        """
        a = np.array([1, 2, 2, 3], dtype=self.get_dtype())
        b = np.array([2, 2, 4, 6, 8], dtype=self.get_dtype())

        i = snp.intersect(a, b)
        self.assertEqual(list(i), [2, 2])
        self.assertEqual(i.dtype, self.get_dtype())

    def test_raise_multi_dim(self):
        """
        Check that passing in a multi dimensional array raises an exception.
        """
        a = np.zeros((10, 2), dtype=self.get_dtype())
        b = np.array([2, 3, 5, 6], dtype=self.get_dtype())

        self.assertRaises(ValueError, snp.intersect, a, b)
        self.assertRaises(ValueError, snp.intersect, b, a)
        self.assertRaises(ValueError, snp.intersect, a, a)
        
    def test_raise_non_array(self):
        """
        Check that passing in a non-numpy-array raises an exception.
        """
        b = np.array([2, 3, 5, 6], dtype=self.get_dtype())

        self.assertRaises(TypeError, snp.intersect, 3, b)
        self.assertRaises(TypeError, snp.intersect, b, 2)
        self.assertRaises(TypeError, snp.intersect, 3, "a")

    def test_reference_counting_principle(self):
        """
        Check that the reference counting works as expected with standard
        numpy arrays.
        """

        # Create inputs
        a = np.arange(10, dtype=self.get_dtype()) * 3
        b = np.arange(10, dtype=self.get_dtype()) * 2 + 5

        # Check ref count for input. Numpy arrays have two references.
        self.assertEqual(sys.getrefcount(a), 2)
        self.assertEqual(sys.getrefcount(b), 2)

        # Create weak refs for inputs
        weak_a = weakref.ref(a)
        weak_b = weakref.ref(b)

        self.assertEqual(sys.getrefcount(a), 2)
        self.assertEqual(sys.getrefcount(b), 2)
        self.assertIsNotNone(weak_a())
        self.assertIsNotNone(weak_b())

        # Delete a
        del a
        self.assertEqual(sys.getrefcount(b), 2)
        self.assertIsNone(weak_a())
        self.assertIsNotNone(weak_b())

        # Delete b
        del b
        self.assertIsNone(weak_a())
        self.assertIsNone(weak_b())

    def test_reference_counting(self):
        """
        Check that the reference counting is done correctly.
        """

        # Create inputs
        a = np.arange(10, dtype=self.get_dtype()) * 3
        b = np.arange(10, dtype=self.get_dtype()) * 2 + 5

        # Check ref count for input. Numpy arrays have two references.
        self.assertEqual(sys.getrefcount(a), 2)
        self.assertEqual(sys.getrefcount(b), 2)

        # Create weak refs for inputs
        weak_a = weakref.ref(a)
        weak_b = weakref.ref(b)

        self.assertEqual(sys.getrefcount(a), 2)
        self.assertEqual(sys.getrefcount(b), 2)
        self.assertIsNotNone(weak_a())
        self.assertIsNotNone(weak_b())

        ## Intersect
        i = snp.intersect(a, b)

        self.assertEqual(sys.getrefcount(a), 2)
        self.assertEqual(sys.getrefcount(b), 2)
        self.assertEqual(sys.getrefcount(i), 2)

        # Create weakref for i
        weak_i = weakref.ref(i)
        self.assertEqual(sys.getrefcount(a), 2)
        self.assertEqual(sys.getrefcount(b), 2)
        self.assertEqual(sys.getrefcount(i), 2)
        self.assertIsNotNone(weak_a())
        self.assertIsNotNone(weak_b())
        self.assertIsNotNone(weak_i())

        # Delete a
        del a
        self.assertEqual(sys.getrefcount(b), 2)
        self.assertEqual(sys.getrefcount(i), 2)
        self.assertIsNone(weak_a())
        self.assertIsNotNone(weak_b())
        self.assertIsNotNone(weak_i())

        # Delete b
        del b
        self.assertEqual(sys.getrefcount(i), 2)
        self.assertIsNone(weak_a())
        self.assertIsNone(weak_b())
        self.assertIsNotNone(weak_i())

        # Delete i
        del i
        self.assertIsNone(weak_a())
        self.assertIsNone(weak_b())
        self.assertIsNone(weak_i())

    def test_reference_counting_early_exit_type(self):
        """
        Check that the reference counts of the input arrary does not change
        even when the the method exists premature due to incompatible inputs
        types.
        """
        a = np.array(10)

        self.assertEqual(sys.getrefcount(a), 2)
        self.assertRaises(TypeError, snp.intersect, a, [1, 2])
        self.assertEqual(sys.getrefcount(a), 2)

        self.assertEqual(sys.getrefcount(a), 2)
        self.assertRaises(TypeError, snp.intersect, [1, 2], a)
        self.assertEqual(sys.getrefcount(a), 2)

    def test_reference_counting_early_exit_dim(self):
        """
        Check that the reference counts of the input arrary does not change
        even when the the method exists premature due multidimensional input
        arrays.
        """
        a = np.zeros((10, 2))
        b = np.arange(10)

        self.assertEqual(sys.getrefcount(a), 2)
        self.assertEqual(sys.getrefcount(b), 2)
        self.assertRaises(ValueError, snp.intersect, a, b)
        self.assertEqual(sys.getrefcount(a), 2)
        self.assertEqual(sys.getrefcount(b), 2)

        self.assertEqual(sys.getrefcount(a), 2)
        self.assertRaises(ValueError, snp.intersect, b, a)
        self.assertEqual(sys.getrefcount(a), 2)
        self.assertEqual(sys.getrefcount(b), 2)


class IntersectTestCase_Double(IntersectBase, unittest.TestCase):
    def get_dtype(self):
        return 'float'


