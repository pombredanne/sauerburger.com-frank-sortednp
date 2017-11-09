
from abc import ABCMeta, abstractmethod
import sys
import weakref
import unittest
import numpy as np

import sortednp as snp

class MergeBase(metaclass=ABCMeta):
    """
    Define general test cases for the merge method. Sub-classes need to
    implement have to overwrite the dtype method.
    """

    @abstractmethod
    def get_dtype(self):
        """
        Returns the numpy data type, which should be used for all tests.
        """
        pass
    

    def test_simple(self):
        """
        Check that merging two non-empty arrays returns the union of the two
        arrays.
        """
        a = np.array([1, 3, 7], dtype=self.get_dtype())
        b = np.array([2, 5, 6], dtype=self.get_dtype())

        m = snp.merge(a, b)
        self.assertEqual(list(m), [1, 2, 3, 5, 6, 7])
        self.assertEqual(m.dtype, self.get_dtype())

    def test_separated(self):
        """
        Check that merging two non-empty arrays returns the union of the two
        arrays if all element in on array are greater than all elements in the
        other. This tests the copy parts of the implementation.
        """
        a = np.array([1, 3, 7], dtype=self.get_dtype())
        b = np.array([9, 10, 16], dtype=self.get_dtype())

        m = snp.merge(a, b)
        self.assertEqual(list(m), [1, 3, 7, 9, 10, 16])
        self.assertEqual(m.dtype, self.get_dtype())

    def test_empty_single(self):
        """
        Check that merging two arrays returns a copy of the first one if
        the other is empty.
        """
        a = np.array([1, 3, 7], dtype=self.get_dtype())
        b = np.array([], dtype=self.get_dtype())

        m = snp.merge(a, b)
        self.assertEqual(list(m), [1, 3, 7])
        self.assertEqual(list(a), [1, 3, 7])
        self.assertEqual(m.dtype, self.get_dtype())
        m[0] = 0
        self.assertEqual(list(a), [1, 3, 7])

        m = snp.merge(b, a)
        self.assertEqual(list(m), [1, 3, 7])
        self.assertEqual(list(a), [1, 3, 7])
        self.assertEqual(m.dtype, self.get_dtype())
        m[0] = 0
        self.assertEqual(list(a), [1, 3, 7])


    def test_empty_both(self):
        """
        Check that merging two empty arrays returns an empty array.
        """
        a = np.array([], dtype=self.get_dtype())
        b = np.array([], dtype=self.get_dtype())

        m = snp.merge(a, b)
        self.assertEqual(list(m), [])
        self.assertEqual(len(m), 0)
        self.assertEqual(m.dtype, self.get_dtype())


    def test_identical(self):
        """
        Check that merging two identical arrays returns each element twice.
        """
        a = np.array([1, 3, 7], dtype=self.get_dtype())

        m = snp.merge(a, a)
        self.assertEqual(list(m), [1, 1, 3, 3, 7, 7])
        self.assertEqual(m.dtype, self.get_dtype())


    def test_duplicates_same(self):
        """
        Check that duplications in a single array are passed to the result.
        """
        a = np.array([1, 3, 3, 7], dtype=self.get_dtype())
        b = np.array([2, 5, 6], dtype=self.get_dtype())

        m = snp.merge(a, b)
        self.assertEqual(list(m), [1, 2, 3, 3, 5, 6, 7])
        self.assertEqual(m.dtype, self.get_dtype())

    def test_duplicates_other(self):
        """
        Check that duplications in the other array are passed to the result.
        """
        a = np.array([1, 3, 7], dtype=self.get_dtype())
        b = np.array([2, 3, 5, 6], dtype=self.get_dtype())

        m = snp.merge(a, b)
        self.assertEqual(list(m), [1, 2, 3, 3, 5, 6, 7])
        self.assertEqual(m.dtype, self.get_dtype())

    def test_duplicates_both(self):
        """
        Check that duplications in a single and the other array are both passed to
        the result.
        """
        a = np.array([1, 3, 3, 7], dtype=self.get_dtype())
        b = np.array([2, 3, 5, 6], dtype=self.get_dtype())

        m = snp.merge(a, b)
        self.assertEqual(list(m), [1, 2, 3, 3, 3, 5, 6, 7])
        self.assertEqual(m.dtype, self.get_dtype())
        

    def test_raise_multi_dim(self):
        """
        Check that passing in a multi dimensional array raises an exception.
        """
        a = np.zeros((10, 2), dtype=self.get_dtype())
        b = np.array([2, 3, 5, 6], dtype=self.get_dtype())

        self.assertRaises(ValueError, snp.merge, a, b)
        self.assertRaises(ValueError, snp.merge, b, a)
        self.assertRaises(ValueError, snp.merge, a, a)
        
    def test_raise_non_array(self):
        """
        Check that passing in a non-numpy-array raises an exception.
        """
        b = np.array([2, 3, 5, 6], dtype=self.get_dtype())

        self.assertRaises(TypeError, snp.merge, 3, b)
        self.assertRaises(TypeError, snp.merge, b, 2)
        self.assertRaises(TypeError, snp.merge, 3, "a")
        
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
        m = snp.merge(a, b)

        self.assertEqual(sys.getrefcount(a), 2)
        self.assertEqual(sys.getrefcount(b), 2)
        self.assertEqual(sys.getrefcount(m), 2)

        # Create weakref for m
        weak_m = weakref.ref(m)
        self.assertEqual(sys.getrefcount(a), 2)
        self.assertEqual(sys.getrefcount(b), 2)
        self.assertEqual(sys.getrefcount(m), 2)
        self.assertIsNotNone(weak_a())
        self.assertIsNotNone(weak_b())
        self.assertIsNotNone(weak_m())

        # Delete a
        del a
        self.assertEqual(sys.getrefcount(b), 2)
        self.assertEqual(sys.getrefcount(m), 2)
        self.assertIsNone(weak_a())
        self.assertIsNotNone(weak_b())
        self.assertIsNotNone(weak_m())

        # Delete b
        del b
        self.assertEqual(sys.getrefcount(m), 2)
        self.assertIsNone(weak_a())
        self.assertIsNone(weak_b())
        self.assertIsNotNone(weak_m())

        # Delete m
        del m
        self.assertIsNone(weak_a())
        self.assertIsNone(weak_b())
        self.assertIsNone(weak_m())

    def test_reference_counting_early_exit_type(self):
        """
        Check that the reference counts of the input arrary does not change
        even when the the method exists premature due to incompatible inputs
        types.
        """
        a = np.array(10)

        self.assertEqual(sys.getrefcount(a), 2)
        self.assertRaises(TypeError, snp.merge, a, [1, 2])
        self.assertEqual(sys.getrefcount(a), 2)

        self.assertEqual(sys.getrefcount(a), 2)
        self.assertRaises(TypeError, snp.merge, [1, 2], a)
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
        self.assertRaises(ValueError, snp.merge, a, b)
        self.assertEqual(sys.getrefcount(a), 2)
        self.assertEqual(sys.getrefcount(b), 2)

        self.assertEqual(sys.getrefcount(a), 2)
        self.assertRaises(ValueError, snp.merge, b, a)
        self.assertEqual(sys.getrefcount(a), 2)
        self.assertEqual(sys.getrefcount(b), 2)

class MergeTestCase_Double(MergeBase, unittest.TestCase):
    def get_dtype(self):
        return 'float'



