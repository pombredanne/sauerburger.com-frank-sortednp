
from abc import ABCMeta, abstractmethod
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
        

class MergeTestCase_Double(MergeBase, unittest.TestCase):
    def get_dtype(self):
        return 'float'



