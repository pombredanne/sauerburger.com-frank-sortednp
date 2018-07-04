// Copyright (C) 2016, Frank Sauerburger
// Sortednp module

#ifndef SRC_SORTEDNPMODULE_H_
#define SRC_SORTEDNPMODULE_H_

#include <Python.h>

#ifndef NPY_NO_DEPRECATED_API
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#endif

#include <numpy/arrayobject.h>

enum method_t {SIMPLE, BINARY, GALLOPPING};

/**
 * Advance the pointer in order to find the given value in the sorted array.  If
 * the value is found, the array index is at the first occurrence of the value
 * and false is returned. If the given values smaller than any other value in
 * the array, the pointer is not advanced and false is returned. If the given
 * value is larger than any other value in the array, the pointer is at the
 * last item of the array and true is returned.
 *
 * The return value is true, if the given value is too large to be in the array.
 * If the search is used in an intersection, true indicates the end of the
 * intersection.
 *
 * The implementation checks each element one-by-one to find the given target
 * value.
 */
template <class T>
bool simple_search(T target, PyArrayObject *array, npy_intp *i,
    const npy_intp &len) {
  for (; *i < len; (*i)++) {
    T value = *(reinterpret_cast<T*>(PyArray_GETPTR1(array, *i)));

    if (target < value) {
      // Search value is too small
      return false;
    } else if (value == target) {
      // Value found
      return false;
    }
  }

  (*i)--;  // Pointer moved beyond limits of array. Move it back.

  // Reached the end of array without finding the value
  return true;
}

/**
 * The interface is identical to simple_search.
 *
 * The implementation searches the remaining array (between i and len) by
 * performing binary splits. The same procedure is applied iteratively to one of
 * the two half until the target value is found.
 */
template <class T>
bool binary_search(T target, PyArrayObject *array, npy_intp *i,
    const npy_intp &len) {
  T value = *(reinterpret_cast<T*>(PyArray_GETPTR1(array, *i)));

  // If already at correct location or beyond
  if (target <= value) {
    return false;
  }

  npy_intp i_right = len - 1;  // is always GREATER OR EQUAL
  npy_intp i_left = *i;  // is always LESS than value

  T right = *(reinterpret_cast<T*>(PyArray_GETPTR1(array, i_right)));
  if (right < target) {
    *i = i_right;
    return true;  // indicate target value too large
  }

  while (i_left + 1 < i_right) {
    *i = (i_right + i_left) / 2;
    value = *(reinterpret_cast<T*>(PyArray_GETPTR1(array, *i)));

    if (target <= value) {
      i_right = *i;
    } else {
      i_left = *i;
    }
  }

  *i = i_right;
  return false;
}

/**
 * The interface is identical to simple_search.
 *
 * The implementation searchs the remaining array sequentially with increasing
 * step size (times two). If the current value is larger than the target value,
 * preform a binary search for all values encloses by the last step.
 */
template <class T>
bool galloping_search(T target, PyArrayObject *array, npy_intp *i,
    const npy_intp &len) {
  T value = *(reinterpret_cast<T*>(PyArray_GETPTR1(array, *i)));

  // If already at correct location or beyond
  if (target <= value) {
    return false;
  }

  npy_intp delta = 1;
  npy_intp i_prev = *i;

  while (value < target) {
    i_prev = *i;
    *i += delta;
    if (len <= *i) {
      // Gallop jump reached end of array.
      *i = len - 1;
      value = *(reinterpret_cast<T*>(PyArray_GETPTR1(array, *i)));
      break;
    }

    value = *(reinterpret_cast<T*>(PyArray_GETPTR1(array, *i)));
    // Increase step size.
    delta *= 2;
  }

  npy_intp higher = *i;
  higher++;  // Convert pointer position to length.
  *i = i_prev;  // This is the lower boundary and the active counter.

  return binary_search(target, array, i, higher);
}


template <class T>
PyObject* intersect(PyArrayObject *a_array, PyArrayObject *b_array,
  method_t search_method);
PyObject* sortednp_intersect(PyObject *self, PyObject *args, PyObject *kwds);

template <class T>
PyObject* merge(PyArrayObject *a_array, PyArrayObject *b_array);

PyObject* sortednp_merge(PyObject *self, PyObject *args);


#endif  // SRC_SORTEDNPMODULE_H_
