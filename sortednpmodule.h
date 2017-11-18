
#ifndef SORTEDNPMODULE_H
#define SORTEDNPMODULE_H

#include <Python.h>

#define NPY_NO_DEPRECATED_API NPY_1_12_API_VERSION
#include <numpy/arrayobject.h>

template <class T>
void galloping_serach(T target, PyArrayObject *array, npy_intp &i, npy_intp &len);

template <class T>
void binary_search(T target, PyArrayObject *array, npy_intp &i, npy_intp &len);

template <class T>
void simple_search(T target, PyArrayObject *array, npy_intp &i, npy_intp &len);

template <class T>
PyObject* intersect(PyArrayObject *a_array, PyArrayObject *b_array);

PyObject *sortednp_intersect(PyObject *self, PyObject *args);

template <class T>
PyObject* merge(PyArrayObject *a_array, PyArrayObject *b_array);

PyObject *sortednp_merge(PyObject *self, PyObject *args);


#endif  // SORTEDNPMODULE_H
