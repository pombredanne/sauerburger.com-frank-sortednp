
#include "gtest/gtest.h"

#include <Python.h>
#define NPY_NO_DEPRECATED_API NPY_1_12_API_VERSION
#include <numpy/arrayobject.h>

#include "sortednpmodule.h"


TEST(Basic, create_numpy) {
    npy_intp dims[1] = {2};

    PyObject *obj = PyArray_SimpleNew(1, dims,  NPY_FLOAT64);
    ASSERT_TRUE(obj != NULL);

    PyArrayObject *arr = (PyArrayObject*) PyArray_FROM_OF(obj, NPY_ARRAY_CARRAY_RO);
    ASSERT_TRUE(arr != NULL);

    double *p;

    p = (double*) PyArray_GETPTR1(arr, 0);
    *p = 42;

    p = (double*) PyArray_GETPTR1(arr, 1);
    *p = 67;

    //
    p = (double*) PyArray_GETPTR1(arr, 0);
    EXPECT_EQ(42, *p);

    p = (double*) PyArray_GETPTR1(arr, 1);
    EXPECT_EQ(67, *p);
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);

    Py_Initialize();
    import_array();

    return RUN_ALL_TESTS();
}

