
#include "gtest/gtest.h"

#include <Python.h>
#define NPY_NO_DEPRECATED_API NPY_1_12_API_VERSION
#include <numpy/arrayobject.h>

#include "sortednpmodule.h"

typedef ::testing::Types<uint8_t, uint16_t, uint32_t, uint64_t, int8_t, int16_t,
  int32_t, int64_t, float, double> SupportedTypes;

////////////////////////////////////////////////////////////////////////////////
// Definition of helper functions

// Helper method which returns the numpy typenum associated with the type of
// the argument. This method should be used to determined the appropriate
// data type when creating numpy arrays.
template <typename T>
int getNpyType();

template<> int getNpyType<uint8_t>()  { return NPY_UINT8;   }
template<> int getNpyType<uint16_t>() { return NPY_UINT16;  }
template<> int getNpyType<uint32_t>() { return NPY_UINT32;  }
template<> int getNpyType<uint64_t>() { return NPY_UINT64;  }
template<> int getNpyType<int8_t>()   { return NPY_INT8;    }
template<> int getNpyType<int16_t>()  { return NPY_INT16;   }
template<> int getNpyType<int32_t>()  { return NPY_INT32;   }
template<> int getNpyType<int64_t>()  { return NPY_INT64;   }
template<> int getNpyType<float>()    { return NPY_FLOAT32; }
template<> int getNpyType<double>()   { return NPY_FLOAT64; }

// Helper method to easily convert a c-style array to a numpy array. The
// function creates a new numpy array of the given length and fills it with
// the given values. The type of the array is determined by getNpyType(). The
// ownership of the new array is passed to the called, i.e. the caller should
// call Py_DECREF at the end of the test case.
template <class T>
PyArrayObject *toArray(int len, T *values) {
    // Create new array
    npy_intp dims[1] = {len};
    int type = getNpyType<T>();
    PyObject *obj = PyArray_SimpleNew(1, dims,  type);
    PyArrayObject *arr = (PyArrayObject*) PyArray_FROM_OF(obj, NPY_ARRAY_CARRAY_RO);

    // Fill values
    for (npy_intp i = 0; i < len; i++) {
        *(T*) PyArray_GETPTR1(arr, i) = values[i];
    }

    return arr;
}

// Fill the format pointer with a valid sprintf format, which matches the
// template type.
const char INT_FORMAT[] = "%d";
const char FLOAT_FORMAT[] = "%g";
template <typename T>
const char *getFormatString() { return INT_FORMAT; }

template <>
const char *getFormatString<float>() { return FLOAT_FORMAT; }

template <>
const char *getFormatString<double>() { return FLOAT_FORMAT; }

// Return the array as a string, similar to pythons list representation, the
// template type is used to access to array items and to determine the format
// string.
template <class T>
std::string toString(PyArrayObject* array) {
    std::string out = "[";
    npy_intp len = PyArray_DIMS(array)[0];
    char buffer[255];
    const char *format;

    for (npy_intp i = 0; i < len; i++) {
        format = getFormatString<T>();
        sprintf(buffer, format, *(T*) PyArray_GETPTR1(array, i));
        out.append(buffer);

        if (i + 1 < len) {
            // if not last element
            out.append(", ");
        }
    }
    out.append("]");
    return out;
}

////////////////////////////////////////////////////////////////////////////////
// Test of helper functions, i.e. self-test


template <class T>
class SelfTest : public ::testing::Test { };

TYPED_TEST_CASE(SelfTest, SupportedTypes);

// Test the helper method, whether it indeed returns the correct numpy typenum
// for a few selected types.
TEST(SelfTest, getNpyType) {
    EXPECT_EQ(getNpyType<uint8_t>(), NPY_UINT8);
    EXPECT_EQ(getNpyType<int16_t>(), NPY_INT16);
    EXPECT_EQ(getNpyType<uint32_t>(), NPY_UINT32);
    EXPECT_EQ(getNpyType<int64_t>(), NPY_INT64);
    EXPECT_EQ(getNpyType<float>(), NPY_FLOAT32);
    EXPECT_EQ(getNpyType<double>(), NPY_FLOAT64);
}

// Check that the helper function toArray returns the correct arrays from a
// field of different sizes.
TYPED_TEST(SelfTest, toArray) {
  {
    // length 3
    const int len = 3;
    TypeParam values[len] = {4, 2, 9};

    PyArrayObject *arr = toArray(len, values);
    ASSERT_TRUE(arr != NULL);
    EXPECT_EQ(len, PyArray_DIMS(arr)[0]);

    EXPECT_EQ(4, *(TypeParam*) PyArray_GETPTR1(arr, 0));
    EXPECT_EQ(2, *(TypeParam*) PyArray_GETPTR1(arr, 1));
    EXPECT_EQ(9, *(TypeParam*) PyArray_GETPTR1(arr, 2));

    Py_DECREF(arr);
  }
  {
    // length 0
    const int len = 0;
    TypeParam values[1] = {49};
    PyArrayObject *arr = toArray(len, values);
    ASSERT_TRUE(arr != NULL);
    EXPECT_EQ(len, PyArray_DIMS(arr)[0]);

    Py_DECREF(arr);
  }
}

// Test the helper method which sets a format string based on the template
// type.
TEST(SelfTest, getFormatString) {
    const char *format;
    
    format = getFormatString<uint8_t>();
    EXPECT_STREQ("%d", format);

    format = getFormatString<int16_t>();
    EXPECT_STREQ("%d", format);

    format = getFormatString<uint32_t>();
    EXPECT_STREQ("%d", format);

    format = getFormatString<int64_t>();
    EXPECT_STREQ("%d", format);

    format = getFormatString<float>();
    EXPECT_STREQ("%g", format);

    format = getFormatString<double>();
    EXPECT_STREQ("%g", format);

}

// Check that arrays of different lengths are correctly converted to
// strings.
TYPED_TEST(SelfTest, toString) {
  {
    // length 3
    const int len = 3;
    TypeParam values[len] = {4, 34, 9};
    PyArrayObject *arr = toArray(len, values);
    EXPECT_EQ("[4, 34, 9]", toString<TypeParam>(arr));
    Py_DECREF(arr);
  }
  {
    // length 0
    const int len = 0;
    TypeParam values[1] = {49};
    PyArrayObject *arr = toArray(len, values);
    EXPECT_EQ("[]", toString<TypeParam>(arr));
    Py_DECREF(arr);
  }
}


////////////////////////////////////////////////////////////////////////////////
// Basic tests

// The basic ability to create a numpy array and manipulate its data. This is
// more like a self test, or a proof of principle test. If this test fail,
// there should be no way, that the other tests succeed.
TEST(basic, create_numpy) {
    // New array should have on axis with two elements.
    npy_intp dims[1] = {2};

    PyObject *obj = PyArray_SimpleNew(1, dims,  NPY_FLOAT64);
    ASSERT_TRUE(obj != NULL);  // Assert existence of new array

    PyArrayObject *arr = (PyArrayObject*) PyArray_FROM_OF(obj, NPY_ARRAY_CARRAY_RO);
    ASSERT_TRUE(arr != NULL);  // Assert obtaining PyArray pointer worked

    // Fill the array
    *(double*) PyArray_GETPTR1(arr, 0) = 42;
    *(double*) PyArray_GETPTR1(arr, 1) = 67;

    // Check array contents
    double *p; 
    p = (double*) PyArray_GETPTR1(arr, 0);
    EXPECT_EQ(42, *p);

    p = (double*) PyArray_GETPTR1(arr, 1);
    EXPECT_EQ(67, *p);

    Py_DECREF(obj);
}



// Main function. Initialize Google Test, Python and numpy. Finally run all
// test cases.
int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);

    Py_Initialize();
    import_array();

    return RUN_ALL_TESTS();
}
