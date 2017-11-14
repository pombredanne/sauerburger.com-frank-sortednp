
#include <Python.h>

#define NPY_NO_DEPRECATED_API NPY_1_12_API_VERSION

#include <numpy/arrayobject.h>

#define SEARCH_METHOD simple_search

template <class T>
void galloping_serach(T target, PyArrayObject *array, npy_intp &i, npy_intp &len) {
    // Advance the pointer in order to find the given value in the sorted array.
    // If the value if found, array index is at the first occurrence of value.
    // If the value is not found, the array index is at the first item larger
    // than the search value, or beyond the array limits.
    T value = *((T*) PyArray_GETPTR1(array, i));

    // If already at correct location or beyond
    if (target <= value) {
      return;
    }

    npy_intp delta = 1;
    npy_intp i_prev = i;
    while (value < target) {
        i_prev = i;
        i += delta;
        if (len <= i) {
            i = len;
            value = *((T*) PyArray_GETPTR1(array, i));
            break;
        }
        value = *((T*) PyArray_GETPTR1(array, i));
        delta *= 2;
    }

    binary_search(target, array, i_prev, i);
}

template <class T>
void binary_search(T target, PyArrayObject *array, npy_intp &i, npy_intp &len) {
    // Advance the pointer in order to find the given value in the sorted array.
    // If the value if found, array index is at the first occurrence of value.
    // If the value is not found, the array index is at the first item larger
    // than the search value, or beyond the array limits.
    T value = *((T*) PyArray_GETPTR1(array, i));

    // If already at correct location or beyond
    if (target <= value) {
      return;
    }

    npy_intp i_right = len - 1;  // is always GREATER OR EQUAL
    npy_intp i_left = i;  // is always LESS than value


    T right = *((T*) PyArray_GETPTR1(array, i_right));
    if (right < target) {
      i = i_right + 1;  // move beyond array bounds
      return;
    }

    while (i_left + 1 < i_right) {
        i = (i_right + i_left) / 2;
        value = *((T*) PyArray_GETPTR1(array, i));

        if (target <= value) {
          i_right = i;
        } else {
          i_left = i;
        }
    }

    i = i_right;
}

template <class T>
void simple_search(T target, PyArrayObject *array, npy_intp &i, npy_intp &len) {
    // Advance the pointer in order to find the given value in the sorted array.
    // If the value if found, array index is at the first occurrence of value.
    // If the value is not found, the array index is at the first item larger
    // than the search value, or beyond the array limits.
    T value = *((T*) PyArray_GETPTR1(array, i));

    while (value < target && i < len) {
        i++;  // potentially move beyond the array bounds
        value = *((T*) PyArray_GETPTR1(array, i));
    }
}


/**
 * Internal helper method.
 *
 * Calculate the intersection for the given arrays and the given type. The
 * method returns the pointer to the newly created output array. The first
 * parameter is used to call the templated function.
 */
template <class T>
PyObject* intersect(T template_type, PyArrayObject *a_array, PyArrayObject *b_array) {
    // Since the size of the intersection array can not be known in advance we
    // need to create an array of at least the size of the smaller array.
    npy_intp len_a = PyArray_DIMS(a_array)[0];
    npy_intp len_b = PyArray_DIMS(b_array)[0];
    npy_intp new_dim[1] = {len_a < len_b ? len_a : len_b};

    // Creating the new array sets the reference counter to 1 and passes the
    // ownership of the returned reference to the caller. The method steals the
    // type descriptor, which is why we have to increment its count before
    // calling the method.
    PyArray_Descr* type = PyArray_DESCR(a_array);
    Py_INCREF(type);
    PyObject *out;
    out = PyArray_SimpleNewFromDescr(1, new_dim, type);
    if (out == NULL) {
        // Probably a memory error occurred.
        return NULL;
    }
    PyArrayObject *out_array = (PyArrayObject*) out;

    npy_intp i_a = 0;
    npy_intp i_b = 0;
    npy_intp i_o = 0;
    T v_a = *((T*) PyArray_GETPTR1(a_array, i_a));
    T v_b = *((T*) PyArray_GETPTR1(b_array, i_b));

    // Actual computation of the intersection.
    while (i_a < len_a && i_b < len_b) {
        if (v_a < v_b) {
            SEARCH_METHOD(v_b, a_array, i_a, len_a);
            v_a = *((T*) PyArray_GETPTR1(a_array, i_a));
        } else if (v_b < v_a) {
            SEARCH_METHOD(v_a, b_array, i_b, len_b);
            v_b = *((T*) PyArray_GETPTR1(b_array, i_b));
        }

        if (v_a == v_b) {
            T *t = (T*) PyArray_GETPTR1(out_array, i_o);
            *t = v_a;

            i_o++;
            i_a++;
            i_b++;

            v_a = *((T*) PyArray_GETPTR1(a_array, i_a));
            v_b = *((T*) PyArray_GETPTR1(b_array, i_b));
        }
    }

    // Resize the array after intersect operation.
    new_dim[0] = i_o;
    PyArray_Dims dims;
    dims.ptr = new_dim;
    dims.len = 1;
    PyArray_Resize(out_array, &dims, 0, NPY_CORDER);

    return out;
}

/*
 * The sortednp_intersect function expects exactly two references to sorted
 * arrays as positional arguments. The function borrows the references. The
 * return value is a reference to a new sorted numpy array containing only
 * elements common in both arrays. The ownership of the returned reference is
 * passed to the caller.
 */
static PyObject *sortednp_intersect(PyObject *self, PyObject *args) {
    // References to the arguments are borrowed. Counter should not be
    // incremented since input arrays are not stored.

    PyObject *a, *b;

    // PyArg_ParseTuple returns borrowed references. This is fine, the input
    // arrays are not stored.
    if (!PyArg_ParseTuple(args, "O!O!", &PyArray_Type, &a, &PyArray_Type, &b)) {
        // Reference counters have not been changed.
        return NULL;
    }

    // PyArray_FROM_OF steals a reference according to 
    // https://groups.google.com/forum/#!topic/numpy/Rg9BpWoc37U. This usually
    // means that the reference count is decremented in the end. However the
    // method also returns a reference. It is not clear , whether ew own the
    // reference or not.
    // 
    // A quick investigation showed that Py_REFCNT(a) returns 2 before the
    // PyArray_FROM_OF call, and 3 after. This indicates that the method creates
    // a new reference to the array for use and passes ownership of the
    // references to use. To keep the reference count constant we have to
    // decrement it.
    a = PyArray_FROM_OF(a, NPY_ARRAY_CARRAY_RO);
    b = PyArray_FROM_OF(b, NPY_ARRAY_CARRAY_RO);

    // Decrement reference counter of the non-null array, see above comment
    // block.
    Py_XDECREF(a);
    Py_XDECREF(b);

    if (a == NULL || b == NULL) {
      // Reference counter of input arrays have been fixed. It is safe to exit.
        return NULL;
    }

    // Some methods need a PyObject* other nee a PyArrayObject*.
    PyArrayObject *a_array = (PyArrayObject*) a;
    PyArrayObject *b_array = (PyArrayObject*) b;

    if (PyArray_NDIM(a_array) != 1 || PyArray_NDIM(b_array) != 1) {
      PyErr_SetString(PyExc_ValueError, "Arguments can not be multi-dimensional.");
      // Reference counter of input arrays have been fixed. It is safe to exit.
      return NULL;
    }

    if (PyArray_TYPE(a_array) != PyArray_TYPE(b_array)) {
      PyErr_SetString(PyExc_ValueError, "Arguments must have the same data type.");
      // Reference counter of input arrays have been fixed. It is safe to exit.
      return NULL;
    }


    PyObject* out;

    // Use the dummy variables to call the typed intersect method.
    int8_t dtype_int8 = 0;
    int16_t dtype_int16 = 0;
    int32_t dtype_int32 = 0;
    int64_t dtype_int64 = 0;
    uint8_t dtype_uint8 = 0;
    uint16_t dtype_uint16 = 0;
    uint32_t dtype_uint32 = 0;
    uint64_t dtype_uint64 = 0;
    float dtype_float32 = 0;
    double dtype_float64 = 0;

    // Differentiate between different data types.
    switch (PyArray_TYPE(a_array)) {
        case NPY_INT8:
          out = intersect(dtype_int8, a_array, b_array);
          break;
        case NPY_INT16:
          out = intersect(dtype_int16, a_array, b_array);
          break;
        case NPY_INT32:
          out = intersect(dtype_int32, a_array, b_array);
          break;
        case NPY_INT64:
          out = intersect(dtype_int64, a_array, b_array);
          break;
        case NPY_UINT8:
          out = intersect(dtype_uint8, a_array, b_array);
          break;
        case NPY_UINT16:
          out = intersect(dtype_uint16, a_array, b_array);
          break;
        case NPY_UINT32:
          out = intersect(dtype_uint32, a_array, b_array);
          break;
        case NPY_UINT64:
          out = intersect(dtype_uint64, a_array, b_array);
          break;
        case NPY_FLOAT32:
          out = intersect(dtype_float32, a_array, b_array);
          break;
        case NPY_FLOAT64:
          out = intersect(dtype_float64, a_array, b_array);
          break;
        default:
          PyErr_SetString(PyExc_ValueError, "Data type not supported.");
          // Reference counter of input arrays have been fixed. It is safe to exit.
          return NULL;
    }

    // Passes ownership of the returned reference to the  caller.
    return out;
}

/**
 * Internal helper method.
 *
 * Calculate the union for the given arrays and the given type. The
 * method returns the pointer to the newly created output array. The first
 * parameter is used to call the templated function.
 */
template <class T>
PyObject* merge(T template_type, PyArrayObject *a_array, PyArrayObject *b_array) {
    // Since the size of the merged array can not be known in advance we
    // need to create an array of at least the size of the concatenation of both
    // arrays.
    npy_intp len_a = PyArray_DIMS(a_array)[0];
    npy_intp len_b = PyArray_DIMS(b_array)[0];
    npy_intp new_dim[1] = {len_a + len_b};

    // Creating the new array sets the reference counter to 1 and passes the
    // ownership of the returned reference to the caller. The method steals the
    // type descriptor, which is why we have to increment its count before
    // calling the method.
    PyArray_Descr* type = PyArray_DESCR(a_array);
    Py_INCREF(type);
    PyObject *out;
    out = PyArray_SimpleNewFromDescr(1, new_dim, type);
    if (out == NULL) {
        // Probably a memory error occurred.
        return NULL;
    }
    PyArrayObject* out_array = (PyArrayObject*) out;

    npy_intp i_a = 0;
    npy_intp i_b = 0;
    npy_intp i_o = 0;
    T v_a = *((T*) PyArray_GETPTR1(a_array, i_a));
    T v_b = *((T*) PyArray_GETPTR1(b_array, i_b));

    // Actually merging the arrays.
    while (i_a < len_a && i_b < len_b) {
        T *t = (T*) PyArray_GETPTR1(out_array, i_o);

        if (v_a < v_b) {
            *t = v_a;
            i_a++;
            i_o++;
            v_a = *((T*) PyArray_GETPTR1(a_array, i_a));
        } else {
            *t = v_b;
            i_b++;
            i_o++;
            v_b = *((T*) PyArray_GETPTR1(b_array, i_b));
        }
    }

    // If the end of one of the two arrays has been reached in the above loop,
    // we need to copy all the elements left the array to the output.
    while (i_a < len_a) {
        T v_a = *((T*) PyArray_GETPTR1(a_array, i_a));
        T *t = (T*) PyArray_GETPTR1(out_array, i_o);
        *t = v_a;
        i_a++;
        i_o++;
    }
    while (i_b < len_b) {
        T v_b = *((T*) PyArray_GETPTR1(b_array, i_b));
        T *t = (T*) PyArray_GETPTR1(out_array, i_o);
        *t = v_b;
        i_b++;
        i_o++;
    }

    return out;
}

/*
 * The sortednp_merge function expects exactly two references to sorted
 * arrays as positional arguments. The function borrows the references. The
 * return value is a reference to a new sorted numpy array containing all
 * elements of the input arrays. The ownership of the returned reference is
 * passed to the caller.
 */
static PyObject *sortednp_merge(PyObject *self, PyObject *args) {
    // References to the arguments are borrowed. Counter should not be
    // incremented since input arrays are not stored.

    PyObject *a, *b;

    // PyArg_ParseTuple returns borrowed references. This is fine, the input
    // arrays are not stored.
    if (!PyArg_ParseTuple(args, "O!O!", &PyArray_Type, &a, &PyArray_Type, &b))
        // Reference counters have not been changed.
        return NULL;

    // PyArray_FROM_OF steals a reference according to 
    // https://groups.google.com/forum/#!topic/numpy/Rg9BpWoc37U. This usually
    // means that the reference count is decremented in the end. However the
    // method also returns a reference. It is not clear , whether ew own the
    // reference or not.
    // 
    // A quick investigation showed that Py_REFCNT(a) returns 2 before the
    // PyArray_FROM_OF call, and 3 after. This indicates that the method creates
    // a new reference to the array for use and passes ownership of the
    // references to use. To keep the reference count constant we have to
    // decrement it.
    a = PyArray_FROM_OF(a, NPY_ARRAY_CARRAY_RO);
    b = PyArray_FROM_OF(b, NPY_ARRAY_CARRAY_RO);

    // Decrement reference counter of the non-null array, see above comment
    // block.
    Py_XDECREF(a);
    Py_XDECREF(b);

    if (a == NULL || b == NULL) {
      // Reference counter of input arrays have been fixed. It is safe to exit.
        return NULL;
    }

    // Some methods need a PyObject* other nee a PyArrayObject*.
    PyArrayObject *a_array = (PyArrayObject*) a;
    PyArrayObject *b_array = (PyArrayObject*) b;


    if (PyArray_NDIM(a_array) != 1 || PyArray_NDIM(b_array) != 1) {
      PyErr_SetString(PyExc_ValueError, "Arguments can not be multi-dimensional.");
      // Reference counter of input arrays have been fixed. It is safe to exit.
      return NULL;
    }

    if (PyArray_TYPE(a_array) != PyArray_TYPE(b_array)) {
      PyErr_SetString(PyExc_ValueError, "Arguments must have the same data type.");
      // Reference counter of input arrays have been fixed. It is safe to exit.
      return NULL;
    }

    PyObject* out;

    // Use the dummy variables to call the typed merge method.
    int8_t dtype_int8 = 0;
    int16_t dtype_int16 = 0;
    int32_t dtype_int32 = 0;
    int64_t dtype_int64 = 0;
    uint8_t dtype_uint8 = 0;
    uint16_t dtype_uint16 = 0;
    uint32_t dtype_uint32 = 0;
    uint64_t dtype_uint64 = 0;
    float dtype_float32 = 0;
    double dtype_float64 = 0;

    // Differentiate between different data types.
    switch (PyArray_TYPE(a_array)) {
        case NPY_INT8:
          out = merge(dtype_int8, a_array, b_array);
          break;
        case NPY_INT16:
          out = merge(dtype_int16, a_array, b_array);
          break;
        case NPY_INT32:
          out = merge(dtype_int32, a_array, b_array);
          break;
        case NPY_INT64:
          out = merge(dtype_int64, a_array, b_array);
          break;
        case NPY_UINT8:
          out = merge(dtype_uint8, a_array, b_array);
          break;
        case NPY_UINT16:
          out = merge(dtype_uint16, a_array, b_array);
          break;
        case NPY_UINT32:
          out = merge(dtype_uint32, a_array, b_array);
          break;
        case NPY_UINT64:
          out = merge(dtype_uint64, a_array, b_array);
          break;
        case NPY_FLOAT32:
          out = merge(dtype_float32, a_array, b_array);
          break;
        case NPY_FLOAT64:
          out = merge(dtype_float64, a_array, b_array);
          break;
        default:
          PyErr_SetString(PyExc_ValueError, "Data type not supported.");
          // Reference counter of input arrays have been fixed. It is safe to exit.
          return NULL;
    }

    // Passes ownership of the returned reference to the  caller.
    return out;
}

// Define list of methods in the module.
static PyMethodDef SortedNpMethods[] = {
    {"merge",  sortednp_merge, METH_VARARGS, "Merge two sorted numpy arrays."},
    {"intersect",  sortednp_intersect, METH_VARARGS, "Intersect two sorted numpy arrays."},
    {NULL, NULL, 0, NULL}  // Sentinel
};

// Define module itself.
static struct PyModuleDef sortednpmodule = {
    PyModuleDef_HEAD_INIT,
    "_sortednp",  // Name of the module
    NULL,  // Module docstring
    -1,  // The module keeps state in global variables.
    SortedNpMethods
};

// Init method
PyMODINIT_FUNC PyInit__sortednp(void) {
    import_array();
    return PyModule_Create(&sortednpmodule);
}
