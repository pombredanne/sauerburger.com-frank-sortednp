
#include <stdbool.h>
#include <Python.h>
#include <numpy/arrayobject.h>

// Turn debug information about reference counting on or off.
// #define REF_DEBUG 1

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

    #ifdef REF_DEBUG
        printf("\n -- n_ref(a, before FROM OF) = %d\n", (int) Py_REFCNT(a));  // returns 2
    #endif  // REF_DEBUG

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

    #ifdef REF_DEBUG
        printf(" -- n_ref(a, after DECREF)   = %d  (should be the same)\n", (int) Py_REFCNT(a));  // returns 2
    #endif  // REF_DEBUG

    int nd_a = PyArray_NDIM(a);
    int nd_b = PyArray_NDIM(b);

    if (PyArray_NDIM(a) != 1 || PyArray_NDIM(b) != 1) {
      PyErr_SetString(PyExc_ValueError, "Arguments can not be multi-dimensional.");
      // Reference counter of input arrays have been fixed. It is safe to exit.
      return NULL;
    }

    // Since the size of the intersection array can not be known in advance we
    // need to create an array of at least the size of the smaller array.
    npy_intp len_a = PyArray_DIMS(a)[0];
    npy_intp len_b = PyArray_DIMS(b)[0];
    npy_intp new_dim[1] = {len_a < len_b ? len_a : len_b};

    // Creating the new array sets the reference counter to 1 and passes the
    // ownership of the returned reference to the caller. The method steals the
    // type descriptor, which is why we have to increment its count before
    // calling the method.
    PyArray_Descr* type = PyArray_DESCR(a);
    #ifdef REF_DEBUG
        printf(" -- n_ref(t, before create)  = %d\n", (int) Py_REFCNT(type));
    #endif  // REF_DEBUG
    Py_INCREF(type);
    PyArrayObject *out;
    out = PyArray_SimpleNewFromDescr(1, new_dim, type);

    #ifdef REF_DEBUG
        printf(" -- n_ref(t, after create)   = %d  (shouldn't be less)\n", (int) Py_REFCNT(type));
        printf(" -- n_ref(i, after create)   = %d  (should be 1)\n", (int) Py_REFCNT(out));  // returns 1
    #endif  // REF_DEBUG
    
    npy_intp i_a = 0;
    npy_intp i_b = 0;
    npy_intp i_o = 0;
    double v_a = *((double*) PyArray_GETPTR1(a, i_a));
    double v_b = *((double*) PyArray_GETPTR1(b, i_b));

    // Actual computation of the intersection.
    while (i_a < len_a && i_b < len_b) {
        bool matched = false;
        if (v_a == v_b) {
          double *t = (double*) PyArray_GETPTR1(out, i_o);
          *t = v_a;

          i_o++;
          matched = true;
        }
        
        if (v_a < v_b || matched) {
            i_a++;
            v_a = *((double*) PyArray_GETPTR1(a, i_a));
        }

        if (v_b < v_a || matched) {
            i_b++;
            v_b = *((double*) PyArray_GETPTR1(b, i_b));
        }
    }

    #ifdef REF_DEBUG
        // Accessing pointers should not affect the reference count.
        printf(" -- n_ref(i, after fill)     = %d  (should be 1)\n", (int) Py_REFCNT(out));  // returns 1
    #endif  // REF_DEBUG
    
    // Resize the array after intersect operation.
    new_dim[0] = i_o;
    PyArray_Dims dims;
    dims.ptr = new_dim;
    dims.len = 1;
    PyArray_Resize(out, &dims, 0, NPY_CORDER);

    #ifdef REF_DEBUG
        // Resizing the array should not affect the reference count.
        printf(" -- n_ref(i, after resize)   = %d  (should be 1)\n", (int) Py_REFCNT(out));  // returns 1
    #endif  // REF_DEBUG

    // Passes ownership of the returned reference to the  caller.
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

    #ifdef REF_DEBUG
        printf("\n -- n_ref(a, before FROM OF) = %d\n", (int) Py_REFCNT(a));  // returns 2
    #endif  // REF_DEBUG

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

    #ifdef REF_DEBUG
        printf(" -- n_ref(a, after DECREF)   = %d  (should be the same)\n", (int) Py_REFCNT(a));  // returns 2
    #endif  // REF_DEBUG

    int nd_a = PyArray_NDIM(a);
    int nd_b = PyArray_NDIM(b);

    if (PyArray_NDIM(a) != 1 || PyArray_NDIM(b) != 1) {
      PyErr_SetString(PyExc_ValueError, "Arguments can not be multi-dimensional.");
      // Reference counter of input arrays have been fixed. It is safe to exit.
      return NULL;
    }

    // Since the size of the merged array can not be known in advance we
    // need to create an array of at least the size of the concatenation of both
    // arrays.
    npy_intp len_a = PyArray_DIMS(a)[0];
    npy_intp len_b = PyArray_DIMS(b)[0];
    npy_intp new_dim[1] = {len_a + len_b};

    // Creating the new array sets the reference counter to 1 and passes the
    // ownership of the returned reference to the caller. The method steals the
    // type descriptor, which is why we have to increment its count before
    // calling the method.
    PyArray_Descr* type = PyArray_DESCR(a);
    #ifdef REF_DEBUG
        printf(" -- n_ref(t, before create)  = %d\n", (int) Py_REFCNT(type));
    #endif  // REF_DEBUG
    Py_INCREF(type);
    PyArrayObject *out;
    out = PyArray_SimpleNewFromDescr(1, new_dim, type);

    #ifdef REF_DEBUG
        printf(" -- n_ref(t, after create)   = %d  (shouldn't be less)\n", (int) Py_REFCNT(type));
        printf(" -- n_ref(m, after create)   = %d  (should be 1)\n", (int) Py_REFCNT(out));  // returns 1
    #endif  // REF_DEBUG
    
    npy_intp i_a = 0;
    npy_intp i_b = 0;
    npy_intp i_o = 0;
    double v_a = *((double*) PyArray_GETPTR1(a, i_a));
    double v_b = *((double*) PyArray_GETPTR1(b, i_b));

    // Actually merging the arrays.
    while (i_a < len_a && i_b < len_b) {
        double *t = (double*) PyArray_GETPTR1(out, i_o);

        if (v_a < v_b) {
            *t = v_a;
            i_a++;
            i_o++;
            v_a = *((double*) PyArray_GETPTR1(a, i_a));
        } else {
            *t = v_b;
            i_b++;
            i_o++;
            v_b = *((double*) PyArray_GETPTR1(b, i_b));
        }
    }

    // If the end of one of the two arrays has been reached in the above loop,
    // we need to copy all the elements left the array to the output.
    while (i_a < len_a) {
        double v_a = *((double*) PyArray_GETPTR1(a, i_a));
        double *t = (double*) PyArray_GETPTR1(out, i_o);
        *t = v_a;
        i_a++;
        i_o++;
    }
    while (i_b < len_b) {
        double v_b = *((double*) PyArray_GETPTR1(b, i_b));
        double *t = (double*) PyArray_GETPTR1(out, i_o);
        *t = v_b;
        i_b++;
        i_o++;
    }

    #ifdef REF_DEBUG
        // Resizing the array should not affect the reference count.
        printf(" -- n_ref(m, after resize)   = %d  (should be 1)\n", (int) Py_REFCNT(out));  // returns 1
    #endif  // REF_DEBUG

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
    "sortednp",  // Name of the module
    NULL,  // Module docstring
    -1,  // The module keeps state in global variables.
    SortedNpMethods
};

// Init method
PyMODINIT_FUNC PyInit_sortednp(void) {
    import_array();
    return PyModule_Create(&sortednpmodule);
}
