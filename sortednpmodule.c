
#include <stdbool.h>
#include <Python.h>
#include <numpy/arrayobject.h>

static PyObject * sortednp_intersect(PyObject *self, PyObject *args) {
    PyObject *a, *b;

    if (!PyArg_ParseTuple(args, "O!O!", &PyArray_Type, &a, &PyArray_Type, &b))
        return NULL;

    a = PyArray_FROM_OF(a, NPY_ARRAY_CARRAY_RO);
    b = PyArray_FROM_OF(b, NPY_ARRAY_CARRAY_RO);

    if (a == NULL || b == NULL) {
        return NULL;
    }

    int nd_a = PyArray_NDIM(a);
    int nd_b = PyArray_NDIM(b);

    if (PyArray_NDIM(a) != 1 || PyArray_NDIM(b) != 1) {
      return NULL;
    }

    npy_intp len_a = PyArray_DIMS(a)[0];
    npy_intp len_b = PyArray_DIMS(b)[0];

    npy_intp new_dim[1] = {len_a < len_b ? len_a : len_b};

    PyArrayObject *out =  
      PyArray_SimpleNewFromDescr(1, new_dim, PyArray_DESCR(a));
    
    npy_intp i_a = 0;
    npy_intp i_b = 0;
    npy_intp i_o = 0;
    double v_a = *((double*) PyArray_GETPTR1(a, i_a));
    double v_b = *((double*) PyArray_GETPTR1(b, i_b));

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

    // resize
    new_dim[0] = i_o;
    PyArray_Dims dims;
    dims.ptr = new_dim;
    dims.len = 1;

    PyArray_Resize(out, &dims, 0, NPY_CORDER);

    return out;
}

static PyObject * sortednp_merge(PyObject *self, PyObject *args) {
    PyObject *a, *b;
    if (!PyArg_ParseTuple(args, "O!O!", &PyArray_Type, &a, &PyArray_Type, &b))
        return NULL;

    a = PyArray_FROM_OF(a, NPY_ARRAY_CARRAY_RO);
    b = PyArray_FROM_OF(b, NPY_ARRAY_CARRAY_RO);

    if (a == NULL || b == NULL) {
        return NULL;
    }

    int nd_a = PyArray_NDIM(a);
    int nd_b = PyArray_NDIM(b);

    if (PyArray_NDIM(a) != 1 || PyArray_NDIM(b) != 1) {
      return NULL;
    }

    npy_intp len_a = PyArray_DIMS(a)[0];
    npy_intp len_b = PyArray_DIMS(b)[0];

    npy_intp new_dim[1] = {len_a + len_b};

    PyArrayObject *out =  
      PyArray_SimpleNewFromDescr(1, new_dim, PyArray_DESCR(a));
    
    npy_intp i_a = 0;
    npy_intp i_b = 0;
    npy_intp i_o = 0;
    double v_a = *((double*) PyArray_GETPTR1(a, i_a));
    double v_b = *((double*) PyArray_GETPTR1(b, i_b));
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

    // copy remaining items
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

    return out;
}

static PyMethodDef SortedNpMethods[] = {
    {"merge",  sortednp_merge, METH_VARARGS, "Merge two sorted numpy arrays."},
    {"intersect",  sortednp_intersect, METH_VARARGS, "Intersect two sorted numpy arrays."},
    {NULL, NULL, 0, NULL}        /* Sentinel */
};

static struct PyModuleDef sortednpmodule = {
    PyModuleDef_HEAD_INIT,
    "sortednp",   /* name of module */
    NULL, /* module documentation, may be NULL */
    -1,       /* size of per-interpreter state of the module,
                 or -1 if the module keeps state in global variables. */
    SortedNpMethods
};

PyMODINIT_FUNC PyInit_sortednp(void) {
    import_array();
    return PyModule_Create(&sortednpmodule);
}
