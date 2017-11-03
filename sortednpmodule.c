
#include <Python.h>
#include <numpy/arrayobject.h>

static PyObject * sortednp_merge(PyObject *self, PyObject *args) {
    // const char *command;
    // int sts;

    // if (!PyArg_ParseTuple(args, "s", &command))
    //     return NULL;
    // sts = system(command);
//    printf("CP 0\n");
    PyObject *a, *b;
    if (!PyArg_ParseTuple(args, "OO", &a, &b))
         return NULL;
//    printf("CP 1\n");

    a = PyArray_FROM_OF(a, NPY_ARRAY_CARRAY_RO);
    b = PyArray_FROM_OF(b, NPY_ARRAY_CARRAY_RO);

    if (a == NULL || b == NULL) {
        return NULL;
    }

//    printf("CP 1.5\n");

    int nd_a = PyArray_NDIM(a);
    int nd_b = PyArray_NDIM(b);

//    printf("nd a = %d\n", nd_a);
//    printf("nd b = %d\n", nd_b);

    if (PyArray_NDIM(a) != 1 || PyArray_NDIM(b) != 1) {
      return NULL;
    }
//    printf("CP 2\n");

    npy_intp len_a = PyArray_DIMS(a)[0];
    npy_intp len_b = PyArray_DIMS(b)[0];

//    printf("len a = %d\n", (int) len_a);
//    printf("len b = %d\n", (int) len_b);

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
    // for (npy_intp i_a=0; i_a < len_a; i_a++) {
    //     for (npy_intp i_a=0; i_a < len_a; i_a++) {
    //         double *v = (double*) PyArray_GETPTR1(input, j);
//    //         printf("Hell %d %f\n", j, *v);
    //         // double *v = (double*) PyArray_GETPTR1(input, j)
    //         *v = (*v) * .343;
    //     }
    // }

    return out;
}

static PyMethodDef SortedNpMethods[] = {
    {"merge",  sortednp_merge, METH_VARARGS,
     "Merge two sorted numpy arrays."},
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
