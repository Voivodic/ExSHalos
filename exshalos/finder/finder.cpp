#include <stdio.h>
#define FINDER_MODULE

#include "finder_h.h"

/*This declares the compute function*/
static PyObject *finder_check_precision(PyObject *self, PyObject *args);
// static PyObject *grid_compute(PyObject * self, PyObject * args, PyObject
// *kwargs);

/*This tells Python what methods this module has. See the Python-C API for more
 * information.*/
static PyMethodDef finder_methods[] = {
    {"check_precision", finder_check_precision, METH_VARARGS,
     "Returns precision used by the finder functions"},
    //{"grid_compute", grid_compute, METH_VARARGS | METH_KEYWORDS, "Computes the
    //density grid of a given list of particles"},
    {NULL, NULL, 0, NULL}};

/*Return the precision used in the grid computations*/
static PyObject *finder_check_precision(PyObject *self, PyObject *args) {
    return Py_BuildValue("i", sizeof(fft_real));
}

/* This initiates the module using the above definitions. */
#if PY_VERSION_HEX >= 0x03000000
static struct PyModuleDef finder = {PyModuleDef_HEAD_INIT, "finder", NULL, -1,
                                    finder_methods};

PyMODINIT_FUNC PyInit_finder(void) {
    PyObject *m;
    m = PyModule_Create(&finder);
    if (!m) {
        return NULL;
    }
    return m;
}
#else
PyMODINIT_FUNC initfinder(void) {
    PyObject *m = Py_InitModule("finder", finder_methods);
    import_array();
    if (!m) {
        return NULL;
    }
}
#endif
