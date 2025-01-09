#define FINDER_MODULE

/*Import the headers with all C functions*/
#include "finder_h.hpp"
#include "voronoi.hpp"

/*Import the python and numpy APIs*/
#include <Python.h>
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>

/*This declares the compute function*/
static PyObject *check_precision(PyObject *self, PyObject *args);
static PyObject *voronoi_compute(PyObject *self, PyObject *args,
                                 PyObject *kwargs);
/*This tells Python what methods this module has. See the Python-C API for more
 * information.*/
static PyMethodDef finder_methods[] = {
    {"check_precision", (PyCFunction)check_precision, METH_VARARGS,
     "Returns precision used by the finder functions"},
    {"voronoi_compute", (PyCFunction)voronoi_compute, METH_VARARGS | METH_KEYWORDS,
     "Computes the voronoi tesselation"},
    {NULL, NULL, 0, NULL}};

/*Return the precision used in the grid computations*/
static PyObject *check_precision(PyObject *self, PyObject *args) {
    return Py_BuildValue("i", sizeof(fft_real));
}

/*Compute the voronoi tesselation for a collection of points*/
static PyObject *voronoi_compute(PyObject *self, PyObject *args,
                                 PyObject *kwargs) {
    size_t np;
    int ndx, ndy, ndz, verbose, nthreads;
    fft_real Lc, Om0;
    fft_real *pos, *vol;

    /*Define the list of parameters*/
    static char *kwlist[] = {"pos",     "nd",       "L", "Om0",
                             "verbose", "nthreads", NULL};
    import_array();

    /*Define the pyobject with the 3D position of the tracers*/
    PyArrayObject *pos_array;

/*Read the input arguments*/
#ifdef DOUBLEPRECISION_FFTW
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "Oiiiddii", kwlist,
                                     &pos_array, &ndx, &ndy, &ndz, &Lc, &Om0,
                                     &verbose, &nthreads))
        return NULL;
#else
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "Oiiiffii", kwlist,
                                     &pos_array, &ndx, &ndy, &ndz, &Lc, &Om0,
                                     &verbose, &nthreads))
        return NULL;
#endif

    /*Set the box structure*/
    set_cosmology(Om0, 0.0,
                  1.686);  // Not used in this function but needed by set box
    set_box(ndx, ndy, ndz, Lc);

    /*Convert the PyObjects to C arrays*/
    pos = (fft_real *) PyArray_DATA(pos_array);
    np = (size_t) PyArray_DIMS(pos_array)[0];

    /*Set the number of threads to be used*/
    omp_set_num_threads(nthreads);

    /*Define the dimensions of the output*/
    npy_intp dims_vol[] = {(npy_intp)np, (npy_intp)3};

    /*Alloc the numpy arrays for the output*/
    PyArrayObject *np_vol =
        (PyArrayObject *)PyArray_ZEROS(2, dims_vol, NP_OUT_TYPE, 0);
    vol = (fft_real *) PyArray_DATA(np_vol);

    /*Compute the voronoi tesselation*/
    Compute_Voronoi(pos, np, vol, TRUE);

    /*Put the array in the output dict*/
    PyObject *dict = PyDict_New();

    PyDict_SetItemString(dict, "vol", PyArray_Return(np_vol));

    return dict;
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
