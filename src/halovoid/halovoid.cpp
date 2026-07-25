#include "numpy/ndarrayobject.h"
#include "numpy/ndarraytypes.h"
#include "pytypedefs.h"
#define FINDER_MODULE

// Import the libraries with the functions of this module
#include "finder.hpp"
#include "halovoid_h.hpp"
#include <print>

// Import the headers with python and numpy APIs
#include <Python.h>
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>

// This declares the compute function
static PyObject *halovoid_check_precision(PyObject *self, PyObject *args);
static PyObject *total_cell_volume(PyObject *self, PyObject *args,
                                   PyObject *kwargs);
static PyObject *find(PyObject *self, PyObject *args, PyObject *kwargs);

/*This tells Python what methods this module has. See the Python-C API for more
 * information.*/
static PyMethodDef halovoid_methods[] = {
    {"check_precision", (PyCFunction)halovoid_check_precision, METH_VARARGS,
     "Returns precision used by the estimators of the spectra"},
    {"total_volume", (PyCFunction)total_cell_volume,
     METH_VARARGS | METH_KEYWORDS, "Compute the total volume of all cells."},
    {"find", (PyCFunction)find, METH_VARARGS | METH_KEYWORDS,
     "Find halos and voids in 2D or 3D using voro++."},
    {NULL, NULL, 0, NULL}};

// Function that frees the memory allocated for an array
template <typename T> static void free_array(PyObject *capsule) {
    void *ptr = PyCapsule_GetPointer(capsule, NULL);
    if (ptr != NULL) {
        delete[] static_cast<T *>(ptr);
    }
}

// Create a numpy array from a C array
template <typename T>
static PyObject *create_numpy_array(T *data, npy_intp *dims, int ndims,
                                    auto macro) {
    // Wrap the data in a numpy array
    PyObject *np_array = PyArray_SimpleNewFromData(ndims, dims, macro,
                                                   static_cast<void *>(data));
    if (np_array == NULL) {
        delete[] data;
        return NULL;
    }

    // Create a capsule to hold the data
    PyObject *capsule =
        PyCapsule_New(static_cast<void *>(data), NULL, free_array<T>);
    if (capsule == NULL) {
        Py_DECREF(np_array);
        delete[] data;
        return NULL;
    }

    // Attach the capsule to a base object
    if (PyArray_SetBaseObject(reinterpret_cast<PyArrayObject *>(np_array),
                              capsule) < 0) {
        Py_DECREF(capsule);
        Py_DECREF(np_array);
        delete[] data;
        return NULL;
    }

    return np_array;
}

/*Return the precision used in the grid computations*/
static PyObject *halovoid_check_precision(PyObject *self, PyObject *args) {
    return Py_BuildValue("i", sizeof(fft_real));
}

// Function that computes the total volume of all cells
static PyObject *total_cell_volume(PyObject *self, PyObject *args,
                                   PyObject *kwargs) {
    std::size_t np, ndim;
    int Nd;
    fft_real *pos;
    fft_real L;

    // Define the list of parameters
    static const char *kwlist[] = {"particles", "Nd", "L", NULL};

    // Define the pyobjests
    PyArrayObject *pos_array;

// Read the input arguments
#ifdef DOUBLEPRECISION_FFTW
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "Oid", kwlist, &pos_array,
                                     &Nd, &L))
        return NULL;
#else
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "Oif", kwlist, &pos_array,
                                     &Nd, &L))
        return NULL;
#endif

    /*Convert the PyObjects to C arrays*/
    pos = static_cast<fft_real *>(PyArray_DATA(pos_array));
    np = (std::size_t)PyArray_DIMS(pos_array)[0];
    ndim = (std::size_t)PyArray_DIMS(pos_array)[1];

    // Check the dimensionality of the input array
    if (ndim != 3) { // Only 3D arrays are supported
        PyErr_SetString(PyExc_ValueError, "Only 3D arrays are supported");
        return NULL;
    }

    // Check the number of particles
    if (np < 1) {
        PyErr_SetString(PyExc_ValueError, "At least one particle is required");
        return NULL;
    }

    // Check the number of dimensions
    if (Nd < 1) {
        PyErr_SetString(PyExc_ValueError, "The number of dimensions must be at "
                                          "least 1");
        return NULL;
    }

    // Check the length of the box
    if (L < 1) {
        PyErr_SetString(PyExc_ValueError, "The length of the box must be at "
                                          "least 1");
        return NULL;
    }

    // Create the container
    Container con = Container(pos, np, Nd, L);

    // Compute the total volume
    double volume = total_volume(con);

    return Py_BuildValue("d", volume);
}

// Function that finds halos and voids in 2D or 3D using voro++
static PyObject *find(PyObject *self, PyObject *args, PyObject *kwargs) {
    int nd;
    fft_real delta_h, delta_v, L, r_max;
    PyArrayObject *pos_array;

    // Define the list of parameters
    static const char *kwlist[] = {"particles", "delta_h", "delta_v", "L",
                                   "nd",        "r_max",   NULL};

// Read the input arguments
#ifdef DOUBLEPRECISION_FFTW
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "Odddid", kwlist, &pos_array,
                                     &delta_h, &delta_v, &L, &nd, &r_max))
        return NULL;
#else
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "Offfif", kwlist, &pos_array,
                                     &delta_h, &delta_v, &L, &nd, &r_max))
        return NULL;
#endif

    /*Convert the PyObjects to C arrays*/
    fft_real *pos = static_cast<fft_real *>(PyArray_DATA(pos_array));
    const std::size_t np = (std::size_t)PyArray_DIMS(pos_array)[0];
    const std::size_t ndim = (std::size_t)PyArray_DIMS(pos_array)[1];

    // Check the dimensionality of the input array
    if (ndim != 3) { // Only 3D arrays are supported
        PyErr_SetString(PyExc_ValueError, "Only 3D arrays are supported");
        return NULL;
    }

    // Check the number of particles
    if (np < 1) {
        PyErr_SetString(PyExc_ValueError, "At least one particle is required");
        return NULL;
    }

    // Check the number of dimensions
    if (nd < 1) {
        PyErr_SetString(PyExc_ValueError, "The number of dimensions must be at "
                                          "least 1");
        return NULL;
    }

    // Check the length of the box
    if (L <= 0.0) {
        PyErr_SetString(PyExc_ValueError,
                        "The length of the box must be larger than 0!\n");
        return NULL;
    }

    // Create the container
    Container con = Container(pos, np, nd, L);

    // Compute the densities
    const fft_real rho_m = static_cast<fft_real>(np) / (L * L * L);
    const fft_real rho_halos = delta_h * rho_m;
    const fft_real rho_voids = delta_v * rho_m;
    const double dist_tol =
        1e-6 * static_cast<double>(L) / std::pow(static_cast<double>(np), 1.0 / 3.0);

    // Compute the voronoit tessellation of the particles
    HaloVoid halos = HaloVoid(np / 100);
    HaloVoid voids = HaloVoid(np / 100);
    compute_voronoi_3d(con, halos, rho_halos, voids, rho_voids, false, NULL, dist_tol);
    const std::size_t n_halos = halos.n;
    const std::size_t n_voids = voids.n;

    // Create the PyObjects for the output
    PyObject *dict = PyDict_New();
    if (dict == NULL)
        return NULL;

    // Set the halos for the output
    if (rho_halos > 0.0) {
        // Create the numpy arrays for the output
        npy_intp pos_dims[2] = {static_cast<npy_intp>(n_halos),
                                static_cast<npy_intp>(3)};
        PyObject *halos_pos =
            create_numpy_array<fft_real>(halos.pos, pos_dims, 2, NP_OUT_TYPE);
        npy_intp rho_dims[1] = {static_cast<npy_intp>(n_halos)};
        PyObject *halos_rho =
            create_numpy_array<fft_real>(halos.den, rho_dims, 1, NP_OUT_TYPE);
        // Ownership of pos/den transferred to the numpy capsules; prevent the
        // HaloVoid destructor from freeing the same memory (double free).
        halos.pos = nullptr;
        halos.den = nullptr;

        // create_numpy_array returns NULL on failure (and frees the buffer).
        if (halos_pos == NULL || halos_rho == NULL) {
            Py_XDECREF(halos_pos);
            Py_XDECREF(halos_rho);
            Py_DECREF(dict);
            return NULL;
        }

        // Put the arrays in the output dict
        if (PyDict_SetItemString(dict, "halos_pos", halos_pos) < 0) {
            Py_DECREF(halos_pos);
            Py_DECREF(halos_rho);
            Py_DECREF(dict);
            return NULL;
        }
        if (PyDict_SetItemString(dict, "halos_den", halos_rho) < 0) {
            Py_DECREF(halos_pos);
            Py_DECREF(halos_rho);
            Py_DECREF(dict);
            return NULL;
        }

        // Decrement the reference counts
        Py_DECREF(halos_pos);
        Py_DECREF(halos_rho);
    }

    // Set the voids for the output
    if (rho_voids > 0.0) {
        // Create the numpy arrays for the output
        npy_intp pos_dims[2] = {static_cast<npy_intp>(n_voids),
                                static_cast<npy_intp>(3)};
        PyObject *voids_pos =
            create_numpy_array<fft_real>(voids.pos, pos_dims, 2, NP_OUT_TYPE);
        npy_intp rho_dims[1] = {static_cast<npy_intp>(n_voids)};
        PyObject *voids_rho =
            create_numpy_array<fft_real>(voids.den, rho_dims, 1, NP_OUT_TYPE);
        // Ownership transferred to the numpy capsules; prevent double free.
        voids.pos = nullptr;
        voids.den = nullptr;

        // create_numpy_array returns NULL on failure (and frees the buffer).
        if (voids_pos == NULL || voids_rho == NULL) {
            Py_XDECREF(voids_pos);
            Py_XDECREF(voids_rho);
            Py_DECREF(dict);
            return NULL;
        }

        // Put the arrays in the output dict
        if (PyDict_SetItemString(dict, "voids_pos", voids_pos) < 0) {
            Py_DECREF(voids_pos);
            Py_DECREF(voids_rho);
            Py_DECREF(dict);
            return NULL;
        }
        if (PyDict_SetItemString(dict, "voids_den", voids_rho) < 0) {
            Py_DECREF(voids_pos);
            Py_DECREF(voids_rho);
            Py_DECREF(dict);
            return NULL;
        }

        // Decrement the reference counts
        Py_DECREF(voids_pos);
        Py_DECREF(voids_rho);
    }

    return dict;
}

#ifdef __cplusplus
extern "C" {
#endif

// Define the Python module
static struct PyModuleDef halovoid_module = {PyModuleDef_HEAD_INIT, "halovoid",
                                             NULL, -1, halovoid_methods};
PyMODINIT_FUNC PyInit_halovoid(void) {
    import_array();
    return PyModule_Create(&halovoid_module);
}

#ifdef __cplusplus
}
#endif
