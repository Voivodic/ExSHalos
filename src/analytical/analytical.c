#define ANALYTICALMODULE_MODULE

/*Import the c functions*/
#include "analytical_h.h"
#include "clpt.h"

/*Import the headers with python and numpy APIs*/
#include <Python.h>
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>

/*This declares the compute function*/
static PyObject *check_precision(PyObject * self, PyObject * args);
static PyObject *correlation_compute(PyObject *self, PyObject *args, PyObject *kwargs);
static PyObject *xilm_compute(PyObject *self, PyObject *args, PyObject *kwargs);
static PyObject *clpt_compute(PyObject *self, PyObject *args, PyObject *kwargs);

/*This tells Python what methods this module has. See the Python-C API for more information.*/
static PyMethodDef analytical_methods[] = {
    {"check_precision", (PyCFunction)check_precision, METH_VARARGS, "Returns precision used by the analytical module"},
    {"correlation_compute", (PyCFunction)correlation_compute, METH_VARARGS | METH_KEYWORDS, "Computes the correlation function or the power spectrum"},
    {"xilm_compute", (PyCFunction)xilm_compute, METH_VARARGS | METH_KEYWORDS, "Computes the generalized correlation functions with fftlog"},
    {"clpt_compute", (PyCFunction)clpt_compute, METH_VARARGS | METH_KEYWORDS, "Computes the power spectra using CLPT"},
    {NULL, NULL, 0, NULL}
};

/*Return the precision used in the grid computations*/
static PyObject *check_precision(PyObject * self, PyObject * args){
    return Py_BuildValue("i", sizeof(fft_real));
}

/*Compute the correlation function for a given spectrum or the oposite*/
static PyObject *correlation_compute(PyObject *self, PyObject *args, PyObject *kwargs){
	int i, direction, Nk, verbose;
    fft_real *k, *P, *R, *Xi;
    double *kd, *Pd, *Rd, *Xid;

	/*Define the list of parameters*/
	static char *kwlist[] = {"k", "Pk", "direction", "verbose", NULL};
	import_array();

	/*Define the pyobject with the 3D position of the tracers*/
	PyArrayObject *k_array, *P_array;  

	/*Read the input arguments*/
	if (!PyArg_ParseTupleAndKeywords(args, kwargs, "OOii", kwlist, &k_array, &P_array, &direction, &verbose))
		return NULL;

    k = (fft_real *) PyArray_DATA(k_array);
    P = (fft_real *) PyArray_DATA(P_array);
    Nk = (int) PyArray_DIMS(k_array)[0];

    if(verbose == TRUE)
        printf("Direction = %d, Nk = %d\n", direction, Nk);

    /*Prepare the PyObject arrays for the outputs*/
	npy_intp dims_k[] = {(npy_intp) Nk};

	/*Alloc the PyObjects for the output*/
	PyArrayObject *np_R = (PyArrayObject *) PyArray_ZEROS(1, dims_k, NP_OUT_TYPE, 0);
    PyArrayObject *np_Xi = (PyArrayObject *) PyArray_ZEROS(1, dims_k, NP_OUT_TYPE, 0);
	R = (fft_real *) PyArray_DATA(np_R);
    Xi = (fft_real *) PyArray_DATA(np_Xi);

    /*Alloc the intermediate arrays used by fftlog*/
    kd = (double *)malloc(Nk*sizeof(double));
    Pd = (double *)malloc(Nk*sizeof(double));
    Rd = (double *)malloc(Nk*sizeof(double));
    Xid = (double *)malloc(Nk*sizeof(double));

    /*Compute the correlation function*/
    if(direction == 1){
        for(i=0;i<Nk;i++){
            kd[i] = (double) k[i];
            Pd[i] = (double) P[i];
        }

        pk2xi(Nk, kd, Pd, Rd, Xid);

        for(i=0;i<Nk;i++){
            R[i] = (fft_real) Rd[i];
            Xi[i] = (fft_real) Xid[i];
        }
    }

    /*Compute the power spectrum*/
    else if(direction == -1){
        for(i=0;i<Nk;i++){
            Rd[i] = (double) k[i];
            Xid[i] = (double) P[i];
        }

        xi2pk(Nk, Rd, Xid, kd, Pd);

        for(i=0;i<Nk;i++){
            R[i] = (fft_real) kd[i];
            Xi[i] = (fft_real) Pd[i];
        }
    }

    else{
        printf("Wrong direction! Direction must be = 1 or = -1 and not %d!\n", direction);
        exit(0);
    }

    /*Free the arrays*/
    free(kd); free(Pd); free(Rd); free(Xid);

    /*Output the mesurements in PyObject format*/
    PyObject *dict = PyDict_New();

    if(direction == 1){
        PyDict_SetItemString(dict, "R", PyArray_Return(np_R));
        PyDict_SetItemString(dict, "Xi", PyArray_Return(np_Xi));
    }
    else{
        PyDict_SetItemString(dict, "k", PyArray_Return(np_R));
        PyDict_SetItemString(dict, "Pk", PyArray_Return(np_Xi));        
    }

    return dict;
}

/*Computes the generalized correlation functions with fftlog*/
static PyObject *xilm_compute(PyObject *self, PyObject *args, PyObject *kwargs){
	int i, l, mk, mr, K, Nk, Nr, verbose;
    double *k, *P, *Plin, *r, *xilm, Rmax;
    double Lambda, alpha;

	/*Define the list of parameters*/
	static char *kwlist[] = {"r", "k", "Pk", "Lambda", "l", "mk", "mr", "K", "alpha", "Rmax", "verbose", NULL};
	import_array();

	/*Define the pyobject with the 3D position of the tracers*/
	PyArrayObject *r_array, *k_array, *P_array;  

	/*Read the input arguments*/
	if (!PyArg_ParseTupleAndKeywords(args, kwargs, "OOOdiiiiddi", kwlist, &r_array, &k_array, &P_array, &Lambda, &l, &mk, &mr, &K, &alpha, &Rmax, &verbose))
		return NULL;

    /*Get the data from the arrays*/
    r = (double *) PyArray_DATA(r_array);
    Nr = (int) PyArray_DIMS(r_array)[0];
    k = (double *) PyArray_DATA(k_array);
    Plin = (double *) PyArray_DATA(P_array);
    Nk = (int) PyArray_DIMS(k_array)[0];

    /*Prepare the PyObject arrays for the outputs*/
	npy_intp dims_r[] = {(npy_intp) Nr};

	/*Alloc the PyObjects for the output*/
    PyArrayObject *np_xilm = (PyArrayObject *) PyArray_ZEROS(1, dims_r, NPY_DOUBLE, 0);
    xilm = (double *) PyArray_DATA(np_xilm);

    /*Smooth the input power spectrum*/
    P = (double *)malloc(Nk*sizeof(double));
    P_smooth(k, Plin, P, Nk, Lambda);

    /*Compute Xi_lm*/
    Xi_lm(k, P, Nk, r, xilm, Nr, l, mk, mr, K, alpha, Rmax);
    free(P);

    /*Output the computations in PyObject format*/
    return PyArray_Return(np_xilm);
}

/*Computes the power spectra using CLPT*/
static PyObject *clpt_compute(PyObject *self, PyObject *args, PyObject *kwargs){
	int i, nmin, nmax, Nk, verbose;
    double *k, *kout, *P, *Plin, *P11;
    double Lambda, kmax;

	/*Define the list of parameters*/
	static char *kwlist[] = {"k", "Pk", "Lambda", "kmax", "nmin", "nmax", "verbose", NULL};
	import_array();

	/*Define the pyobject with the 3D position of the tracers*/
	PyArrayObject *k_array, *P_array;  

	/*Read the input arguments*/
	if (!PyArg_ParseTupleAndKeywords(args, kwargs, "OOddiii", kwlist, &k_array, &P_array, &Lambda, &kmax, &nmin, &nmax, &verbose))
		return NULL;

    /*Get the data from the arrays*/
    k = (double *) PyArray_DATA(k_array);
    Plin = (double *) PyArray_DATA(P_array);
    Nk = (int) PyArray_DIMS(k_array)[0];

    /*Prepare the PyObject arrays for the outputs*/
	npy_intp dims_k[] = {(npy_intp) Nk};

	/*Alloc the PyObjects for the output*/
    PyArrayObject *np_k = (PyArrayObject *) PyArray_ZEROS(1, dims_k, NPY_DOUBLE, 0);
    PyArrayObject *np_P = (PyArrayObject *) PyArray_ZEROS(1, dims_k, NPY_DOUBLE, 0);
    PyArrayObject *np_P11 = (PyArrayObject *) PyArray_ZEROS(1, dims_k, NPY_DOUBLE, 0);

    kout = (double *) PyArray_DATA(np_k);
	P = (double *) PyArray_DATA(np_P);
    P11 = (double *) PyArray_DATA(np_P11); 

    /*Construct the output array of k*/
    for(i=0;i<Nk;i++)
        kout[i] = k[i];

    /*Smooth the input power spectrum*/
    P_smooth(k, Plin, P, Nk, Lambda);

    /*Compute P11*/
    CLPT_P11(k, P, Nk, P11, nmin, nmax, kmax);

    /*Output the computations in PyObject format*/
    PyObject *dict = PyDict_New();

    PyDict_SetItemString(dict, "k", PyArray_Return(np_k));
    PyDict_SetItemString(dict, "Plin", PyArray_Return(np_P));
    PyDict_SetItemString(dict, "P11", PyArray_Return(np_P11));

    return dict;
}


/* This initiates the module using the above definitions. */
#if PY_VERSION_HEX >= 0x03000000
static struct PyModuleDef analytical = {
   PyModuleDef_HEAD_INIT,
   "analytical", NULL, -1, analytical_methods
};

PyMODINIT_FUNC PyInit_analytical(void)
{
    PyObject *m;
    m = PyModule_Create(&analytical);
    if (!m) {
        return NULL;
    }
    return m;
}
#else
PyMODINIT_FUNC initanalytical(void)
{
    PyObject *m = Py_InitModule("analytical", analytical_methods);
    import_array();
    if (!m) {
        return NULL;
    }
}
#endif
