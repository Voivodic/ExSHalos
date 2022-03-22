#define EXSHALOS_MODULE

#include "exshalos_h.h"
#include "density_grid.h"

/*This declares the compute function*/
static PyObject *exshalos_check_precision(PyObject * self, PyObject * args);
static PyObject *correlation_compute(PyObject *self, PyObject *args, PyObject *kwargs);
static PyObject *density_grid_compute(PyObject *self, PyObject *args, PyObject *kwargs);

/*This tells Python what methods this module has. See the Python-C API for more information.*/
static PyMethodDef exshalos_methods[] = {
    {"check_precision", exshalos_check_precision, METH_VARARGS, "Returns precision used by the estimators of the spectra"},
    {"correlation_compute", correlation_compute, METH_VARARGS | METH_KEYWORDS, "Computes the correlation function or the power spectrum"},
    {"density_grid_compute", density_grid_compute, METH_VARARGS | METH_KEYWORDS, "Generate the gaussian density grid"},
    {NULL, NULL, 0, NULL}
};

/*Return the precision used in the grid computations*/
static PyObject *exshalos_check_precision(PyObject * self, PyObject * args){
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

    k = (fft_real *) k_array->data;
    P = (fft_real *) P_array->data;
    Nk = (int) k_array->dimensions[0];

    if(verbose == TRUE)
        printf("Direction = %d, Nk = %d\n", direction, Nk);

    /*Prepare the PyObject arrays for the outputs*/
	npy_intp dims_k[] = {(npy_intp) Nk};

	/*Alloc the PyObjects for the output*/
	PyArrayObject *np_R = (PyArrayObject *) PyArray_ZEROS(1, dims_k, NP_OUT_TYPE, 0);
    PyArrayObject *np_Xi = (PyArrayObject *) PyArray_ZEROS(1, dims_k, NP_OUT_TYPE, 0);
	R = (fft_real *) np_R->data;
    Xi = (fft_real *) np_Xi->data;

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
    PyObject *tupleresult = PyTuple_New(2);
    PyTuple_SetItem(tupleresult, 0, PyArray_Return(np_R));
    PyTuple_SetItem(tupleresult, 1, PyArray_Return(np_Xi));

    return PyArray_Return((PyArrayObject*) tupleresult);
}

/*Compute the Gaussian density grid*/
static PyObject *density_grid_compute(PyObject *self, PyObject *args, PyObject *kwargs){
    int ndx, ndy, ndz, outk, Nk, verbose, nthreads, seed;
    fft_real Lc, R_max, *K, *P, *delta;
    fft_complex *deltak;

	/*Define the list of parameters*/
	static char *kwlist[] = {"k", "P", "R_max", "Ndx", "Ndy", "Ndz", "Lc/Mc", "outk", "seed", "verbose", "nthreads", NULL};
	import_array();

	/*Define the pyobject with the 3D position of the tracers*/
	PyArrayObject *K_array, *P_array;  

	/*Read the input arguments*/
	#ifdef DOUBLEPRECISION_FFTW
		if (!PyArg_ParseTupleAndKeywords(args, kwargs, "OOdiiidiiii", kwlist, &K_array, &P_array, &R_max, &ndx, &ndy, &ndz, &Lc, &outk, &seed, &verbose, &nthreads))
			return NULL;
	#else
		if (!PyArg_ParseTupleAndKeywords(args, kwargs, "OOfiiifiiii", kwlist, &K_array, &P_array, &R_max, &ndx, &ndy, &ndz, &Lc, &outk, &seed, &verbose, &nthreads))
			return NULL;
	#endif

    if(verbose == TRUE)
        printf("Nd = (%d, %d, %d), Lc = %f, Nthreads = %d, seed = %d\n", ndx, ndy, ndz, Lc, nthreads, seed);

    /*Set the box structure*/
    set_cosmology(0.31, 0.0, 1.0, 1.686);   //Not used in this function but needed by set box
    set_box(ndx, ndy, ndz, Lc);

	/*Convert the PyObjects to C arrays*/
    Nk = (int) K_array->dimensions[0];
    K = (fft_real *) K_array->data;
    P = (fft_real *) P_array->data;

    /*Initialize FFTW and openmp to run in parallel*/
    omp_set_num_threads(nthreads);
    FFTW(init_threads)();
    FFTW(plan_with_nthreads)(nthreads);

    /*Define the variables for the output*/
    npy_intp dims_grid[] = {(npy_intp) ndx, (npy_intp) ndy, (npy_intp) ndz};
    npy_intp dims_gridk[] = {(npy_intp) ndx, (npy_intp) ndy, (npy_intp) ndz/2 + 1, (npy_intp) 2};
    PyArrayObject *np_grid, *np_gridk;
    PyObject *tupleresult;

    /*Output the grid only in real space*/
    if(outk == FALSE){
        /*Alloc the PyObjects for the output*/
        np_grid = (PyArrayObject *) PyArray_ZEROS(3, dims_grid, NP_OUT_TYPE, 0);
        delta = (fft_real *) np_grid->data;
        deltak = (fft_complex *) malloc(((size_t) ndx)*((size_t) ndy)*((size_t) ndz/2+1)*sizeof(fft_complex));

        /*Compute the density grids*/
        Compute_Den(K, P, Nk, R_max, delta, deltak, seed);

        /*Output the mesurements in PyObject format*/
        tupleresult = PyTuple_New(1);
        PyTuple_SetItem(tupleresult, 0, PyArray_Return(np_grid));  
        FFTW(free)(deltak);     
    }

    /*Output the grid in real and Fourier space*/
    else{
        /*Alloc the PyObjects for the output*/
        np_grid = (PyArrayObject *) PyArray_ZEROS(3, dims_grid, NP_OUT_TYPE, 0);
        delta = (fft_real *) np_grid->data;
        np_gridk = (PyArrayObject *) PyArray_ZEROS(4, dims_gridk, NP_OUT_TYPE, 0);
        deltak = (fft_complex *) np_gridk->data;

        /*Compute the density grids*/
        Compute_Den(K, P, Nk, R_max, delta, deltak, seed);

        /*Output the mesurements in PyObject format*/
        tupleresult = PyTuple_New(2);
        PyTuple_SetItem(tupleresult, 0, PyArray_Return(np_grid));
        PyTuple_SetItem(tupleresult, 1, PyArray_Return(np_gridk));
    }

    return PyArray_Return((PyArrayObject*) tupleresult);       
}

/* This initiates the module using the above definitions. */
#if PY_VERSION_HEX >= 0x03000000
static struct PyModuleDef exshalos = {
   PyModuleDef_HEAD_INIT,
   "power", NULL, -1, exshalos_methods
};

PyMODINIT_FUNC PyInit_exshalos(void)
{
    PyObject *m;
    m = PyModule_Create(&exshalos);
    if (!m) {
        return NULL;
    }
    return m;
}
#else
PyMODINIT_FUNC initexshalos(void)
{
    PyObject *m = Py_InitModule("exshalos", exshalos_methods);
    import_array();
    if (!m) {
        return NULL;
    }
}
#endif
