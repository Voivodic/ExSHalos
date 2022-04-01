#define SPECTRUM_MODULE

#include "spectrum_h.h"
#include "gridmodule.h"
#include "powermodule.h"
#include "bimodule.h"
#include "trimodule.h"

/*This declares the compute function*/
static PyObject *spectrum_check_precision(PyObject * self, PyObject * args);
static PyObject *grid_compute(PyObject * self, PyObject * args, PyObject *kwargs);
static PyObject *power_compute(PyObject * self, PyObject * args, PyObject *kwargs);
static PyObject *bi_compute(PyObject * self, PyObject * args, PyObject *kwargs);
static PyObject *tri_compute(PyObject * self, PyObject * args, PyObject *kwargs);

/*This tells Python what methods this module has. See the Python-C API for more information.*/
static PyMethodDef spectrum_methods[] = {
    {"check_precision", spectrum_check_precision, METH_VARARGS, "Returns precision used by the estimators of the spectra"},
    {"grid_compute", grid_compute, METH_VARARGS | METH_KEYWORDS, "Computes the density grid of a given list of particles"},
    {"power_compute", power_compute, METH_VARARGS | METH_KEYWORDS, "Computes the PowerSpectrum of a given density grid"},
    {"bi_compute", bi_compute, METH_VARARGS | METH_KEYWORDS, "Computes the BiSpectrum of a given density grid"},
    {"tri_compute", tri_compute, METH_VARARGS | METH_KEYWORDS, "Computes the TriSpectrum of a given density grid"},
    {NULL, NULL, 0, NULL}
};

/*Return the precision used in the grid computations*/
static PyObject *spectrum_check_precision(PyObject * self, PyObject * args){
    return Py_BuildValue("i", sizeof(fft_real));
}

/*Function that computes the density grid for each tracer and outputs it in numpy format*/
static PyObject *grid_compute(PyObject *self, PyObject *args, PyObject *kwargs){
	size_t np;
	int nmass, ntype, nd, window, interlacing, *type, verbose, nthreads, direction;
	fft_real *pos, *vel, *mass, *grid;
	fft_real L, R, R_times;

	/*Define the list of parameters*/
	static char *kwlist[] = {"pos", "vel", "mass", "nmass", "type", "ntype", "nd", "L", "direction", "window", "R", "R_times", "interlacing", "verbose", "nthreads", NULL};
	import_array();

	/*Define the pyobject with the 3D position of the tracers*/
	PyArrayObject *pos_array, *vel_array, *mass_array, *type_array;  

	/*Read the input arguments*/
	#ifdef DOUBLEPRECISION_FFTW
		if (!PyArg_ParseTupleAndKeywords(args, kwargs, "OOOiOiifiiffiii", kwlist, &pos_array,  &vel_array, &np, &mass_array, &nmass, &type_array, &ntype, &nd, &L, &direction, &window, &R, &R_times, &interlacing, &verbose, &nthreads))
			return NULL;
	#else
		if (!PyArg_ParseTupleAndKeywords(args, kwargs, "OOOiOiifiiffiii", kwlist, &pos_array, &vel_array, &mass_array, &nmass, &type_array, &ntype, &nd, &L, &direction, &window, &R, &R_times, &interlacing, &verbose, &nthreads))
			return NULL;
	#endif

	/*Convert the PyObjects to C arrays*/
	pos = (fft_real *) pos_array->data;
	np = (size_t) pos_array->dimensions[0];
    if(direction != -1)
        vel = (fft_real *) vel_array->data;
    else
        vel = NULL;
	if(nmass == TRUE)
		mass = (fft_real *) mass_array->data;
	else
		mass = NULL;
	if(ntype > 1)
		type = (int *) type_array->data;
	else
		type = NULL;

	omp_set_num_threads(nthreads);

	if(verbose == TRUE){
		printf("Computing the density grid\n");
		printf("Np = %d, L = %f, direction = %d, Ntype = %d, Ncells = %d, interlacing = %d, window = %d", (int) np, (float) L, direction, ntype, nd, interlacing, window);
		if(window > 2)
			printf(", R = %f", (float) R);
		if(window == 4)
			printf(", R_times = %f", (float) R_times);
		printf(", nthreads = %d\n", nthreads);
	}

	/*Prepare the PyObject arrays for the outputs*/
	npy_intp dims_grid[] = {(npy_intp) interlacing+1, (npy_intp) ntype, (npy_intp) nd, (npy_intp) nd, (npy_intp) nd};

	/*Alloc the PyObjects for the output*/
	PyArrayObject *np_grid = (PyArrayObject *) PyArray_ZEROS(5, dims_grid, NP_OUT_TYPE, 0);
	grid = (fft_real *) np_grid->data;

	/*Compute the grids for each tracer*/
	Tracer_Grid(grid, nd, L, direction, pos, vel, np, mass, type, ntype, window, R, R_times, interlacing);

	/*Returns the density grids*/
	return PyArray_Return(np_grid);
}

/*Function that computes the cross power spectrum for all tracers and outputs it in numpy format*/
static PyObject *power_compute(PyObject *self, PyObject *args, PyObject *kwargs){
    long *count_k;
	int i, j, k, ntype, nd, window, interlacing, Nk, verbose, nthreads, NPs, l_max;
    long double *P, *Kmean;
	fft_real *grid, k_min, k_max, *P_out, *Kmean_out;
	fft_real L, R;

	/*Define the list of parameters*/
	static char *kwlist[] = {"grid", "ntype", "nd", "L", "window", "R", "interlacing", "Nk", "k_min", "k_max", "l_max", "verbose", "nthreads", NULL};
	import_array();

	/*Define the pyobject with the 3D position of the tracers*/
	PyArrayObject *grid_array;  

	/*Read the input arguments*/
	#ifdef DOUBLEPRECISION_FFTW
		if (!PyArg_ParseTupleAndKeywords(args, kwargs, "Oiididiiddiii", kwlist, &grid_array &ntype, &nd, &L, &window, &R, &interlacing, &Nk, &k_min, &k_max, &l_max, &verbose, &nthreads))
			return NULL;
	#else
		if (!PyArg_ParseTupleAndKeywords(args, kwargs, "Oiififiiffiii", kwlist, &grid_array, &ntype, &nd, &L, &window, &R, &interlacing, &Nk, &k_min, &k_max, &l_max, &verbose, &nthreads))
			return NULL;
	#endif

	/*Convert the PyObjects to C arrays*/
	grid = (fft_real *) grid_array->data;
    NPs = (ntype*(ntype+1))/2;

    /*Initialize FFTW and openmp to run in parallel*/
    omp_set_num_threads(nthreads);
    FFTW(init_threads)();
    FFTW(plan_with_nthreads)(nthreads);

    if(verbose == TRUE){
		printf("Computing the power spectra\n");
		printf("kmin = %f, kmax = %f, Nk = %d, l_max = %d, L = %f, Ntype = %d, Ncells = %d, interlacing = %d, window = %d", (float) k_min, (float) k_max, Nk, l_max, (float) L, ntype, nd, interlacing, window);
		if(window > 2)
			printf(", R = %f", (float) R);
		printf(", nthreads = %d\n", nthreads);
	}

    /*Prepare the PyObject arrays for the outputs*/
	npy_intp dims_P[] = {(npy_intp) NPs, (npy_intp) l_max + 1, (npy_intp) Nk};
    npy_intp dims_k[] = {(npy_intp) Nk};

	/*Alloc the PyObjects for the output*/
    PyArrayObject *np_P = (PyArrayObject *) PyArray_ZEROS(3, dims_P, NP_OUT_TYPE, 0);
    PyArrayObject *np_k = (PyArrayObject *) PyArray_ZEROS(1, dims_k, NP_OUT_TYPE, 0);
    PyArrayObject *np_count_k = (PyArrayObject *) PyArray_ZEROS(1, dims_k, PyArray_LONG, 0);   
    P_out = (fft_real *) np_P->data;
    Kmean_out = (fft_real *) np_k->data;
    count_k = (long *) np_count_k->data;

    /*Allocs the arrays for P and k*/
    P = (long double *) malloc(NPs*Nk*(l_max+1)*sizeof(long double));
    Kmean = (long double *) malloc(Nk*sizeof(long double));
    for(i=0;i<Nk;i++){
        Kmean[i] = 0.0;
        for(j=0;j<NPs;j++)
            for(k=0;k<=l_max;k++)
                P[(j*(l_max+1) + k)*Nk + i] = 0.0;
    }   

	/*Compute the spectra for all tracers*/
	Power_Spectrum(grid, nd, L, ntype, window, R, interlacing, Nk, k_min, k_max, Kmean, P, count_k, l_max);

    /*Put the values in the outputs*/
    for(i=0;i<Nk;i++){
        Kmean_out[i] = (fft_real) Kmean[i];
        for(j=0;j<NPs;j++)
            for(k=0;k<=l_max;k++)
                P_out[(j*(l_max+1) + k)*Nk + i] = (fft_real) P[(j*(l_max+1) + k)*Nk + i];
    }

    /*Free the arrays*/
    free(Kmean);
    free(P);

    /*Output the mesurements in PyObject format*/
    PyObject *tupleresult = PyTuple_New(3);
    PyTuple_SetItem(tupleresult, 0, PyArray_Return(np_k));
    PyTuple_SetItem(tupleresult, 1, PyArray_Return(np_P));
    PyTuple_SetItem(tupleresult, 2, PyArray_Return(np_count_k));

    return PyArray_Return((PyArrayObject*) tupleresult);
}

/*Function that computes the cross bispectrum for all tracers and outputs it in numpy format*/
static PyObject *bi_compute(PyObject *self, PyObject *args, PyObject *kwargs){
	int i, j, ntype, nd, window, interlacing, Nk, verbose, nthreads, NPs, NBs, Ntri;
    long double **P, *KP, **B, *K1, *K2, *K3, *I, *IP;
	fft_real *grid, k_min, k_max, *P_out, *KP_out, *B_out, *KB_out, *count_k, *count_tri;
	fft_real L, R;

	/*Define the list of parameters*/
	static char *kwlist[] = {"grid", "ntype", "nd", "L", "window", "R", "interlacing", "Nk", "k_min", "k_max", "verbose", "nthreads", NULL};
	import_array();

	/*Define the pyobject with the 3D position of the tracers*/
	PyArrayObject *grid_array;  

	/*Read the input arguments*/
	#ifdef DOUBLEPRECISION_FFTW
		if (!PyArg_ParseTupleAndKeywords(args, kwargs, "Oiididiiddii", kwlist, &grid_array &ntype, &nd, &L, &window, &R, &interlacing, &Nk, &k_min, &k_max, &verbose, &nthreads))
			return NULL;
	#else
		if (!PyArg_ParseTupleAndKeywords(args, kwargs, "Oiififiiffii", kwlist, &grid_array, &ntype, &nd, &L, &window, &R, &interlacing, &Nk, &k_min, &k_max, &verbose, &nthreads))
			return NULL;
	#endif

	/*Convert the PyObjects to C arrays*/
	grid = (fft_real *) grid_array->data;
    NPs = ntype*(ntype+1)/2;
    NBs = pow(ntype, 3);

    /*Initialize FFTW and openmp to run in parallel*/
    omp_set_num_threads(nthreads);
    FFTW(init_threads)();
    FFTW(plan_with_nthreads)(nthreads);

    if(verbose == TRUE){
		printf("Computing the bispectra\n");
		printf("k_min = %f, kmax = %f, Nk = %d, L = %f, Ntype = %d, Ncells = %d, interlacing = %d, window = %d", (float) k_min, (float) k_max, Nk, (float) L, ntype, nd, interlacing, window);
		if(window > 2)
			printf(", R = %f", (float) R);
		printf(", nthreads = %d\n", nthreads);
	}

    /*Allocs the arrays for for the spectra*/
    Ntri = 1000 + floor(1000*pow((float) Nk/20.0, 3.0));
    P = (long double **) malloc(NPs*sizeof(long double *));
    for(i=0;i<NPs;i++)
        P[i] = (long double *) malloc(Nk*sizeof(long double));
    B = (long double **) malloc(NBs*sizeof(long double *));
    for(i=0;i<NBs;i++)
        B[i] = (long double *) malloc(Ntri*sizeof(long double));    
    KP = (long double *) malloc(Nk*sizeof(long double));
    K1 = (long double *) malloc(Ntri*sizeof(long double));
    K2 = (long double *) malloc(Ntri*sizeof(long double));
    K3 = (long double *) malloc(Ntri*sizeof(long double));  
    IP = (long double *) malloc(Nk*sizeof(long double));
    I = (long double *) malloc(Ntri*sizeof(long double));  

    /*Computes the bispectrum*/
    Ntri = Bi_Spectrum(grid, nd, L, ntype, window, R, interlacing, Nk, k_min, k_max, K1, K2, K3, B, I, KP, P, IP);

    /*Prepare the PyObject arrays for the outputs*/
    npy_intp dims_B[] = {(npy_intp) NBs, (npy_intp) Ntri};
    npy_intp dims_P[] = {(npy_intp) NPs, (npy_intp) Nk};
    npy_intp dims_kB[] = {(npy_intp) 3, (npy_intp) Ntri};
    npy_intp dims_kP[] = {(npy_intp) Nk};
    npy_intp dims_I[] = {(npy_intp) Ntri};

	/*Alloc the PyObjects for the output*/
    PyArrayObject *np_P = (PyArrayObject *) PyArray_ZEROS(2, dims_P, NP_OUT_TYPE, 0);
    PyArrayObject *np_kP = (PyArrayObject *) PyArray_ZEROS(1, dims_kP, NP_OUT_TYPE, 0);
    PyArrayObject *np_count_k = (PyArrayObject *) PyArray_ZEROS(1, dims_kP, NP_OUT_TYPE, 0);
    PyArrayObject *np_B = (PyArrayObject *) PyArray_ZEROS(2, dims_B, NP_OUT_TYPE, 0);
    PyArrayObject *np_kB = (PyArrayObject *) PyArray_ZEROS(2, dims_kB, NP_OUT_TYPE, 0);
    PyArrayObject *np_count_tri = (PyArrayObject *) PyArray_ZEROS(1, dims_I, NP_OUT_TYPE, 0);   
    P_out = (fft_real *) np_P->data;
    KP_out = (fft_real *) np_kP->data;
    count_k = (fft_real *) np_count_k->data;
    B_out = (fft_real *) np_B->data;
    KB_out = (fft_real *) np_kB->data;
    count_tri = (fft_real *) np_count_tri->data;

    /*Put the values in the outputs*/
    for(i=0;i<Nk;i++){
        KP_out[i] = (fft_real) KP[i];
        count_k[i] = (fft_real) IP[i];
        for(j=0;j<NPs;j++)
            P_out[j*Nk + i] = (fft_real) P[j][i];
    }
    for(i=0;i<Ntri;i++){
        KB_out[i] = (fft_real) K1[i];
        KB_out[i + Ntri] = (fft_real) K2[i];
        KB_out[i + 2*Ntri] = (fft_real) K3[i];   
        count_tri[i] = (fft_real) I[i];  
        for(j=0;j<NBs;j++)
            B_out[j*Ntri + i] = (fft_real) B[j][i];  
    }

    /*Free the arrays*/
    for(i=0;i<NPs;i++)
        free(P[i]);
    for(i=0;i<NBs;i++)
        free(B[i]);
    free(KP); free(K1); free(K2); free(K3);
    free(P); free(B);
    free(IP); free(I);

    /*Output the mesurements in PyObject format*/
    PyObject *tupleresult = PyTuple_New(6);
    PyTuple_SetItem(tupleresult, 0, PyArray_Return(np_kP));
    PyTuple_SetItem(tupleresult, 1, PyArray_Return(np_P));
    PyTuple_SetItem(tupleresult, 2, PyArray_Return(np_count_k));
    PyTuple_SetItem(tupleresult, 3, PyArray_Return(np_kB));
    PyTuple_SetItem(tupleresult, 4, PyArray_Return(np_B));
    PyTuple_SetItem(tupleresult, 5, PyArray_Return(np_count_tri));

    return PyArray_Return((PyArrayObject*) tupleresult);
}

/*Function that computes the cross bispectrum for all tracers and outputs it in numpy format*/
static PyObject *tri_compute(PyObject *self, PyObject *args, PyObject *kwargs){
	int i, j, ntype, nd, window, interlacing, Nk, verbose, nthreads, NPs, NTs, Nsq;
    long double **P, *KP, **T, **Tu, *K1, *K2, *I, *IP;
	fft_real *grid, k_min, k_max, *P_out, *KP_out, *T_out, *Tu_out, *KT_out, *count_k, *count_sq;
	fft_real L, R;

	/*Define the list of parameters*/
	static char *kwlist[] = {"grid", "ntype", "nd", "L", "window", "R", "interlacing", "Nk", "k_min", "k_max", "verbose", "nthreads", NULL};
	import_array();

	/*Define the pyobject with the 3D position of the tracers*/
	PyArrayObject *grid_array;  

	/*Read the input arguments*/
	#ifdef DOUBLEPRECISION_FFTW
		if (!PyArg_ParseTupleAndKeywords(args, kwargs, "Oiididiiddii", kwlist, &grid_array &ntype, &nd, &L, &window, &R, &interlacing, &Nk, &k_min, &k_max, &verbose, &nthreads))
			return NULL;
	#else
		if (!PyArg_ParseTupleAndKeywords(args, kwargs, "Oiififiiffii", kwlist, &grid_array, &ntype, &nd, &L, &window, &R, &interlacing, &Nk, &k_min, &k_max, &verbose, &nthreads))
			return NULL;
	#endif

	/*Convert the PyObjects to C arrays*/
	grid = (fft_real *) grid_array->data;
    NPs = ntype*(ntype+1)/2;
    NTs = pow(NPs, 2);

    /*Initialize FFTW and openmp to run in parallel*/
    omp_set_num_threads(nthreads);
    FFTW(init_threads)();
    FFTW(plan_with_nthreads)(nthreads);

    if(verbose == TRUE){
		printf("Computing the trispectra\n");
		printf("k_min = %f, kmax = %f, Nk = %d, L = %f, Ntype = %d, Ncells = %d, interlacing = %d, window = %d", (float) k_min, (float) k_max, Nk, (float) L, ntype, nd, interlacing, window);
		if(window > 2)
			printf(", R = %f", (float) R);
		printf(", nthreads = %d\n", nthreads);
	}

    /*Allocs the arrays for for the spectra*/
    Nsq = (int) Nk*(Nk+1)/2;
    P = (long double **) malloc(NPs*sizeof(long double *));
    for(i=0;i<NPs;i++)
        P[i] = (long double *) malloc(Nk*sizeof(long double));
    T = (long double **) malloc(NTs*sizeof(long double *));
    Tu = (long double **) malloc(NTs*sizeof(long double *));
    for(i=0;i<NTs;i++){
        T[i] = (long double *) malloc(Nsq*sizeof(long double));    
        Tu[i] = (long double *) malloc(Nsq*sizeof(long double));  
    }
    KP = (long double *) malloc(Nk*sizeof(long double));
    K1 = (long double *) malloc(Nsq*sizeof(long double));
    K2 = (long double *) malloc(Nsq*sizeof(long double));
    IP = (long double *) malloc(Nk*sizeof(long double));
    I = (long double *) malloc(Nsq*sizeof(long double));  

    /*Computes the bispectrum*/
    Nsq = Tri_Spectrum(grid, nd, L, ntype, window, R, interlacing, Nk, k_min, k_max, K1, K2, T, Tu, I, KP, P, IP);

    /*Prepare the PyObject arrays for the outputs*/
    npy_intp dims_T[] = {(npy_intp) NTs, (npy_intp) Nsq};
    npy_intp dims_P[] = {(npy_intp) NPs, (npy_intp) Nk};
    npy_intp dims_kT[] = {(npy_intp) 2, (npy_intp) Nsq};
    npy_intp dims_kP[] = {(npy_intp) Nk};
    npy_intp dims_I[] = {(npy_intp) Nsq};

	/*Alloc the PyObjects for the output*/
    PyArrayObject *np_P = (PyArrayObject *) PyArray_ZEROS(2, dims_P, NP_OUT_TYPE, 0);
    PyArrayObject *np_kP = (PyArrayObject *) PyArray_ZEROS(1, dims_kP, NP_OUT_TYPE, 0);
    PyArrayObject *np_count_k = (PyArrayObject *) PyArray_ZEROS(1, dims_kP, NP_OUT_TYPE, 0);
    PyArrayObject *np_T = (PyArrayObject *) PyArray_ZEROS(2, dims_T, NP_OUT_TYPE, 0);
    PyArrayObject *np_Tu = (PyArrayObject *) PyArray_ZEROS(2, dims_T, NP_OUT_TYPE, 0);       
    PyArrayObject *np_kT = (PyArrayObject *) PyArray_ZEROS(2, dims_kT, NP_OUT_TYPE, 0);
    PyArrayObject *np_count_sq = (PyArrayObject *) PyArray_ZEROS(1, dims_I, NP_OUT_TYPE, 0);  
    P_out = (fft_real *) np_P->data;
    KP_out = (fft_real *) np_kP->data;
    count_k = (fft_real *) np_count_k->data;
    T_out = (fft_real *) np_T->data;
    Tu_out = (fft_real *) np_Tu->data;
    KT_out = (fft_real *) np_kT->data;
    count_sq = (fft_real *) np_count_sq->data;

    /*Put the values in the outputs*/
    for(i=0;i<Nk;i++){
        KP_out[i] = (fft_real) KP[i];
        count_k[i] = (fft_real) IP[i];
        for(j=0;j<NPs;j++)
            P_out[j*Nk + i] = (fft_real) P[j][i];
    }
    for(i=0;i<Nsq;i++){
        KT_out[i] = (fft_real) K1[i];
        KT_out[i + Nsq] = (fft_real) K2[i];
        count_sq[i] = (fft_real) I[i];  
        for(j=0;j<NTs;j++){
            T_out[j*Nsq + i] = (fft_real) T[j][i];  
            Tu_out[j*Nsq + i] = (fft_real) Tu[j][i];
        }
    }

    /*Free the arrays*/
    for(i=0;i<NPs;i++)
        free(P[i]);
    for(i=0;i<NTs;i++)
        free(T[i]);
    for(i=0;i<NTs;i++)
        free(Tu[i]);
    free(KP); free(K1); free(K2);
    free(P); free(T); free(Tu);
    free(IP); free(I);

    /*Output the mesurements in PyObject format*/
    PyObject *tupleresult = PyTuple_New(7);
    PyTuple_SetItem(tupleresult, 0, PyArray_Return(np_kP));
    PyTuple_SetItem(tupleresult, 1, PyArray_Return(np_P));
    PyTuple_SetItem(tupleresult, 2, PyArray_Return(np_count_k));
    PyTuple_SetItem(tupleresult, 3, PyArray_Return(np_kT));
    PyTuple_SetItem(tupleresult, 4, PyArray_Return(np_T));
    PyTuple_SetItem(tupleresult, 5, PyArray_Return(np_Tu));
    PyTuple_SetItem(tupleresult, 6, PyArray_Return(np_count_sq));

    return PyArray_Return((PyArrayObject*) tupleresult);
}

/* This initiates the module using the above definitions. */
#if PY_VERSION_HEX >= 0x03000000
static struct PyModuleDef spectrum = {
   PyModuleDef_HEAD_INIT,
   "power", NULL, -1, spectrum_methods
};

PyMODINIT_FUNC PyInit_spectrum(void)
{
    PyObject *m;
    m = PyModule_Create(&spectrum);
    if (!m) {
        return NULL;
    }
    return m;
}
#else
PyMODINIT_FUNC initspectrum(void)
{
    PyObject *m = Py_InitModule("spectrum", spectrum_methods);
    import_array();
    if (!m) {
        return NULL;
    }
}
#endif