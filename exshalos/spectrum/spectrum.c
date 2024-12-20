#include <stdio.h>
#define SPECTRUM_MODULE

#include "spectrum_h.h"
#include "gridmodule.h"
#include "powermodule.h"
#include "bimodule.h"
#include "trimodule.h"
#include "abundance.h"
#include "bias.h"

/*This declares the compute function*/
static PyObject *spectrum_check_precision(PyObject * self, PyObject * args);
static PyObject *grid_compute(PyObject * self, PyObject * args, PyObject *kwargs);
static PyObject *power_compute(PyObject * self, PyObject * args, PyObject *kwargs);
static PyObject *power_compute_individual(PyObject * self, PyObject * args, PyObject *kwargs);
static PyObject *bi_compute(PyObject * self, PyObject * args, PyObject *kwargs);
static PyObject *tri_compute(PyObject * self, PyObject * args, PyObject *kwargs);
static PyObject *abundance_compute(PyObject * self, PyObject * args, PyObject *kwargs);
static PyObject *histogram_compute(PyObject *self, PyObject *args, PyObject *kwargs);

/*This tells Python what methods this module has. See the Python-C API for more information.*/
static PyMethodDef spectrum_methods[] = {
    {"check_precision", spectrum_check_precision, METH_VARARGS, "Returns precision used by the estimators of the spectra"},
    {"grid_compute", grid_compute, METH_VARARGS | METH_KEYWORDS, "Computes the density grid of a given list of particles"},
    {"power_compute", power_compute, METH_VARARGS | METH_KEYWORDS, "Computes the PowerSpectrum of a given density grid"},
    {"power_compute_individual", power_compute_individual, METH_VARARGS | METH_KEYWORDS, "Computes the PowerSpectrum of a given density grid for individual particles"},
    {"bi_compute", bi_compute, METH_VARARGS | METH_KEYWORDS, "Computes the BiSpectrum of a given density grid"},
    {"tri_compute", tri_compute, METH_VARARGS | METH_KEYWORDS, "Computes the TriSpectrum of a given density grid"},
    {"abundance_compute", abundance_compute, METH_VARARGS | METH_KEYWORDS, "Computes the differential abundance of halos"},
    {"histogram_compute", histogram_compute, METH_VARARGS | METH_KEYWORDS, "Computed the (un)masked histogram of delta"},
    {NULL, NULL, 0, NULL}
};

/*Return the precision used in the grid computations*/
static PyObject *spectrum_check_precision(PyObject * self, PyObject * args){
    return Py_BuildValue("i", sizeof(fft_real));
}

/*Function that computes the density grid for each tracer and outputs it in numpy format*/
static PyObject *grid_compute(PyObject *self, PyObject *args, PyObject *kwargs){
	size_t np;
	int nmass, ntype, nd, window, interlacing, *type, verbose, nthreads, direction, folds;
	fft_real *pos, *vel, *mass, *grid;
	fft_real L, R, R_times, Om0, z;

	/*Define the list of parameters*/
	static char *kwlist[] = {"pos", "vel", "mass", "nmass", "type", "ntype", "nd", "L", "Om0", "z", "direction", "window", "R", "R_times", "interlacing", "folds", "verbose", "nthreads", NULL};
	import_array();

	/*Define the pyobject with the 3D position of the tracers*/
	PyArrayObject *pos_array, *vel_array, *mass_array, *type_array;  

	/*Read the input arguments*/
	#ifdef DOUBLEPRECISION_FFTW
		if (!PyArg_ParseTupleAndKeywords(args, kwargs, "OOOiOiifffiiffiiii", kwlist, &pos_array,  &vel_array, &np, &mass_array, &nmass, &type_array, &ntype, &nd, &L, &Om0, &z, &direction, &window, &R, &R_times, &interlacing, &folds, &verbose, &nthreads))
			return NULL;
	#else
		if (!PyArg_ParseTupleAndKeywords(args, kwargs, "OOOiOiifffiiffiiii", kwlist, &pos_array, &vel_array, &mass_array, &nmass, &type_array, &ntype, &nd, &L, &Om0, &z, &direction, &window, &R, &R_times, &interlacing, &folds, &verbose, &nthreads))
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
	npy_intp dims_grid[] = {(npy_intp) ntype, (npy_intp) interlacing+1, (npy_intp) nd, (npy_intp) nd, (npy_intp) nd};

	/*Alloc the PyObjects for the output*/
	PyArrayObject *np_grid = (PyArrayObject *) PyArray_ZEROS(5, dims_grid, NP_OUT_TYPE, 0);
	grid = (fft_real *) np_grid->data;

	/*Compute the grids for each tracer*/
	Tracer_Grid(grid, nd, L, direction, pos, vel, np, mass, type, ntype, window, R, R_times, interlacing, Om0, z, folds);

	/*Returns the density grids*/
	return PyArray_Return(np_grid);
}

/*Function that computes the cross power spectrum for all tracers and outputs it in numpy format*/
static PyObject *power_compute(PyObject *self, PyObject *args, PyObject *kwargs){
    long *count_k;
	int i, j, k, ntype, nd, window, interlacing, Nk, verbose, nthreads, NPs, l_max, direction, ls;
    long double *P, *Kmean;
	fft_real *grid, k_min, k_max, *P_out, *Kmean_out;
	fft_real L, R;

	/*Define the list of parameters*/
	static char *kwlist[] = {"grid", "ntype", "nd", "L", "window", "R", "interlacing", "Nk", "k_min", "k_max", "l_max", "direction", "verbose", "nthreads", NULL};
	import_array();

	/*Define the pyobject with the 3D position of the tracers*/
	PyArrayObject *grid_array;  

	/*Read the input arguments*/
	#ifdef DOUBLEPRECISION_FFTW
		if (!PyArg_ParseTupleAndKeywords(args, kwargs, "Oiididiiddiiii", kwlist, &grid_array, &ntype, &nd, &L, &window, &R, &interlacing, &Nk, &k_min, &k_max, &l_max, &direction, &verbose, &nthreads))
			return NULL;
	#else
		if (!PyArg_ParseTupleAndKeywords(args, kwargs, "Oiififiiffiiii", kwlist, &grid_array, &ntype, &nd, &L, &window, &R, &interlacing, &Nk, &k_min, &k_max, &l_max, &direction, &verbose, &nthreads))
			return NULL;
	#endif

	/*Convert the PyObjects to C arrays*/
	grid = (fft_real *) grid_array->data;
    NPs = (ntype*(ntype+1))/2;
    ls = floor(l_max/2) + 1;

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
	npy_intp dims_P[] = {(npy_intp) NPs, (npy_intp) ls, (npy_intp) Nk};
    npy_intp dims_k[] = {(npy_intp) Nk};

    /*Alloc the PyObjects for the output*/
    PyArrayObject *np_P = (PyArrayObject *) PyArray_ZEROS(3, dims_P, NP_OUT_TYPE, 0);
    PyArrayObject *np_k = (PyArrayObject *) PyArray_ZEROS(1, dims_k, NP_OUT_TYPE, 0);
    PyArrayObject *np_count_k = (PyArrayObject *) PyArray_ZEROS(1, dims_k, NPY_LONG, 0);   
    P_out = (fft_real *) np_P->data;
    Kmean_out = (fft_real *) np_k->data;
    count_k = (long *) np_count_k->data;
   
    /*Allocs the arrays for P and k*/
    P = (long double *) malloc(NPs*Nk*ls*sizeof(long double));
    Kmean = (long double *) malloc(Nk*sizeof(long double));
    for(i=0;i<Nk;i++){
        Kmean[i] = 0.0;
        for(j=0;j<NPs;j++)
            for(k=0;k<ls;k++)
                P[(j*ls + k)*Nk + i] = 0.0;
    }   

    /*Compute the spectra for all tracers*/
	Power_Spectrum(grid, nd, L, ntype, window, R, interlacing, Nk, k_min, k_max, Kmean, P, count_k, l_max, direction);
   
    /*Put the values in the outputs*/
    for(i=0;i<Nk;i++){
        Kmean_out[i] = (fft_real) Kmean[i];
        for(j=0;j<NPs;j++)
            for(k=0;k<ls;k++)
                P_out[(j*ls + k)*Nk + i] = (fft_real) P[(j*ls + k)*Nk + i];
    }
   
    /*Free the arrays*/
    free(Kmean);
    free(P);
   
    /*Construct the output tuple for each case*/
    PyObject *dict = PyDict_New();
   
    PyDict_SetItemString(dict, "k", PyArray_Return(np_k));
    PyDict_SetItemString(dict, "Pk", PyArray_Return(np_P));
    PyDict_SetItemString(dict, "Nk", PyArray_Return(np_count_k));
   
    return dict;
}

/*Function that computes the cross power spectrum for all individual tracers and outputs it in numpy format*/
static PyObject *power_compute_individual(PyObject *self, PyObject *args, PyObject *kwargs){
    long *count_k;
	int i, j, k, ntype, nd, np, window, interlacing, Nk, verbose, nthreads, NPs, l_max, direction, ls;
    long double *P, *Kmean;
	fft_real *grid, *pos, k_min, k_max, *P_out, *Kmean_out;
	fft_real L, R;

	/*Define the list of parameters*/
	static char *kwlist[] = {"grid", "pos", "ntype", "nd", "L", "window", "R", "interlacing", "Nk", "k_min", "k_max", "l_max", "direction", "verbose", "nthreads", NULL};
	import_array();

	/*Define the pyobject with the 3D position of the tracers*/
	PyArrayObject *grid_array, *pos_array;  

	/*Read the input arguments*/
	#ifdef DOUBLEPRECISION_FFTW
		if (!PyArg_ParseTupleAndKeywords(args, kwargs, "OOiididiiddiiii", kwlist, &grid_array, &pos_array, &ntype, &nd, &L, &window, &R, &interlacing, &Nk, &k_min, &k_max, &l_max, &direction, &verbose, &nthreads))
			return NULL;
	#else
		if (!PyArg_ParseTupleAndKeywords(args, kwargs, "OOiififiiffiiii", kwlist, &grid_array, &pos_array, &ntype, &nd, &L, &window, &R, &interlacing, &Nk, &k_min, &k_max, &l_max, &direction, &verbose, &nthreads))
			return NULL;
	#endif

	/*Convert the PyObjects to C arrays*/
	grid = (fft_real *) grid_array->data;
    pos = (fft_real *) pos_array->data;
    np = (int) pos_array->dimensions[0];
    NPs = ntype*np;
    ls = floor(l_max/2) + 1;

    /*Initialize FFTW and openmp to run in parallel*/
    omp_set_num_threads(nthreads);
    FFTW(init_threads)();
    FFTW(plan_with_nthreads)(nthreads);

    if(verbose == TRUE){
		printf("Computing the power spectra of individual particles\n");
		printf("kmin = %f, kmax = %f, Nk = %d, l_max = %d, L = %f, Ntype = %d, Ncells = %d, interlacing = %d, window = %d", (float) k_min, (float) k_max, Nk, l_max, (float) L, ntype, nd, interlacing, window);
		if(window > 2)
			printf(", R = %f", (float) R);
		printf(", nthreads = %d\n", nthreads);
	}

    /*Prepare the PyObject arrays for the outputs*/
	npy_intp dims_P[] = {(npy_intp) np, (npy_intp) ntype, (npy_intp) ls, (npy_intp) Nk};
    npy_intp dims_k[] = {(npy_intp) Nk};

	/*Alloc the PyObjects for the output*/
    PyArrayObject *np_P = (PyArrayObject *) PyArray_ZEROS(4, dims_P, NP_OUT_TYPE, 0);
    PyArrayObject *np_k = (PyArrayObject *) PyArray_ZEROS(1, dims_k, NP_OUT_TYPE, 0);
    PyArrayObject *np_count_k = (PyArrayObject *) PyArray_ZEROS(1, dims_k, NPY_LONG, 0);   
    P_out = (fft_real *) np_P->data;
    Kmean_out = (fft_real *) np_k->data;
    count_k = (long *) np_count_k->data;

    /*Allocs the arrays for P and k*/
    P = (long double *) malloc(NPs*Nk*ls*sizeof(long double));
    Kmean = (long double *) malloc(Nk*sizeof(long double));
    for(i=0;i<Nk;i++){
        Kmean[i] = 0.0;
        for(j=0;j<NPs;j++)
            for(k=0;k<ls;k++)
                P[(j*ls + k)*Nk + i] = 0.0;
    }   

	/*Compute the spectra for all tracers*/
	Power_Spectrum_individual(grid, pos, np, nd, L, ntype, window, R, interlacing, Nk, k_min, k_max, Kmean, P, count_k, l_max, direction);

    /*Put the values in the outputs*/
    for(i=0;i<Nk;i++){
        Kmean_out[i] = (fft_real) Kmean[i];
        for(j=0;j<NPs;j++)
            for(k=0;k<ls;k++)
                P_out[(j*ls + k)*Nk + i] = (fft_real) P[(j*ls + k)*Nk + i];
    }

    /*Free the arrays*/
    free(Kmean);
    free(P);

    /*Construct the output tuple for each case*/
    PyObject *dict = PyDict_New();
    
    PyDict_SetItemString(dict, "k", PyArray_Return(np_k));
    PyDict_SetItemString(dict, "Pk", PyArray_Return(np_P));
    PyDict_SetItemString(dict, "Nk", PyArray_Return(np_count_k));

    return dict;
}

/*Function that computes the cross bispectrum for all tracers and outputs it in numpy format*/
static PyObject *bi_compute(PyObject *self, PyObject *args, PyObject *kwargs){
	int i, j, ntype, nd, window, interlacing, Nk, verbose, nthreads, NPs, NBs, Ntri;
    long double **P, *KP, **B, *K1, *K2, *K3, *IB, *IP;
	fft_real *grid, k_min, k_max, *P_out, *KP_out, *B_out, *KB_out, *count_k, *count_tri;
	fft_real L, R;

	/*Define the list of parameters*/
	static char *kwlist[] = {"grid", "ntype", "nd", "L", "window", "R", "interlacing", "Nk", "k_min", "k_max", "verbose", "nthreads", NULL};
	import_array();

	/*Define the pyobject with the 3D position of the tracers*/
	PyArrayObject *grid_array;  

	/*Read the input arguments*/
	#ifdef DOUBLEPRECISION_FFTW
		if (!PyArg_ParseTupleAndKeywords(args, kwargs, "Oiididiiddii", kwlist, &grid_array, &ntype, &nd, &L, &window, &R, &interlacing, &Nk, &k_min, &k_max, &verbose, &nthreads))
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
    IB = (long double *) malloc(Ntri*sizeof(long double));  

    /*Computes the bispectrum*/
    Ntri = Bi_Spectrum(grid, nd, L, ntype, window, R, interlacing, Nk, k_min, k_max, K1, K2, K3, B, IB, KP, P, IP, verbose);

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
        count_tri[i] = (fft_real) IB[i];  
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
    free(IP); free(IB);

    /*Construct the output tuple for each case*/
    PyObject *dict = PyDict_New();
    
    PyDict_SetItemString(dict, "kP", PyArray_Return(np_kP));
    PyDict_SetItemString(dict, "Pk", PyArray_Return(np_P));
    PyDict_SetItemString(dict, "Nk", PyArray_Return(np_count_k));
    PyDict_SetItemString(dict, "kB", PyArray_Return(np_kB));
    PyDict_SetItemString(dict, "Bk", PyArray_Return(np_B));
    PyDict_SetItemString(dict, "Ntri", PyArray_Return(np_count_tri));

    return dict;
}

/*Function that computes the cross bispectrum for all tracers and outputs it in numpy format*/
static PyObject *tri_compute(PyObject *self, PyObject *args, PyObject *kwargs){
	int i, j, ntype, nd, window, interlacing, Nk, verbose, nthreads, NPs, NTs, Nsq;
    long double **P, *KP, **T, **Tu, *K1, *K2, *IT, *IP;
	fft_real *grid, k_min, k_max, *P_out, *KP_out, *T_out, *Tu_out, *KT_out, *count_k, *count_sq;
	fft_real L, R;

	/*Define the list of parameters*/
	static char *kwlist[] = {"grid", "ntype", "nd", "L", "window", "R", "interlacing", "Nk", "k_min", "k_max", "verbose", "nthreads", NULL};
	import_array();

	/*Define the pyobject with the 3D position of the tracers*/
	PyArrayObject *grid_array;  

	/*Read the input arguments*/
	#ifdef DOUBLEPRECISION_FFTW
		if (!PyArg_ParseTupleAndKeywords(args, kwargs, "Oiididiiddii", kwlist, &grid_array, &ntype, &nd, &L, &window, &R, &interlacing, &Nk, &k_min, &k_max, &verbose, &nthreads))
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
    IT = (long double *) malloc(Nsq*sizeof(long double));  

    /*Computes the bispectrum*/
    Nsq = Tri_Spectrum(grid, nd, L, ntype, window, R, interlacing, Nk, k_min, k_max, K1, K2, T, Tu, IT, KP, P, IP, verbose);

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
        count_sq[i] = (fft_real) IT[i];  
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
    free(IP); free(IT);

    /*Construct the output tuple for each case*/
    PyObject *dict = PyDict_New();
    
    PyDict_SetItemString(dict, "kP", PyArray_Return(np_kP));
    PyDict_SetItemString(dict, "Pk", PyArray_Return(np_P));
    PyDict_SetItemString(dict, "Nk", PyArray_Return(np_count_k));
    PyDict_SetItemString(dict, "kT", PyArray_Return(np_kT));
    PyDict_SetItemString(dict, "Tk", PyArray_Return(np_T));
    PyDict_SetItemString(dict, "Tuk", PyArray_Return(np_Tu));
    PyDict_SetItemString(dict, "Nsq", PyArray_Return(np_count_sq));

    return dict;
}

/*Function that computes the differential abundance of a given halo catalogue*/
static PyObject *abundance_compute(PyObject *self, PyObject *args, PyObject *kwargs){
    int ndx, ndy, ndz, Nm, verbose;
    size_t i, nh;
    fft_real *Mh, Mmin, Mmax, Lc, *dn, *Mmean, *dn_err, Lx, Ly, Lz;

	/*Define the list of parameters*/
	static char *kwlist[] = {"Mh_array", "Mmin", "Mmax", "Nm", "Lc", "ndx", "ndy", "ndz", "verbose", NULL};
	import_array();

	/*Define the pyobject with the 3D position of the tracers*/
	PyArrayObject *Mh_array;  

	/*Read the input arguments*/
	#ifdef DOUBLEPRECISION_FFTW
		if (!PyArg_ParseTupleAndKeywords(args, kwargs, "Oddidiiii", kwlist, &Mh_array, &Mmin, &Mmax, &Nm, &Lc, &ndx, &ndy, &ndz, &verbose))
			return NULL;
	#else
		if (!PyArg_ParseTupleAndKeywords(args, kwargs, "Offifiiii", kwlist, &Mh_array, &Mmin, &Mmax, &Nm, &Lc, &ndx, &ndy, &ndz, &verbose))
			return NULL;
	#endif

    /*Convert the PyObjects to C arrays*/
    nh = (size_t) Mh_array->dimensions[0];
    Mh = (fft_real *) Mh_array->data;

    /*Set some parameters*/
    Lx = Lc*ndx;
    Ly = Lc*ndy;
    Lz = Lc*ndz;

    /*Find Mmin and Mmax, if they were not given*/
    if(Mmin < 0.0){
        Mmin = 1e+20;
        for(i=0;i<nh;i++)
            if(Mh[i] < Mmin)
                Mmin = Mh[i];
        Mmin = Mmin*0.9999;
    }
    if(Mmax < 0.0){
        for(i=0;i<nh;i++)
            if(Mh[i] > Mmax)
                Mmax = Mh[i];
        Mmax = Mmax*1.0001;
    }

    /*Prepare the PyObject arrays for the outputs*/
    npy_intp dims_M[] = {(npy_intp) Nm};

	/*Alloc the PyObjects for the output*/
    PyArrayObject *np_Mmean = (PyArrayObject *) PyArray_ZEROS(1, dims_M, NP_OUT_TYPE, 0);
    PyArrayObject *np_dn = (PyArrayObject *) PyArray_ZEROS(1, dims_M, NP_OUT_TYPE, 0);
    PyArrayObject *np_dn_err = (PyArrayObject *) PyArray_ZEROS(1, dims_M, NP_OUT_TYPE, 0);
    Mmean = (fft_real *) np_Mmean->data;
    dn = (fft_real *) np_dn->data;
    dn_err = (fft_real *) np_dn_err->data;

    /*Compute the abundance*/
    Measure_Abundance(Mh, nh, Mmin, Mmax, Nm, Mmean, dn, dn_err, Lx, Ly, Lz);

    /*Construct the output tuple for each case*/
    PyObject *dict = PyDict_New();
    
    PyDict_SetItemString(dict, "Mh", PyArray_Return(np_Mmean));
    PyDict_SetItemString(dict, "dn", PyArray_Return(np_dn));
    PyDict_SetItemString(dict, "dn_err", PyArray_Return(np_dn_err));

    return dict;
}

/*Fucntion that computes the the histogram of the delta field and the mask histogram given a halo catalogue*/
static PyObject *histogram_compute(PyObject *self, PyObject *args, PyObject *kwargs){
    int i, Nm, Nbins, Central;
    size_t ndx, ndy, ndz, ng, nh;
    long *flag, *hist_unmasked, *hist_masked;
    fft_real *delta, *Mh, Mmin, Mmax, dmin, dmax, *Mmean, *delta_mean, dlnM, ddelta;

	/*Define the list of parameters*/
	static char *kwlist[] = {"delta", "Mh", "flag", "Mmin", "Mmax", "Nm", "dmin", "dmax", "Nbins", "Central", NULL};
	import_array();

	/*Define the pyobject with the 3D position of the tracers*/
	PyArrayObject *delta_array, *Mh_array, *flag_array;  

	/*Read the input arguments*/
	#ifdef DOUBLEPRECISION_FFTW
		if (!PyArg_ParseTupleAndKeywords(args, kwargs, "OOOddiddii", kwlist, &delta_array, &Mh_array, &flag_array, &Mmin, &Mmax, &Nm, &dmin, &dmax, &Nbins, &Central))
			return NULL;
	#else
		if (!PyArg_ParseTupleAndKeywords(args, kwargs, "OOOffiffii", kwlist, &delta_array, &Mh_array, &flag_array, &Mmin, &Mmax, &Nm, &dmin, &dmax, &Nbins, &Central))
			return NULL;
	#endif

    /*Convert the PyObjects to C arrays*/
    nh = (size_t) Mh_array->dimensions[0];
    ndx = (size_t) delta_array->dimensions[0];
    ndy = (size_t) delta_array->dimensions[1];
    ndz = (size_t) delta_array->dimensions[2];
    ng = ndx*ndy*ndz;
    delta = (fft_real *) delta_array->data;
    Mh = (fft_real *) Mh_array->data;
    flag = (long *) flag_array->data;

    /*Find Mmin and Mmax, if they were not given*/
    if(Mmin < 0.0){
        Mmin = 1e+20;
        for(i=0;i<nh;i++)
            if(Mh[i] < Mmin)
                Mmin = Mh[i];
        Mmin = Mmin*0.9999;
    }
    if(Mmax < 0.0){
        Mmax = 0.0;
        for(i=0;i<nh;i++)
            if(Mh[i] > Mmax)
                Mmax = Mh[i];
        Mmax = Mmax*1.0001;
    }

    /*Prepare the PyObject arrays for the outputs*/
    npy_intp dims_hist_unmasked[] = {(npy_intp) Nbins};
    npy_intp dims_hist_masked[] = {(npy_intp) Nm, (npy_intp) Nbins};
    npy_intp dims_Mmean[] = {(npy_intp) Nm};

	/*Alloc the PyObjects for the output*/
    PyArrayObject *np_hist_unmasked = (PyArrayObject *) PyArray_ZEROS(1, dims_hist_unmasked, NPY_LONG, 0);
    PyArrayObject *np_hist_masked = (PyArrayObject *) PyArray_ZEROS(2, dims_hist_masked, NPY_LONG, 0);
    PyArrayObject *np_Mmean = (PyArrayObject *) PyArray_ZEROS(1, dims_Mmean, NP_OUT_TYPE, 0);
    PyArrayObject *np_delta_mean = (PyArrayObject *) PyArray_ZEROS(1, dims_hist_unmasked, NP_OUT_TYPE, 0);

    hist_unmasked = (long *) np_hist_unmasked->data;
    hist_masked = (long *) np_hist_masked->data;
    Mmean = (fft_real *) np_Mmean->data;
    delta_mean = (fft_real *) np_delta_mean->data;

    /*Find the size of both bins*/
    dlnM = (log10(Mmax) - log10(Mmin))/Nm;
    ddelta = (dmax - dmin)/Nbins;

    /*Compute the central point in the mass and delta bin*/
    for(i=0;i<Nm;i++)
        Mmean[i] = pow(10.0, log10(Mmin) + (i + 0.5)*dlnM);
    for(i=0;i<Nbins;i++)
        delta_mean[i] = dmin + (i + 0.5)*ddelta;

    /*Compute the histograms*/
    Measure_Histogram(delta, Mh, flag, Mmin, Mmax, Nm, dmin, dmax, Nbins, ng, nh, hist_unmasked, hist_masked, Central);

    /*Construct the output tuple for each case*/
    PyObject *dict = PyDict_New();

    PyDict_SetItemString(dict, "delta", PyArray_Return(np_delta_mean));
    PyDict_SetItemString(dict, "Mh", PyArray_Return(np_Mmean)); 
    PyDict_SetItemString(dict, "Unmasked", PyArray_Return(np_hist_unmasked));
    PyDict_SetItemString(dict, "Masked", PyArray_Return(np_hist_masked));

    return dict;
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
