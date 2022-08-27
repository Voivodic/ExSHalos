#define EXSHALOS_MODULE

#include "exshalos_h.h"
#include "density_grid.h"
#include "find_halos.h"
#include "lpt.h"
#include "box.h"

/*This declares the compute function*/
static PyObject *exshalos_check_precision(PyObject * self, PyObject * args);
static PyObject *correlation_compute(PyObject *self, PyObject *args, PyObject *kwargs);
static PyObject *density_grid_compute(PyObject *self, PyObject *args, PyObject *kwargs);
static PyObject *operators_compute(PyObject *self, PyObject *args, PyObject *kwargs);
static PyObject *smooth_field(PyObject *self, PyObject *args, PyObject *kwargs);
static PyObject *find_halos(PyObject *self, PyObject *args, PyObject *kwargs);
static PyObject *lpt_compute(PyObject *self, PyObject *args, PyObject *kwargs);
static PyObject *halos_box_from_pk(PyObject *self, PyObject *args, PyObject *kwargs);
static PyObject *halos_box_from_grid(PyObject *self, PyObject *args, PyObject *kwargs);

/*This tells Python what methods this module has. See the Python-C API for more information.*/
static PyMethodDef exshalos_methods[] = {
    {"check_precision", exshalos_check_precision, METH_VARARGS, "Returns precision used by the estimators of the spectra"},
    {"correlation_compute", correlation_compute, METH_VARARGS | METH_KEYWORDS, "Computes the correlation function or the power spectrum"},
    {"density_grid_compute", density_grid_compute, METH_VARARGS | METH_KEYWORDS, "Generate the gaussian density grid"},
    {"operators_compute", operators_compute, METH_VARARGS | METH_KEYWORDS, "Compute all operatos up to a given order"},
    {"smooth_field", smooth_field, METH_VARARGS | METH_KEYWORDS, "Smooth a given fields in a given scale"},
    {"find_halos", find_halos, METH_VARARGS | METH_KEYWORDS, "Generate the halo catalogue from a given density grid" },
    {"lpt_compute", lpt_compute, METH_VARARGS | METH_KEYWORDS, "Compute the LPT displacements from a given density grid" },
    {"halos_box_from_pk", halos_box_from_pk, METH_VARARGS | METH_KEYWORDS, "Generate a halo catalogue from a linear power spectrum"},
    {"halos_box_from_grid", halos_box_from_grid, METH_VARARGS | METH_KEYWORDS, "Generate a halo catalogue from a density grid"},
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

/*Compute the Gaussian density grid*/
static PyObject *density_grid_compute(PyObject *self, PyObject *args, PyObject *kwargs){
    int ndx, ndy, ndz, outk, Nk, verbose, nthreads, seed, fixed;
    fft_real Lc, R_max, *K, *P, *delta, phase, k_smooth;
    fft_complex *deltak;

	/*Define the list of parameters*/
	static char *kwlist[] = {"k", "P", "R_max", "Ndx", "Ndy", "Ndz", "Lc/Mc", "outk", "seed", "fixed", "phase", "k_smooth", "verbose", "nthreads", NULL};
	import_array();

	/*Define the pyobject with the 3D position of the tracers*/
	PyArrayObject *K_array, *P_array;  

	/*Read the input arguments*/
	#ifdef DOUBLEPRECISION_FFTW
		if (!PyArg_ParseTupleAndKeywords(args, kwargs, "OOdiiidiiiddii", kwlist, &K_array, &P_array, &R_max, &ndx, &ndy, &ndz, &Lc, &outk, &seed, &fixed, &phase, &k_smooth, &verbose, &nthreads))
			return NULL;
	#else
		if (!PyArg_ParseTupleAndKeywords(args, kwargs, "OOfiiifiiiffii", kwlist, &K_array, &P_array, &R_max, &ndx, &ndy, &ndz, &Lc, &outk, &seed, &fixed, &phase, &k_smooth, &verbose, &nthreads))
			return NULL;
	#endif

    if(verbose == TRUE)
        printf("Nd = (%d, %d, %d), Lc = %f, Nthreads = %d, seed = %d\n", ndx, ndy, ndz, Lc, nthreads, seed);

    /*Set the box structure*/
    set_cosmology(0.31, 0.0, 1.686);   //Not used in this function but needed by set box
    set_box(ndx, ndy, ndz, Lc);
    set_barrier(1, 1.0, 0.0, 0.0, seed);

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

    /*Output the grid only in real space*/
    if(outk == FALSE){
        /*Alloc the PyObjects for the output*/
        np_grid = (PyArrayObject *) PyArray_ZEROS(3, dims_grid, NP_OUT_TYPE, 0);
        delta = (fft_real *) np_grid->data;
        deltak = (fft_complex *) FFTW(malloc)(((size_t) ndx)*((size_t) ndy)*((size_t) ndz/2+1)*sizeof(fft_complex));
        check_memory(deltak, "deltak")   
    }

    /*Output the grid in real and Fourier space*/
    else{
        /*Alloc the PyObjects for the output*/
        np_grid = (PyArrayObject *) PyArray_ZEROS(3, dims_grid, NP_OUT_TYPE, 0);
        delta = (fft_real *) np_grid->data;
        np_gridk = (PyArrayObject *) PyArray_ZEROS(4, dims_gridk, NP_OUT_TYPE, 0);
        deltak = (fft_complex *) np_gridk->data;
    }

    /*Compute the density grids*/
    Compute_Den(K, P, Nk, R_max, delta, deltak, fixed, phase, k_smooth);

    /*Put the arrays in the output dict*/
    PyObject *dict = PyDict_New();
    
    PyDict_SetItemString(dict, "grid", PyArray_Return(np_grid));
    if(outk == TRUE)
        PyDict_SetItemString(dict, "gridk", PyArray_Return(np_gridk));
    else
        FFTW(free)(deltak); 

    return dict;       
}

/*Compute all possible scalar operator up to a given order*/
static PyObject *operators_compute(PyObject *self, PyObject *args, PyObject *kwargs){
    size_t ind;
    int ndx, ndy, ndz, nthreads, verbose, order;
    fft_real Lc, *delta, *params, *tidal;
    fft_real *delta2, *K2, *delta3, *K3, *deltaK2;

	/*Define the list of parameters*/
	static char *kwlist[] = {"delta", "Order", "Params", "Lc", "nthreads", "verbose", NULL};
	import_array();

	/*Define the pyobject with the 3D position of the tracers*/
	PyArrayObject *grid_array, *params_array;  

	/*Read the input arguments*/
	#ifdef DOUBLEPRECISION_FFTW
		if (!PyArg_ParseTupleAndKeywords(args, kwargs, "OiOdii", kwlist, &grid_array, &order, &params_array, &Lc, &nthreads, &verbose))
			return NULL;
	#else
		if (!PyArg_ParseTupleAndKeywords(args, kwargs, "OiOfii", kwlist, &grid_array, &order, &params_array, &Lc, &nthreads, &verbose))
			return NULL;
	#endif

    /*Convert PyObjects to C arrays*/
    ndx = (int) grid_array->dimensions[0];
    ndy = (int) grid_array->dimensions[1];
    ndz = (int) grid_array->dimensions[2];
    delta = (fft_real *) grid_array->data;
    params = (fft_real *) params_array->data;
   
    /*Set the box structure*/
    set_box(ndx, ndy, ndz, Lc);

    /*Initialize FFTW and openmp to run in parallel*/
    omp_set_num_threads(nthreads);
    FFTW(init_threads)();
    FFTW(plan_with_nthreads)(nthreads);

    /*Alloc and compute the tidal field*/
    tidal = (fft_real *) malloc(6*(size_t)box.ng*sizeof(fft_real));
    check_memory(tidal, "tidal")

    Compute_Tidal(delta, NULL, tidal);

    /*Define the variables for the output*/
    npy_intp dims_grid[] = {(npy_intp) box.nd[0], (npy_intp) box.nd[1], (npy_intp) box.nd[2]};
    PyObject *dict = PyDict_New();

    /*Alloc and compute the second order operators*/
    if(order >= 2){
        PyArrayObject *np_delta2, *np_K2;
        np_delta2 = (PyArrayObject *) PyArray_ZEROS(3, dims_grid, NP_OUT_TYPE, 0);
        np_K2 = (PyArrayObject *) PyArray_ZEROS(3, dims_grid, NP_OUT_TYPE, 0);

        delta2 = (fft_real *) np_delta2->data;
        K2 = (fft_real *) np_K2->data;

        /*delta^2*/
        Compute_Den_to_n(delta, delta2, 2);
        PyDict_SetItemString(dict, "delta2", PyArray_Return(np_delta2));

        /*K^2*/
        Compute_K2(delta, NULL, tidal, K2, params[0]);
        PyDict_SetItemString(dict, "K2", PyArray_Return(np_K2));
    }

    /*Alloc and compute the third order operators*/
    if(order >= 3){
        PyArrayObject *np_delta3, *np_K3, *np_deltaK2;
        np_delta3 = (PyArrayObject *) PyArray_ZEROS(3, dims_grid, NP_OUT_TYPE, 0);
        np_K3 = (PyArrayObject *) PyArray_ZEROS(3, dims_grid, NP_OUT_TYPE, 0);
        np_deltaK2 = (PyArrayObject *) PyArray_ZEROS(3, dims_grid, NP_OUT_TYPE, 0);

        delta3 = (fft_real *) np_delta3->data;
        K3 = (fft_real *) np_K3->data;
        deltaK2 = (fft_real *) np_deltaK2->data;

        /*delta^3*/
        Compute_Den_to_n(delta, delta3, 3);
        PyDict_SetItemString(dict, "delta3", PyArray_Return(np_delta3));

        /*K^3*/
        Compute_K3(delta, NULL, tidal, K3, params[1], params[2]);
        PyDict_SetItemString(dict, "K3", PyArray_Return(np_K3));

        /*delta*K^2*/
        for(ind=0;ind<box.ng;ind++)
            deltaK2[ind] = delta[ind]*K2[ind];
        PyDict_SetItemString(dict, "deltaK2", PyArray_Return(np_deltaK2));
    }   

    /*Free the tidal array*/
    free(tidal);

    return dict;       
}

/*Compute all possible scalar operator up to a given order*/
static PyObject *smooth_field(PyObject *self, PyObject *args, PyObject *kwargs){
    size_t ind;
    int ndx, ndy, ndz, nthreads, verbose, INk, i, j, k;
    fft_real Lc, ksmooth, *delta, *delta_s, kmod, kvec[3];
    fft_complex *deltak, *deltak_tmp;

	/*Define the list of parameters*/
	static char *kwlist[] = {"delta", "Lc", "ksmooth", "INk", "nthreads", "verbose", NULL};
	import_array();

	/*Define the pyobject with the 3D position of the tracers*/
	PyArrayObject *grid_array;  

	/*Read the input arguments*/
	#ifdef DOUBLEPRECISION_FFTW
		if (!PyArg_ParseTupleAndKeywords(args, kwargs, "Oddiii", kwlist, &grid_array, &Lc, &ksmooth, &INk, &nthreads, &verbose))
			return NULL;
	#else
		if (!PyArg_ParseTupleAndKeywords(args, kwargs, "Offiii", kwlist, &grid_array, &Lc, &ksmooth, &INk, &nthreads, &verbose))
			return NULL;
	#endif

    /*Convert PyObjects to C arrays*/
    ndx = (int) grid_array->dimensions[0];
    ndy = (int) grid_array->dimensions[1];
    ndz = (int) grid_array->dimensions[2];
    if(INk == TRUE){
        ndz = 2*(ndz - 1);
        deltak = (fft_complex *) grid_array->data;
    }
    else
        delta = (fft_real *) grid_array->data;
   
    /*Set the box structure*/
    set_box(ndx, ndy, ndz, Lc);

    /*Initialize FFTW and openmp to run in parallel*/
    omp_set_num_threads(nthreads);
    FFTW(init_threads)();
    FFTW(plan_with_nthreads)(nthreads);

    /*Compute the density grid in Fourier space, if needed*/
    deltak_tmp = (fft_complex *) FFTW(malloc)(((size_t) box.nd[0])*((size_t) box.nd[1])*((size_t) box.nz2)*sizeof(fft_complex));
    check_memory(deltak_tmp, "deltak_tmp")
    if(INk == FALSE)
        Compute_Denk(delta, deltak_tmp);
    else
        for(ind=0;ind<((size_t) box.nd[0])*((size_t) box.nd[1])*((size_t) box.nz2);ind++){
            deltak_tmp[ind][0] = deltak[ind][0];      
            deltak_tmp[ind][1] = deltak[ind][1];                
        }

    /*Set the modes larger than ksmooth to zero*/
    for(i=0;i<box.nd[0];i++){
        if(2*i<box.nd[0]) kvec[0] = i*box.kl[0];
        else kvec[0] = (i-box.nd[0])*box.kl[0];
    
        for(j=0;j<box.nd[1];j++){
            if(2*j<box.nd[1]) kvec[1] = j*box.kl[1];
            else kvec[1] = (j-box.nd[1])*box.kl[1];
    
            for(k=0;k<box.nz2;k++){
                kvec[2] = k*box.kl[2];
                if(k == box.nd[2]/2) kvec[2] = -(fft_real)box.nd[2]/2.0*box.kl[2];

                kmod = sqrt(pow(kvec[0], 2.0) + pow(kvec[1], 2.0) + pow(kvec[2], 2.0));
                ind = (size_t)(i*box.nd[1] + j)*(size_t)box.nz2 + (size_t)k;

                if(kmod > ksmooth){
                    deltak_tmp[ind][0] = 0.0;
                    deltak_tmp[ind][1] = 0.0;
                }
            }
        }
    }

    /*Define the variables for the output*/
    npy_intp dims_grid[] = {(npy_intp) box.nd[0], (npy_intp) box.nd[1], (npy_intp) box.nd[2]};

    PyArrayObject *np_delta_s;
    np_delta_s = (PyArrayObject *) PyArray_ZEROS(3, dims_grid, NP_OUT_TYPE, 0);
    delta_s = (fft_real *) np_delta_s->data;

    /*Compute the grid in real space*/
    FFTW(plan) p1;
    p1 = FFTW(plan_dft_c2r_3d)(box.nd[0], box.nd[1], box.nd[2], deltak_tmp, delta_s, FFTW_ESTIMATE); 
    FFTW(execute)(p1);
    FFTW(destroy_plan)(p1);

    /*Normalize phi*/
    #pragma omp parallel for private(ind) 
    for(ind=0;ind<box.ng;ind++)
        delta_s[ind] = box.Normx*delta_s[ind];

    /*Free memory*/
    FFTW(free)(deltak_tmp);

    PyArray_Return(np_delta_s);
}

/*Generate the halo catalogue from a given density grid*/
static PyObject *find_halos(PyObject *self, PyObject *args, PyObject *kwargs){
    char DO_EB;
    size_t nh, ind;
    long *flag;
    int i, j, ndx, ndy, ndz, Nk, Nmin, verbose, OUT_FLAG;
    fft_real Om0, redshift, dc, Lc, a, beta, alpha, *K, *P, *delta, *Mh, *posh;
    HALOS *halos;

	/*Define the list of parameters*/
	static char *kwlist[] = {"delta", "k", "P", "Lc", "Om0", "redshift", "dc", "Nmin", "a", "beta", "alpha", "OUT_FLAG" "verbose", NULL};
	import_array();

	/*Define the pyobject with the 3D position of the tracers*/
	PyArrayObject *grid_array, *K_array, *P_array;  

	/*Read the input arguments*/
	#ifdef DOUBLEPRECISION_FFTW
		if (!PyArg_ParseTupleAndKeywords(args, kwargs, "OOOddddidddii", kwlist, &grid_array, &K_array, &P_array, &Lc, &Om0, &redshift, &dc, &Nmin, &a, &beta, &alpha, &OUT_FLAG, &verbose))
			return NULL;
	#else
		if (!PyArg_ParseTupleAndKeywords(args, kwargs, "OOOffffifffii", kwlist, &grid_array, &K_array, &P_array, &Lc, &Om0, &redshift, &dc, &Nmin, &a, &beta, &alpha, &OUT_FLAG, &verbose))
			return NULL;
	#endif

    if(verbose == TRUE)
        printf("Nd = (%d, %d, %d), Lc = %f, Nmin = %d, a = %f, beta = %f, alpha = %f\n", ndx, ndy, ndz, Lc, Nmin, (float) a, (float) beta, (float) alpha);

    /*Convert PyObjects to C arrays*/
    Nk = (int) K_array->dimensions[0];
    K = (fft_real *) K_array->data;
    P = (fft_real *) P_array->data;
    ndx = (int) grid_array->dimensions[0];
    ndy = (int) grid_array->dimensions[1];
    ndz = (int) grid_array->dimensions[2];
    delta = (fft_real *) grid_array->data;
    if(a == 1.0 && beta == 0.0 && alpha == 0.0)
        DO_EB = FALSE;
    else
        DO_EB = TRUE;

    /*Set the box structure*/
    set_cosmology(Om0, redshift, dc);  
    set_box(ndx, ndy, ndz, Lc);
    set_barrier(Nmin, a, beta, alpha, 12345);
    set_out(0, 1, 0, 0, 0, DO_EB, 0, (char) verbose);

    /*Alloc the flag array*/
    PyArrayObject *np_flag;
    if(OUT_FLAG == TRUE){
        npy_intp dims_flag[] = {(npy_intp) box.nd, (npy_intp) box.nd, (npy_intp) box.nd};
        np_flag = (PyArrayObject *) PyArray_ZEROS(3, dims_flag, PyArray_LONG, 0);
        flag = (long *) np_flag->data;
    }
    else{
        flag = (long *) malloc(box.ng*sizeof(long));
        check_memory(flag, "flag")
    }

	/*Initialize the flag array*/
	for(ind=0;ind<box.ng;ind++)
		flag[ind] = (long) box.ng;

    /*Find the halos in the density grid*/
    nh = Find_Halos(delta, K, P, Nk, flag, &halos);
    if(OUT_FLAG == FALSE)
        free(flag);

    /*Define the variables for the output*/
    npy_intp dims_pos[] = {(npy_intp) nh, (npy_intp) 3};
    npy_intp dims_Mh[] = {(npy_intp) nh};
    PyArrayObject *np_pos, *np_Mh;

    /*Alloc the PyObjects for the output*/
    np_pos = (PyArrayObject *) PyArray_ZEROS(2, dims_pos, NP_OUT_TYPE, 0);
    np_Mh = (PyArrayObject *) PyArray_ZEROS(1, dims_Mh, NP_OUT_TYPE, 0);

    posh = (fft_real *) np_pos->data;
    Mh = (fft_real *) np_Mh->data;

    /*Put the output in the numpy format*/
    for(i=0;i<nh;i++){
        for(j=0;j<3;j++)
            posh[i*3+j] = (fft_real) Lc*(halos[i].x[j] + 0.5);
        Mh[i] = halos[i].Mh;
    }
    free(halos);

    /*Put the arrays in the output dict*/
    PyObject *dict = PyDict_New();
    
    if(OUT_FLAG == FALSE){
        PyDict_SetItemString(dict, "posh", PyArray_Return(np_pos));
        PyDict_SetItemString(dict, "Mh", PyArray_Return(np_Mh));
    }
    else{
        PyDict_SetItemString(dict, "posh", PyArray_Return(np_pos));
        PyDict_SetItemString(dict, "Mh", PyArray_Return(np_Mh)); 
        PyDict_SetItemString(dict, "flag", PyArray_Return(np_flag));
    }

    return dict;       
}

/*Compute the position of the particles using LPT*/
static PyObject *lpt_compute(PyObject *self, PyObject *args, PyObject *kwargs){
    int ndx, ndy, ndz, verbose, DO_2LPT, OUT_VEL, INk, OUT_POS;
    fft_real Om0, redshift, Lc, k_smooth, *delta, *S, *V;
    fft_complex *deltak;

	/*Define the list of parameters*/
	static char *kwlist[] = {"delta", "Lc", "Om0", "redshift",  "k_smooth", "DO_2LPT", "OUT_VEL", "Ink", "OUT_POS", "verbose", NULL};
	import_array();

	/*Define the pyobject with the 3D position of the tracers*/
	PyArrayObject *grid_array;  

	/*Read the input arguments*/
	#ifdef DOUBLEPRECISION_FFTW
		if (!PyArg_ParseTupleAndKeywords(args, kwargs, "Oddddiiiii", kwlist, &grid_array, &Lc, &Om0, &redshift, &k_smooth, &DO_2LPT, &OUT_VEL, &INk, &OUT_POS, &verbose))
			return NULL;
	#else
		if (!PyArg_ParseTupleAndKeywords(args, kwargs, "Offffiiiii", kwlist, &grid_array, &Lc, &Om0, &redshift, &k_smooth, &DO_2LPT, &OUT_VEL, &INk, &OUT_POS, &verbose))
			return NULL;
	#endif

    /*Convert PyObjects to C arrays*/
    ndx = (int) grid_array->dimensions[0];
    ndy = (int) grid_array->dimensions[1];
    ndz = (int) grid_array->dimensions[2];
    if(INk == TRUE){
        ndz = 2*(ndz - 1);
        deltak = (fft_complex *) grid_array->data;
    }
    else
        delta = (fft_real *) grid_array->data;

    /*Set the box structure*/
    set_cosmology(Om0, redshift, -1.0);  
    set_box(ndx, ndy, ndz, Lc);
    set_out(0, 0, 1, (char) OUT_VEL, (char) DO_2LPT, 0, 0, (char) verbose);

    /*Compute the density grid in Fourier space, if needed*/
    if(INk == FALSE){
        deltak = (fft_complex *) FFTW(malloc)(((size_t) box.nd[0])*((size_t) box.nd[1])*((size_t) box.nz2)*sizeof(fft_complex));
        check_memory(deltak, "deltak")

        Compute_Denk(delta, deltak);
    }

    /*Define the variables for the output*/
    npy_intp dims_pos[] = {(npy_intp) box.ng, (npy_intp) 3};
    PyArrayObject *np_S, *np_V;
    PyObject *dict = PyDict_New();
    
    /*Do not output the velocities*/
    if(OUT_VEL == FALSE){
        np_S = (PyArrayObject *) PyArray_ZEROS(2, dims_pos, NP_OUT_TYPE, 0);
        S = (fft_real *) np_S->data;

        /*Compute the LPT*/
        Compute_1LPT(deltak, NULL, NULL, S, NULL, NULL, k_smooth);
        if(DO_2LPT == TRUE)
            Compute_2LPT(NULL, NULL, S, NULL, NULL, k_smooth);

        /*Compute the final positions of the particles*/
        if(OUT_POS == TRUE)
            Compute_Pos(S);

        /*Output the mesurements in PyObject format*/
        PyDict_SetItemString(dict, "pos", PyArray_Return(np_S)); 
    }

    /*Output the velocities*/
    else{
        np_S = (PyArrayObject *) PyArray_ZEROS(2, dims_pos, NP_OUT_TYPE, 0);
        np_V = (PyArrayObject *) PyArray_ZEROS(2, dims_pos, NP_OUT_TYPE, 0);
        S = (fft_real *) np_S->data;
        V = (fft_real *) np_V->data;

        /*Compute the LPT*/
        Compute_1LPT(deltak, NULL, NULL, S, V, NULL, k_smooth);
        if(DO_2LPT == TRUE)
            Compute_2LPT(NULL, NULL, S, V, NULL, k_smooth);

        /*Compute the final positions of the particles*/
        if(OUT_POS == TRUE)
            Compute_Pos(S);

        /*Output the mesurements in PyObject format*/
        PyDict_SetItemString(dict, "pos", PyArray_Return(np_S));
        PyDict_SetItemString(dict, "vel", PyArray_Return(np_V));
    }

    /*Free memory*/
    if(INk == FALSE)
        FFTW(free)(deltak);

    return dict; 
}

/*Generate the halo catalogue from a given linear power spectrum*/
static PyObject *halos_box_from_pk(PyObject *self, PyObject *args, PyObject *kwargs){
    size_t nh, ind;
    long *flag;
    int i, j, ndx, ndy, ndz, Nk, verbose, nthreads, seed, OUT_DEN, OUT_LPT, Nmin, DO_EB, OUT_VEL, DO_2LPT, OUT_FLAG, fixed;
    fft_real Lc, R_max, k_smooth, *K, *P, *delta, *S, *V, Om0, redshift, dc, a, beta, alpha, *posh, *velh, *posh_out, *velh_out, *Mh, phase;
    HALOS *halos;

	/*Define the list of parameters*/
	static char *kwlist[] = {"k", "P", "R_max", "Ndx", "Ndy", "Ndz", "Lc/Mc", "seed", "k_smooth", "Om0", "redshift", "dc", "Nmin", "a", "beta", "alpha", "fixed", "phase", "OUT_DEN", "OUT_LPT", "OUT_VEL", "DO_2LPT", "OUT_FLAG", "verbose", "nthreads", NULL};
	import_array();

	/*Define the pyobject with the 3D position of the tracers*/
	PyArrayObject *K_array, *P_array;  

	/*Read the input arguments*/
	#ifdef DOUBLEPRECISION_FFTW
		if (!PyArg_ParseTupleAndKeywords(args, kwargs, "OOdiiididdddidddidiiiiiii", kwlist, &K_array, &P_array, &R_max, &ndx, &ndy, &ndz, &Lc, &seed, &k_smooth, &Om0, &redshift, &dc, &Nmin, &a, &beta, &alpha, &fixed, &phase, &OUT_DEN, &OUT_LPT, &OUT_VEL, &DO_2LPT, &OUT_FLAG, &verbose, &nthreads))
			return NULL;
	#else
		if (!PyArg_ParseTupleAndKeywords(args, kwargs, "OOfiiififfffifffifiiiiiii", kwlist, &K_array, &P_array, &R_max, &ndx, &ndy, &ndz, &Lc, &seed, &k_smooth, &Om0, &redshift, &dc, &Nmin, &a, &beta, &alpha, &fixed, &phase, &OUT_DEN, &OUT_LPT, &OUT_VEL, &DO_2LPT, &OUT_FLAG, &verbose, &nthreads))
			return NULL;
	#endif

    if(verbose == TRUE)
        printf("Nd = (%d, %d, %d), Lc = %f, Nthreads = %d, seed = %d\n", ndx, ndy, ndz, Lc, nthreads, seed);

    /*Set the box structure*/
    if(a == 1.0 && beta == 0.0 && alpha == 0.0)
        DO_EB = FALSE;
    else
        DO_EB = TRUE;
    set_cosmology(Om0, redshift, dc);   
    set_box(ndx, ndy, ndz, Lc);
    set_barrier(Nmin, a, beta, alpha, seed);
    set_out((char) OUT_DEN, 1, (char) OUT_LPT, (char) OUT_VEL, (char) DO_2LPT, (char) DO_EB, 0, (char) verbose);

	/*Convert the PyObjects to C arrays*/
    Nk = (int) K_array->dimensions[0];
    K = (fft_real *) K_array->data;
    P = (fft_real *) P_array->data;

    /*Initialize FFTW and openmp to run in parallel*/
    omp_set_num_threads(nthreads);
    FFTW(init_threads)();
    FFTW(plan_with_nthreads)(nthreads);

    /*Alloc some arrays for the cases of outputing intermediate results*/
    npy_intp dims_grid[] = {(npy_intp) box.nd[0], (npy_intp) box.nd[1], (npy_intp) box.nd[2]};
    npy_intp dims_S[] = {(npy_intp) box.ng, (npy_intp) 3};
    npy_intp dims_flag[] = {(npy_intp) box.nd, (npy_intp) box.nd, (npy_intp) box.nd};
    PyArrayObject *np_grid, *np_S, *np_V, *np_flag;

    /*Alloc the density grid*/
    if(out.OUT_DEN == TRUE){
        np_grid = (PyArrayObject *) PyArray_ZEROS(3, dims_grid, NP_OUT_TYPE, 0);
        delta = (fft_real *) np_grid->data;
    }
    else
        delta = NULL;

    /*Alloc the displacements and velocities of the particles*/
    if(out.OUT_LPT == TRUE){
        np_S = (PyArrayObject *) PyArray_ZEROS(2, dims_S, NP_OUT_TYPE, 0);
        S = (fft_real *) np_S->data;
        if(out.OUT_VEL == TRUE){
            np_V = (PyArrayObject *) PyArray_ZEROS(2, dims_S, NP_OUT_TYPE, 0);
            V = (fft_real *) np_V->data;          
        }
        else    
            V = NULL;
    }
    else{
        S = NULL;
        V = NULL;
    }

    /*Alloc the flag array*/
    if(OUT_FLAG == TRUE){
        np_flag = (PyArrayObject *) PyArray_ZEROS(3, dims_flag, PyArray_LONG, 0);
        flag = (long *) np_flag->data;
    }
    else{
        flag = (long *) malloc(box.ng*sizeof(long));
        check_memory(flag, "flag")
    }
    
    /*Initialize the flag array*/
	for(ind=0;ind<box.ng;ind++)
		flag[ind] = (long) box.ng;

    /*Generate the halo catalogue from the linear power spectrum*/
    nh = Generate_Halos_Box_from_Pk(K, P, Nk, R_max, k_smooth, &halos, &posh, &velh, flag, delta, S, V, fixed, phase);
    if(OUT_FLAG == FALSE)
        free(flag);

    /*Alloc the output arrays*/
    npy_intp dims_posh[] = {(npy_intp) nh, (npy_intp) 3};
    npy_intp dims_Mh[] = {(npy_intp) nh};
    PyArrayObject *np_posh, *np_velh, *np_Mh;

    np_Mh = (PyArrayObject *) PyArray_ZEROS(1, dims_Mh, NP_OUT_TYPE, 0);
    np_posh = (PyArrayObject *) PyArray_ZEROS(2, dims_posh, NP_OUT_TYPE, 0);
    Mh = (fft_real *) np_Mh->data;
    posh_out = (fft_real *) np_posh->data;
    if(out.OUT_VEL == TRUE){
        np_velh = (PyArrayObject *) PyArray_ZEROS(2, dims_posh, NP_OUT_TYPE, 0);
        velh_out = (fft_real *) np_velh->data;       
    }

    /*Put the values in the output arrays*/
    for(i=0;i<nh;i++){
        Mh[i] = halos[i].Mh;
        for(j=0;j<3;j++){
            ind = (size_t) 3*i+j;
            posh_out[ind] = posh[ind];
            if(out.OUT_VEL == TRUE)
                velh_out[ind] = velh[ind];
        }
    }

    /*Free the original arrays with the halo information*/
    free(posh);
    free(halos);
    if(out.OUT_VEL == TRUE) free(velh);

    /*Construct the output tuple for each case*/
    PyObject *dict = PyDict_New();

    if(OUT_DEN == FALSE && OUT_LPT == FALSE && OUT_VEL == FALSE && OUT_FLAG == FALSE){
        PyDict_SetItemString(dict, "posh", PyArray_Return(np_posh));
        PyDict_SetItemString(dict, "Mh", PyArray_Return(np_Mh));       
    }
    else if(OUT_DEN == FALSE && OUT_LPT == FALSE && OUT_VEL == TRUE && OUT_FLAG == FALSE){
        PyDict_SetItemString(dict, "posh", PyArray_Return(np_posh));
        PyDict_SetItemString(dict, "velh", PyArray_Return(np_velh));
        PyDict_SetItemString(dict, "Mh", PyArray_Return(np_Mh));
    }
    else if(OUT_DEN == FALSE && OUT_LPT == TRUE && OUT_VEL == FALSE && OUT_FLAG == FALSE){
        PyDict_SetItemString(dict, "posh", PyArray_Return(np_posh));
        PyDict_SetItemString(dict, "Mh", PyArray_Return(np_Mh));
        PyDict_SetItemString(dict, "pos", PyArray_Return(np_S));
    }
    else if(OUT_DEN == FALSE && OUT_LPT == TRUE && OUT_VEL == TRUE && OUT_FLAG == TRUE){
        PyDict_SetItemString(dict, "posh", PyArray_Return(np_posh));
        PyDict_SetItemString(dict, "velh", PyArray_Return(np_velh));
        PyDict_SetItemString(dict, "Mh", PyArray_Return(np_Mh));
        PyDict_SetItemString(dict, "pos", PyArray_Return(np_S));
        PyDict_SetItemString(dict, "vel", PyArray_Return(np_V));
        PyDict_SetItemString(dict, "flag", PyArray_Return(np_flag));
    }   
    else if(OUT_DEN == TRUE && OUT_LPT == TRUE && OUT_VEL == TRUE && OUT_FLAG == TRUE){
        PyDict_SetItemString(dict, "posh", PyArray_Return(np_posh));
        PyDict_SetItemString(dict, "velh", PyArray_Return(np_velh));
        PyDict_SetItemString(dict, "Mh", PyArray_Return(np_Mh));
        PyDict_SetItemString(dict, "pos", PyArray_Return(np_S));
        PyDict_SetItemString(dict, "vel", PyArray_Return(np_V));
        PyDict_SetItemString(dict, "flag", PyArray_Return(np_flag));
        PyDict_SetItemString(dict, "grid", PyArray_Return(np_grid));
    }  
    else if(OUT_DEN == TRUE && OUT_LPT == FALSE && OUT_VEL == TRUE && OUT_FLAG == FALSE){
        PyDict_SetItemString(dict, "posh", PyArray_Return(np_posh));
        PyDict_SetItemString(dict, "velh", PyArray_Return(np_velh));
        PyDict_SetItemString(dict, "Mh", PyArray_Return(np_Mh));
        PyDict_SetItemString(dict, "grid", PyArray_Return(np_grid));
    } 
    else if(OUT_DEN == TRUE && OUT_LPT == TRUE && OUT_VEL == FALSE && OUT_FLAG == FALSE){
        PyDict_SetItemString(dict, "posh", PyArray_Return(np_posh));
        PyDict_SetItemString(dict, "Mh", PyArray_Return(np_Mh));
        PyDict_SetItemString(dict, "pos", PyArray_Return(np_S));
        PyDict_SetItemString(dict, "grid", PyArray_Return(np_grid));
    } 
    else if(OUT_DEN == TRUE && OUT_LPT == TRUE && OUT_VEL == FALSE && OUT_FLAG == TRUE){
        PyDict_SetItemString(dict, "posh", PyArray_Return(np_posh));
        PyDict_SetItemString(dict, "Mh", PyArray_Return(np_Mh));
        PyDict_SetItemString(dict, "pos", PyArray_Return(np_S));
        PyDict_SetItemString(dict, "flag", PyArray_Return(np_flag));
        PyDict_SetItemString(dict, "grid", PyArray_Return(np_grid));
    } 
    else if(OUT_DEN == FALSE && OUT_LPT == TRUE && OUT_VEL == FALSE && OUT_FLAG == TRUE){
        PyDict_SetItemString(dict, "posh", PyArray_Return(np_posh));
        PyDict_SetItemString(dict, "Mh", PyArray_Return(np_Mh));
        PyDict_SetItemString(dict, "pos", PyArray_Return(np_S));
        PyDict_SetItemString(dict, "flag", PyArray_Return(np_flag));
    }

    return dict; 
}

/*Generate the halo catalogue from a given linear power spectrum*/
static PyObject *halos_box_from_grid(PyObject *self, PyObject *args, PyObject *kwargs){
    size_t nh, ind;
    long *flag;
    int i, j, ndx, ndy, ndz, Nk, verbose, nthreads, OUT_LPT, Nmin, DO_EB, OUT_VEL, DO_2LPT, OUT_FLAG, IN_disp;
    fft_real Lc, k_smooth, *K, *P, *delta, *S, *V, Om0, redshift, dc, a, beta, alpha, *posh, *velh, *posh_out, *velh_out, *Mh;
    HALOS *halos;

	/*Define the list of parameters*/
	static char *kwlist[] = {"k", "P", "grid", "S", "V", "Lc/Mc", "k_smooth", "Om0", "redshift", "dc", "Nmin", "a", "beta", "alpha", "OUT_LPT", "OUT_VEL", "DO_2LPT", "OUT_FLAG", "IN_disp", "verbose", "nthreads", NULL};
	import_array();

	/*Define the pyobject with the 3D position of the tracers*/
	PyArrayObject *K_array, *P_array, *grid_array, *S_array, *V_array;  

	/*Read the input arguments*/
	#ifdef DOUBLEPRECISION_FFTW
		if (!PyArg_ParseTupleAndKeywords(args, kwargs, "OOOOOdddddidddiiiiiii", kwlist, &K_array, &P_array, &grid_array, &S_array, &V_array, &Lc, &k_smooth, &Om0, &redshift, &dc, &Nmin, &a, &beta, &alpha, &OUT_LPT, &OUT_VEL, &DO_2LPT, &OUT_FLAG, &IN_disp, &verbose, &nthreads))
			return NULL;
	#else
		if (!PyArg_ParseTupleAndKeywords(args, kwargs, "OOOOOfffffifffiiiiiii", kwlist, &K_array, &P_array, &grid_array, &S_array, &V_array,  &Lc, &k_smooth, &Om0, &redshift, &dc, &Nmin, &a, &beta, &alpha, &OUT_LPT, &OUT_VEL, &DO_2LPT, &OUT_FLAG, &IN_disp, &verbose, &nthreads))
			return NULL;
	#endif

	/*Convert the PyObjects to C arrays*/
    Nk = (int) K_array->dimensions[0];
    K = (fft_real *) K_array->data;
    P = (fft_real *) P_array->data;
    ndx = (int) grid_array->dimensions[0];
    ndy = (int) grid_array->dimensions[1];
    ndz = (int) grid_array->dimensions[2];
    delta = (fft_real *) grid_array->data;
    if(IN_disp == TRUE){
        S = (fft_real *) S_array->data;
        if(OUT_VEL == TRUE)
            V = (fft_real *) V_array->data;
    }

    /*Set the box structure*/
    if(a == 1.0 && beta == 0.0 && alpha == 0.0)
        DO_EB = FALSE;
    else
        DO_EB = TRUE;
    set_cosmology(Om0, redshift, dc);   
    set_box(ndx, ndy, ndz, Lc);
    set_barrier(Nmin, a, beta, alpha, 1234);
    set_out(0, 1, (char) OUT_LPT, (char) OUT_VEL, (char) DO_2LPT, (char) DO_EB, 0, (char) verbose);

    /*Initialize FFTW and openmp to run in parallel*/
    omp_set_num_threads(nthreads);
    FFTW(init_threads)();
    FFTW(plan_with_nthreads)(nthreads);

    /*Alloc some arrays for the cases of outputing intermediate results*/
    npy_intp dims_S[] = {(npy_intp) box.ng, (npy_intp) 3};
    npy_intp dims_flag[] = {(npy_intp) box.nd, (npy_intp) box.nd, (npy_intp) box.nd};
    PyArrayObject *np_S, *np_V, *np_flag;

    /*Alloc the displacements and velocities of the particles*/
    if(out.OUT_LPT == TRUE && IN_disp == FALSE){
        np_S = (PyArrayObject *) PyArray_ZEROS(2, dims_S, NP_OUT_TYPE, 0);
        S = (fft_real *) np_S->data;
        if(out.OUT_VEL == TRUE){
            np_V = (PyArrayObject *) PyArray_ZEROS(2, dims_S, NP_OUT_TYPE, 0);
            V = (fft_real *) np_V->data;          
        }
        else    
            V = NULL;
    }
    else if(IN_disp == FALSE){
        S = NULL;
        V = NULL;
    }

    /*Alloc the flag array*/
    if(OUT_FLAG == TRUE){
        np_flag = (PyArrayObject *) PyArray_ZEROS(3, dims_flag, PyArray_LONG, 0);
        flag = (long *) np_flag->data;
    }
    else{
        flag = (long *) malloc(box.ng*sizeof(long));
        check_memory(flag, "flag")
    }
    
    /*Initialize the flag array*/
	for(ind=0;ind<box.ng;ind++)
		flag[ind] = (long) box.ng;

    /*Generate the halo catalogue from the linear power spectrum*/
    nh = Generate_Halos_Box_from_Grid(K, P, Nk, k_smooth, &halos, &posh, &velh, flag, delta, S, V, IN_disp);
    if(OUT_FLAG == FALSE)
        free(flag);

    /*Alloc the output arrays*/
    npy_intp dims_posh[] = {(npy_intp) nh, (npy_intp) 3};
    npy_intp dims_Mh[] = {(npy_intp) nh};
    PyArrayObject *np_posh, *np_velh, *np_Mh;

    np_Mh = (PyArrayObject *) PyArray_ZEROS(1, dims_Mh, NP_OUT_TYPE, 0);
    np_posh = (PyArrayObject *) PyArray_ZEROS(2, dims_posh, NP_OUT_TYPE, 0);
    Mh = (fft_real *) np_Mh->data;
    posh_out = (fft_real *) np_posh->data;
    if(out.OUT_VEL == TRUE){
        np_velh = (PyArrayObject *) PyArray_ZEROS(2, dims_posh, NP_OUT_TYPE, 0);
        velh_out = (fft_real *) np_velh->data;       
    }

    /*Put the values in the output arrays*/
    for(i=0;i<nh;i++){
        Mh[i] = halos[i].Mh;
        for(j=0;j<3;j++){
            ind = (size_t) 3*i+j;
            posh_out[ind] = posh[ind];
            if(out.OUT_VEL == TRUE)
                velh_out[ind] = velh[ind];
        }
    }

    /*Free the original arrays with the halo information*/
    free(posh);
    free(halos);
    if(out.OUT_VEL == TRUE) free(velh);

    /*Construct the output tuple for each case*/
    PyObject *dict = PyDict_New();

    if(OUT_LPT == FALSE && OUT_VEL == FALSE && OUT_FLAG == FALSE){
        PyDict_SetItemString(dict, "posh", PyArray_Return(np_posh));
        PyDict_SetItemString(dict, "Mh", PyArray_Return(np_Mh));
    }
    else if(OUT_LPT == FALSE && OUT_VEL == TRUE && OUT_FLAG == FALSE){
        PyDict_SetItemString(dict, "posh", PyArray_Return(np_posh));
        PyDict_SetItemString(dict, "velh", PyArray_Return(np_velh));
        PyDict_SetItemString(dict, "Mh", PyArray_Return(np_Mh)); 
    }
    else if(OUT_LPT == TRUE && OUT_VEL == FALSE && OUT_FLAG == FALSE){
        PyDict_SetItemString(dict, "posh", PyArray_Return(np_posh));
        PyDict_SetItemString(dict, "Mh", PyArray_Return(np_Mh));
        PyDict_SetItemString(dict, "pos", PyArray_Return(np_S));
    }
    else if(OUT_LPT == TRUE && OUT_VEL == TRUE && OUT_FLAG == TRUE){
        PyDict_SetItemString(dict, "posh", PyArray_Return(np_posh));
        PyDict_SetItemString(dict, "velh", PyArray_Return(np_velh));
        PyDict_SetItemString(dict, "Mh", PyArray_Return(np_Mh));
        PyDict_SetItemString(dict, "pos", PyArray_Return(np_S));
        PyDict_SetItemString(dict, "vel", PyArray_Return(np_V));
        PyDict_SetItemString(dict, "flag", PyArray_Return(np_flag));
    }   
    else if(OUT_LPT == TRUE && OUT_VEL == FALSE && OUT_FLAG == TRUE){
        PyDict_SetItemString(dict, "posh", PyArray_Return(np_posh));
        PyDict_SetItemString(dict, "Mh", PyArray_Return(np_Mh));
        PyDict_SetItemString(dict, "pos", PyArray_Return(np_S));
        PyDict_SetItemString(dict, "flag", PyArray_Return(np_flag));
    } 

    return dict; 
}

/* This initiates the module using the above definitions. */
#if PY_VERSION_HEX >= 0x03000000
static struct PyModuleDef exshalos = {
   PyModuleDef_HEAD_INIT,
   "exshalos", NULL, -1, exshalos_methods
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
