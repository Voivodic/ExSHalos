#define HOD_MODULE

#include "hod_h.h"
#include "populate_halos.h"
#include "split_galaxies.h"

/*This declares the compute function*/
static PyObject *hod_check_precision(PyObject * self, PyObject * args);
static PyObject *populate_halos(PyObject *self, PyObject *args, PyObject *kwargs);
static PyObject *split_galaxies(PyObject *self, PyObject *args, PyObject *kwargs);

/*This tells Python what methods this module has. See the Python-C API for more information.*/
static PyMethodDef hod_methods[] = {
    {"check_precision", hod_check_precision, METH_VARARGS, "Returns precision used by the estimators of the spectra"},
    {"populate_halos", populate_halos, METH_VARARGS | METH_KEYWORDS, "Populate a list of halos with galaxies"},
    {"split_galaxies", split_galaxies, METH_VARARGS | METH_KEYWORDS, "Split the galaxies between red and blue"},
    {NULL, NULL, 0, NULL}
};

/*Return the precision used in the grid computations*/
static PyObject *hod_check_precision(PyObject * self, PyObject * args){
    return Py_BuildValue("i", sizeof(fft_real));
}

/*Populate the halos*/
static PyObject *populate_halos(PyObject *self, PyObject *args, PyObject *kwargs){
    size_t i, j, nh, ng;
    int verbose, ndx, ndy, ndz, OUT_VEL, IN_C, seed_in, OUT_FLAG;
    long *gal_type;
    fft_real Om0, redshift, Lc, *posh, *velh, *Mh, *Ch, *posg, *velg, logMmin, siglogM, logM0, logM1, alpha, Deltah, sigma;

	/*Define the list of parameters*/
	static char *kwlist[] = {"posh_array", "velh_array", "Mh_array", "Ch_array", "Lc", "Om0", "redshift", "ndx", "ndy", "ndz", "logMmin", "siglogM", "logM0", "logM1", "alpha", "sigma", "Deltah", "seed", "OUT_VEL", "OUT_FLAG", "IN_C", "verbose", NULL};
	import_array();

	/*Define the pyobject with the 3D position of the tracers*/
	PyArrayObject *posh_array, *velh_array, *Mh_array, *Ch_array;  

	/*Read the input arguments*/
	#ifdef DOUBLEPRECISION_FFTW
		if (!PyArg_ParseTupleAndKeywords(args, kwargs, "OOOOdddiiidddddddiiiii", kwlist, &posh_array, &velh_array, &Mh_array, &Ch_array, &Lc, &Om0, &redshift, &ndx, &ndy, &ndz, &logMmin, &siglogM, &logM0, &logM1, &alpha, &sigma, &Deltah, &seed_in, &OUT_VEL, &OUT_FLAG, &IN_C, &verbose))
			return NULL;
	#else
		if (!PyArg_ParseTupleAndKeywords(args, kwargs, "OOOOfffiiifffffffiiiii", kwlist, &posh_array, &velh_array, &Mh_array, &Ch_array, &Lc, &Om0, &redshift, &ndx, &ndy, &ndz, &logMmin, &siglogM, &logM0, &logM1, &alpha, &sigma, &Deltah, &seed_in, &OUT_VEL, &OUT_FLAG, &IN_C, &verbose))
			return NULL;
	#endif

    /*Convert PyObjects to C arrays*/
    nh = (size_t) Mh_array->dimensions[0];
    posh = (fft_real *) posh_array->data;
    if(OUT_VEL == TRUE)
        velh = (fft_real *) velh_array->data;
    Mh = (fft_real *) Mh_array->data;
    if(IN_C == TRUE)
        Ch = (fft_real *) Ch_array->data;
    else
        Ch = (fft_real *) malloc(nh*sizeof(fft_real));

    /*Set the box structure*/
    set_cosmology(Om0, redshift, -1.0);  
    set_box(ndx, ndy, ndz, Lc);
    set_hod(logMmin, siglogM, logM0, logM1, alpha, sigma);
    set_out((char) OUT_VEL, 1, (char) IN_C, (char) verbose);
    if(Deltah > 0.0)
        cosmo.Dv = Deltah;
    cosmo.Mstar = 4.638211e+12;     //CHANGE IT!!!
    seed = seed_in;

    /*Initialize the randonm seed*/
    gsl_rng *rng_ptr;
    rng_ptr = gsl_rng_alloc (gsl_rng_taus);
    gsl_rng_set(rng_ptr, seed);

    /*Alloc the arrays for the galaxies*/
    posg = (fft_real *) malloc(Ng_max*nh*3*sizeof(fft_real));
    if(OUT_VEL == TRUE)
        velg = (fft_real *) malloc(Ng_max*nh*3*sizeof(fft_real));
    gal_type = (long *) malloc(Ng_max*nh*sizeof(long));

    /*Populate the halos*/
    ng = Populate_total(nh, posh, velh, Mh, Ch, posg, velg, gal_type, rng_ptr);
    if(OUT_FLAG == FALSE)
        free(gal_type);

    /*Define the variables for the output*/
    npy_intp dims_pos[] = {(npy_intp) ng, (npy_intp) 3};
    npy_intp dims_flag[] = {(npy_intp) ng};
    fft_real *posg_out, *velg_out;
    long *flag_out;
    PyArrayObject *np_pos, *np_vel, *np_flag;
    PyObject *tupleresult;

    np_pos = (PyArrayObject *) PyArray_ZEROS(2, dims_pos, NP_OUT_TYPE, 0);
    posg_out = (fft_real *) np_pos->data;
    if(OUT_VEL == TRUE){
        np_vel = (PyArrayObject *) PyArray_ZEROS(2, dims_pos, NP_OUT_TYPE, 0);
        velg_out = (fft_real *) np_vel->data;
    }
    if(OUT_FLAG == TRUE){
        np_flag = (PyArrayObject *) PyArray_ZEROS(1, dims_flag, PyArray_LONG, 0);
        flag_out = (long *) np_flag->data;
    }

    /*Put the positions and velocities in the output arrays*/
    for(i=0;i<ng;i++){
        for(j=0;j<3;j++){
            posg_out[3*i+j] = posg[3*i+j];
            if(OUT_VEL == TRUE)
                velg_out[3*i+j] = velg[3*i+j];
        }
        if(OUT_FLAG == TRUE)
            flag_out[i] = gal_type[i];
    }
    free(posg);
    if(OUT_VEL == TRUE)
        free(velg);
    if(OUT_FLAG == TRUE)
        free(gal_type);

    /*Output the mesurements in PyObject format*/
    if(OUT_VEL == TRUE){
        if(OUT_FLAG == FALSE){
            tupleresult = PyTuple_New(2);
            PyTuple_SetItem(tupleresult, 0, PyArray_Return(np_pos));    
            PyTuple_SetItem(tupleresult, 1, PyArray_Return(np_vel));  
        }
        else{
            tupleresult = PyTuple_New(3);
            PyTuple_SetItem(tupleresult, 0, PyArray_Return(np_pos));    
            PyTuple_SetItem(tupleresult, 1, PyArray_Return(np_vel));        
            PyTuple_SetItem(tupleresult, 2, PyArray_Return(np_flag));        
        }
    }
    else{
        if(OUT_FLAG == FALSE){
            tupleresult = PyTuple_New(1);
            PyTuple_SetItem(tupleresult, 0, PyArray_Return(np_pos));  
        }
        else{
            tupleresult = PyTuple_New(2);
            PyTuple_SetItem(tupleresult, 0, PyArray_Return(np_pos));    
            PyTuple_SetItem(tupleresult, 1, PyArray_Return(np_flag));        
        }  
    }

    return PyArray_Return((PyArrayObject*) tupleresult); 
}

/*Split the galaxies between red and blue*/
static PyObject *split_galaxies(PyObject *self, PyObject *args, PyObject *kwargs){
    size_t ng;
    int verbose, seed_in, *type;
    long *flag;
    fft_real *Mh, C3, C2, C1, C0, S3, S2, S1, S0;

	/*Define the list of parameters*/
	static char *kwlist[] = {"Mh_array", "Flag_array", "seed", "verbose", NULL};
	import_array();

	/*Define the pyobject with the 3D position of the tracers*/
	PyArrayObject *Mh_array, *Flag_array;  

	/*Read the input arguments*/
	#ifdef DOUBLEPRECISION_FFTW
		if (!PyArg_ParseTupleAndKeywords(args, kwargs, "OOddddddddii", kwlist, &Mh_array, &Flag_array, &C3, &C2, &C1, &C0, &S3, &S2, &S1, &S0, &seed_in, &verbose))
			return NULL;
	#else
		if (!PyArg_ParseTupleAndKeywords(args, kwargs, "OOffffffffii", kwlist, &Mh_array, &Flag_array, &C3, &C2, &C1, &C0, &S3, &S2, &S1, &S0, &seed_in, &verbose))
			return NULL;
	#endif

    /*Convert PyObjects to C arrays*/
    ng = (size_t) Flag_array->dimensions[0];
    Mh = (fft_real *) Mh_array->data;
    flag = (long *) Flag_array->data;
    seed = seed_in;

    /*Set the free parameters used to do the separation*/
    set_split(C3, C2, C1, C0, S3, S2, S1, S0);

    /*Initialize the randonm seed*/
    gsl_rng *rng_ptr;
    rng_ptr = gsl_rng_alloc (gsl_rng_taus);
    gsl_rng_set(rng_ptr, seed);

    /*Define the variables for the output*/
    npy_intp dims_type[] = {(npy_intp) ng};
    PyArrayObject *np_type;

    np_type = (PyArrayObject *) PyArray_ZEROS(1, dims_type, PyArray_INT, 0);
    type = (int *) np_type->data;

    /*Compute the type of each galaxy*/
    Galaxy_Types(ng, Mh, flag, type, rng_ptr);

	/*Returns the type of each galaxy*/
	return PyArray_Return(np_type);
}

/* This initiates the module using the above definitions. */
#if PY_VERSION_HEX >= 0x03000000
static struct PyModuleDef hod = {
   PyModuleDef_HEAD_INIT,
   "hod", NULL, -1, hod_methods
};

PyMODINIT_FUNC PyInit_hod(void)
{
    PyObject *m;
    m = PyModule_Create(&hod);
    if (!m) {
        return NULL;
    }
    return m;
}
#else
PyMODINIT_FUNC inithod(void)
{
    PyObject *m = Py_InitModule("hod", hod_methods);
    import_array();
    if (!m) {
        return NULL;
    }
}
#endif
