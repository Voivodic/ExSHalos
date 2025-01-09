#define HOD_MODULE

/*Import the headers with the functions of this module*/
#include "hod_h.h"
#include "populate_halos.h"
#include "split_galaxies.h"

/*Impoert the python and numpy APis*/
#include <Python.h>
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>

/*This declares the compute function*/
static PyObject *hod_check_precision(PyObject * self, PyObject * args);
static PyObject *populate_halos(PyObject *self, PyObject *args, PyObject *kwargs);
static PyObject *split_galaxies(PyObject *self, PyObject *args, PyObject *kwargs);

/*This tells Python what methods this module has. See the Python-C API for more information.*/
static PyMethodDef hod_methods[] = {
    {"check_precision", (PyCFunction)hod_check_precision, METH_VARARGS, "Returns precision used by the estimators of the spectra"},
    {"populate_halos", (PyCFunction)populate_halos, METH_VARARGS | METH_KEYWORDS, "Populate a list of halos with galaxies"},
    {"split_galaxies", (PyCFunction)split_galaxies, METH_VARARGS | METH_KEYWORDS, "Split the galaxies between red and blue"},
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
    nh = (size_t) PyArray_DIMS(Mh_array)[0];
    posh = (fft_real *) PyArray_DATA(posh_array);
    if(OUT_VEL == TRUE)
        velh = (fft_real *) PyArray_DATA(velh_array);
    Mh = (fft_real *) PyArray_DATA(Mh_array);
    if(IN_C == TRUE)
        Ch = (fft_real *) PyArray_DATA(Ch_array);
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

    np_pos = (PyArrayObject *) PyArray_ZEROS(2, dims_pos, NP_OUT_TYPE, 0);
    posg_out = (fft_real *) PyArray_DATA(np_pos);
    if(OUT_VEL == TRUE){
        np_vel = (PyArrayObject *) PyArray_ZEROS(2, dims_pos, NP_OUT_TYPE, 0);
        velg_out = (fft_real *) PyArray_DATA(np_vel);
    }
    if(OUT_FLAG == TRUE){
        np_flag = (PyArrayObject *) PyArray_ZEROS(1, dims_flag, NPY_LONG, 0);
        flag_out = (long *) PyArray_DATA(np_flag);
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

    /*Construct the output tuple for each case*/
    PyObject *dict = PyDict_New();

    /*Output the mesurements in PyObject format*/
    if(OUT_VEL == TRUE){
        if(OUT_FLAG == FALSE){
            PyDict_SetItemString(dict, "posg", PyArray_Return(np_pos));
            PyDict_SetItemString(dict, "velg", PyArray_Return(np_vel));
        }
        else{
            PyDict_SetItemString(dict, "posg", PyArray_Return(np_pos));
            PyDict_SetItemString(dict, "velg", PyArray_Return(np_vel));
            PyDict_SetItemString(dict, "flag", PyArray_Return(np_flag));      
        }
    }
    else{
        if(OUT_FLAG == FALSE){
            PyDict_SetItemString(dict, "posg", PyArray_Return(np_pos));
        }
        else{
            PyDict_SetItemString(dict, "posg", PyArray_Return(np_pos));
            PyDict_SetItemString(dict, "flag", PyArray_Return(np_flag));        
        }  
    }

    return dict; 
}

/*Split the galaxies between red and blue*/
static PyObject *split_galaxies(PyObject *self, PyObject *args, PyObject *kwargs){
    size_t ng;
    int verbose, seed_in, *type, ntypes, order_cen, order_sat;
    long *flag;
    fft_real *Mh, *params_cen, *params_sat;

	/*Define the list of parameters*/
	static char *kwlist[] = {"Mh_array", "Flag_array", "params_cen_array", "params_sat_array", "seed", "verbose", NULL};
	import_array();

	/*Define the pyobject with the 3D position of the tracers*/
	PyArrayObject *Mh_array, *Flag_array, *params_cen_array, *params_sat_array;  

	/*Read the input arguments*/
	#ifdef DOUBLEPRECISION_FFTW
		if (!PyArg_ParseTupleAndKeywords(args, kwargs, "OOOOii", kwlist, &Mh_array, &Flag_array, &params_cen_array, &params_sat_array, &seed_in, &verbose))
			return NULL;
	#else
		if (!PyArg_ParseTupleAndKeywords(args, kwargs, "OOOOii", kwlist, &Mh_array, &Flag_array, &params_cen_array, &params_sat_array, &seed_in, &verbose))
			return NULL;
	#endif

    /*Convert PyObjects to C arrays*/
    ng = (size_t) PyArray_DIMS(Flag_array)[0];
    Mh = (fft_real *) PyArray_DATA(Mh_array);
    flag = (long *) PyArray_DATA(Flag_array);
    params_cen = (fft_real *) PyArray_DATA(params_cen_array);
    params_sat = (fft_real *) PyArray_DATA(params_sat_array);
    seed = seed_in;
    if(PyArray_DIMS(params_cen_array)[0] == PyArray_DIMS(params_sat_array)[0])
        ntypes = PyArray_DIMS(params_cen_array)[0] + 1;
    else{
        printf("The number of types for centrals and satellites are different! %d != %d!\n", PyArray_DIMS(params_cen_array)[0],  PyArray_DIMS(params_sat_array)[0]);
        exit(0);
    }
    order_cen = PyArray_DIMS(params_cen_array)[1];
    order_sat = PyArray_DIMS(params_sat_array)[1];

    /*Set the free parameters used to do the separation*/
    set_split(params_cen, params_sat, ntypes, order_cen, order_sat);

    /*Initialize the randonm seed*/
    gsl_rng *rng_ptr;
    rng_ptr = gsl_rng_alloc (gsl_rng_taus);
    gsl_rng_set(rng_ptr, seed);

    /*Define the variables for the output*/
    npy_intp dims_type[] = {(npy_intp) ng};
    PyArrayObject *np_type;

    np_type = (PyArrayObject *) PyArray_ZEROS(1, dims_type, NPY_INT, 0);
    type = (int *) PyArray_DATA(np_type);

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
