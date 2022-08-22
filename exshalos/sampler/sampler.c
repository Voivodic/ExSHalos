#define SAMPLER_MODULE

#include "sampler_h.h"
#include "random_generator.h"
#include "hmc.h"
#include "minimization.h"

/*This declares the compute function*/
static PyObject *sampler_check_precision(PyObject * self, PyObject * args);
static PyObject *random_generator(PyObject *self, PyObject *args, PyObject *kwargs);
static PyObject *hmc(PyObject *self, PyObject *args, PyObject *kwargs);
static PyObject *minimize(PyObject *self, PyObject *args, PyObject *kwargs);

/*This tells Python what methods this module has. See the Python-C API for more information.*/
static PyMethodDef sampler_methods[] = {
    {"check_precision", sampler_check_precision, METH_VARARGS, "Returns precision used by the functions in sampler"},
    {"random_generator", random_generator, METH_VARARGS | METH_KEYWORDS, "Generates an array of random numbers following a given PDF"},
    {"hmc", hmc, METH_VARARGS | METH_KEYWORDS, "Samples the posterior using HMC"},
    {"minimize", minimize, METH_VARARGS | METH_KEYWORDS, "Find the best fit and the covariance of the parameters"},
    {NULL, NULL, 0, NULL}
};

/*Return the precision used in the grid computations*/
static PyObject *sampler_check_precision(PyObject * self, PyObject * args){
    return Py_BuildValue("i", sizeof(fft_real));
}

/*Generates an array of random numbers following a given PDF*/
static PyObject *random_generator(PyObject *self, PyObject *args, PyObject *kwargs){
    int i, verbose, Inter_log, NRs, Neps, nthreads;
    unsigned long long nps;
    double *r, *rho, rmin, rmax, Tot;

	/*Define the list of parameters*/
	static char *kwlist[] = {"PDF", "rmin", "rmax", "nps", "NRs", "Neps", "Tot", "Inter_log", "seed", "nthreads", "verbose", NULL};
	import_array();

	/*Define the pyobject with the 3D position of the tracers*/
	PyObject *PDF;  

	/*Read the input arguments*/
	if (!PyArg_ParseTupleAndKeywords(args, kwargs, "OffKiidiiii", kwlist, &PDF, &rmin, &rmax, &nps, &NRs, &Neps, &Tot, &Inter_log, &seed, &nthreads, &verbose))
		return NULL;

    /*Initialize openmp to run in parallel*/
    omp_set_num_threads(nthreads);

    /*Initialize the randonm seed*/
    gsl_rng *rng_ptr;
    rng_ptr = gsl_rng_alloc (gsl_rng_taus);
    gsl_rng_set(rng_ptr, seed);

    /*Define the variables for the output*/
    npy_intp dims_rng[] = {(npy_intp) nps};
    fft_real *rngc;

    PyArrayObject *np_rng = (PyArrayObject *) PyArray_ZEROS(1, dims_rng, NP_OUT_TYPE, 0);
    rngc = (fft_real *) np_rng->data;

    /*Alloc the arrays for the interpolation*/
    r = (double *) malloc(NRs*sizeof(double));
    check_memory(r, "r")
    rho = (double *) malloc(NRs*sizeof(double));
    check_memory(rho, "rho")
   	gsl_spline *spline_rho;
    spline_rho = gsl_spline_alloc(gsl_interp_cspline, NRs);

    /*Contruct the array of rs where to compute the CDF and interpolate it*/
    if(Inter_log == TRUE)
        for(i=0;i<NRs;i++)
            r[i] = pow(10.0, log10(rmin) + (log10(rmax) - log10(rmin))*i/(NRs - 1));
    else
        for(i=0;i<NRs;i++)
            r[i] = rmin + (rmax - rmin)*i/(NRs - 1);

    /*Compute the PDF as function of r and interpolate it*/
    for(i=0;i<NRs;i++)
        rho[i] = PyFloat_AsDouble(PyObject_CallFunction(PDF, "d", r[i]));
    gsl_spline_init(spline_rho, r, rho, NRs);

    /*Free the arrays*/
    free(r);
    free(rho);

    /*Generate the array of random points*/
    Generate_Random_Array(spline_rho, nps, rmin, rmax, rngc, rng_ptr, Inter_log, NRs, Neps, Tot);
    gsl_spline_free(spline_rho);	

    /*Return the array with random numbers*/
    return PyArray_Return(np_rng);
}

/*Samples the posterior using HMC*/
static PyObject *hmc(PyObject *self, PyObject *args, PyObject *kwargs){
    int L, Nsteps, Nwalkers, Nparams, NO, Nk, nthreads, verbose;
    double *k, *P_data, *P_terms, *params0, *inv_cov, *params_priors, *invmass, eps;  

	/*Define the list of parameters*/
	static char *kwlist[] = {"k", "P_data", "P_terms", "params0", "inv_cov", "params_priors", "invmass", "Nwalkers", "Nsteps", "Nparam", "MODEL", "Prior_type", "L", "eps", "seed", "nthreads", "verbose", NULL};
	import_array();

	/*Define the pyobject with the 3D position of the tracers*/
	PyArrayObject *k_array, *P_data_array, *P_terms_array, *params0_array, *inv_cov_array, *params_priors_array, *invmass_array;  

	/*Read the input arguments*/
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "OOOOOOOiiiiiidiii", kwlist, &k_array, &P_data_array, &P_terms_array, &params0_array, &inv_cov_array, &params_priors_array, &invmass_array, &Nwalkers, &Nsteps, &Nparams, &MODEL, &Prior_type, &L, &eps, &seed, &nthreads, &verbose))
        return NULL;

    /*Initialize openmp to run in parallel*/
    omp_set_num_threads(nthreads);

    /*Initialize the randonm seed*/
    gsl_rng *rng_ptr;
    rng_ptr = gsl_rng_alloc (gsl_rng_taus);
    gsl_rng_set(rng_ptr, seed);

    /*Convert to C array and get some additional data*/
    Nk = (int) k_array->dimensions[0];
    NO = (int) P_terms_array->dimensions[0];
    NO = (int) (sqrt(NO*8 + 1) - 1.0)/2.0;
    k = (double *) k_array->data;
    P_data = (double *) P_data_array->data;
    P_terms = (double *) P_terms_array->data;
    params0 = (double *) params0_array->data;
    inv_cov = (double *) inv_cov_array->data;
    params_priors = (double *) params_priors_array->data;
    invmass = (double *) invmass_array->data;

    /*Alloc the output arrays*/
    npy_intp dims_chains[] = {(npy_intp) Nwalkers, (npy_intp) Nsteps, (npy_intp) Nparams};
    npy_intp dims_logP[] = {(npy_intp) Nwalkers, (npy_intp) Nsteps};
    npy_intp dims_mu[] = {(npy_intp) Nwalkers, (npy_intp) Nparams};
    npy_intp dims_Sigma[] = {(npy_intp) Nwalkers, (npy_intp) Nparams, (npy_intp) Nparams};
    fft_real *chains;
    double *Sigma, *logP, *mu;

    PyArrayObject *np_chains = (PyArrayObject *) PyArray_ZEROS(3, dims_chains, NP_OUT_TYPE, 0);
    PyArrayObject *np_logP = (PyArrayObject *) PyArray_ZEROS(2, dims_logP, PyArray_FLOAT64, 0);
    PyArrayObject *np_mu = (PyArrayObject *) PyArray_ZEROS(2, dims_mu, PyArray_FLOAT64, 0);
    PyArrayObject *np_Sigma = (PyArrayObject *) PyArray_ZEROS(3, dims_Sigma, PyArray_FLOAT64, 0);
    chains = (fft_real *) np_chains->data;   
    logP = (double *) np_logP->data; 
    mu = (double *) np_mu->data;
    Sigma = (double *) np_Sigma->data;

    /*Sample the posterior using HMC*/
    Sample_HMC(k, P_data, inv_cov, params0, params_priors, P_terms, Nk, NO, Nparams, eps, L, chains, logP, Nsteps, Nwalkers, mu, Sigma, invmass);

    /*Create the dictionary for output*/
    PyObject *dict = PyDict_New();
    PyDict_SetItemString(dict, "chains", PyArray_Return(np_chains));
    PyDict_SetItemString(dict, "logP", PyArray_Return(np_logP));
    PyDict_SetItemString(dict, "mu", PyArray_Return(np_mu));
    PyDict_SetItemString(dict, "Sigma", PyArray_Return(np_Sigma));

    return dict;
}

/*Find the best fit and the covariance of the parameters*/
static PyObject *minimize(PyObject *self, PyObject *args, PyObject *kwargs){
    int i, Nparams, Nk, nthreads, verbose, NO, minimizer;
    double *k, *P_data, *P_terms, *params0, *inv_cov, Tot, chi2, alpha;  

	/*Define the list of parameters*/
	static char *kwlist[] = {"k", "P_data", "P_terms", "params0", "inv_cov", "Nparam", "MODEL", "minimizer", "Tolerance", "alpha", "nthreads", "verbose", NULL};
	import_array();

	/*Define the pyobject with the 3D position of the tracers*/
	PyArrayObject *k_array, *P_data_array, *P_terms_array, *params0_array, *inv_cov_array;  

	/*Read the input arguments*/
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "OOOOOiiiddii", kwlist, &k_array, &P_data_array, &P_terms_array, &params0_array, &inv_cov_array, &Nparams, &MODEL, &minimizer, &Tot, &alpha, &nthreads, &verbose))
        return NULL;

    /*Initialize openmp to run in parallel*/
    omp_set_num_threads(nthreads);

    /*Convert to C array and get some additional data*/
    Nk = (int) k_array->dimensions[0];
    NO = (int) P_terms_array->dimensions[0];
    NO = (int) (sqrt(NO*8 + 1) - 1.0)/2.0;
    k = (double *) k_array->data;
    P_data = (double *) P_data_array->data;
    P_terms = (double *) P_terms_array->data;
    params0 = (double *) params0_array->data;
    inv_cov = (double *) inv_cov_array->data;

    /*Alloc the output arrays*/
    npy_intp dims_params[] = {(npy_intp) Nparams};
    npy_intp dims_theory[] = {(npy_intp) Nk};
    npy_intp dims_Sigma[] = {(npy_intp) Nparams, (npy_intp) Nparams};
    double *Sigma, *params, *theory;

    PyArrayObject *np_params = (PyArrayObject *) PyArray_ZEROS(1, dims_params, PyArray_FLOAT64, 0);
    PyArrayObject *np_theory = (PyArrayObject *) PyArray_ZEROS(1, dims_theory, PyArray_FLOAT64, 0);
    PyArrayObject *np_Sigma = (PyArrayObject *) PyArray_ZEROS(2, dims_Sigma, PyArray_FLOAT64, 0);
    params = (double *) np_params->data;   
    theory = (double *) np_theory->data;
    Sigma = (double *) np_Sigma->data;

    /*Set the input parameters*/
    for(i=0;i<Nparams;i++)
        params[i] = params0[i];

    if(minimizer == 0)
        /*Find the best fit and its Hessian using the method of Gradient Descent*/
        chi2 = Gradient_Descent(k, theory, P_data, inv_cov, params, Sigma, P_terms, Nk, NO, Nparams, Tot, alpha);
    if(minimizer == 1)
        /*Find the best fit and its Hessian using the method of Gauss-Newton*/
        chi2 = Gauss_Newton(k, theory, P_data, inv_cov, params, Sigma, P_terms, Nk, NO, Nparams, Tot, alpha);

    /*Create the dictionary for output*/
    PyObject *dict = PyDict_New();
    PyDict_SetItemString(dict, "params", PyArray_Return(np_params));
    PyDict_SetItemString(dict, "theory", PyArray_Return(np_theory));
    PyDict_SetItemString(dict, "Sigma", PyArray_Return(np_Sigma));
    PyDict_SetItemString(dict, "chi2", PyFloat_FromDouble(-2.0*chi2));

    return dict;
}

/* This initiates the module using the above definitions. */
#if PY_VERSION_HEX >= 0x03000000
static struct PyModuleDef sampler = {
   PyModuleDef_HEAD_INIT,
   "sampler", NULL, -1, sampler_methods
};

PyMODINIT_FUNC PyInit_sampler(void)
{
    PyObject *m;
    m = PyModule_Create(&sampler);
    if (!m) {
        return NULL;
    }
    return m;
}
#else
PyMODINIT_FUNC initsampler(void)
{
    PyObject *m = Py_InitModule("sampler", sampler_methods);
    import_array();
    if (!m) {
        return NULL;
    }
}
#endif
