from cmath import nan
import exshalos
import numpy as np 

#Generate an array of random numbers that follow a given pdf
def Generate_Random_Numbers(PDF, nps, rmin, rmax, NRs = 500, Neps = 500, Tot = 1e-2, seed = 12345, Inter_log = False, nthreads = 1, verbose = False):
    """
    PDF: Function (of one variable) with the PDF to be used to generate the random numbers | Python function 
    nps: Number of random points to be generated | int
    rmin: Minimum value for the random numbers | float
    rmax: Maximum value for the random numbers | float
    NRs: Number of bins to be used in the interpolation of the CDF | int
    Neps: Number of bins to be used in the interpolation of the r(eps) | int
    Tot: Tolerance when solving the CDF(r) = Eps equation using Newton's method | float
    Inter_log: Use log or linear binning for the interpolation of the CDF | boolean
    nthreads: Number of threads to be used | int
    verbose: Print or not some information | boolean

    return: nps random numbers generated between rmin and rmax following the PDF given | 1D numpy array
    """

    precision = exshalos.sampler.sampler.check_precision()

    rmin = np.float64(rmin)
    rmax = np.float64(rmax)
    Tot = np.float64(Tot)

    x = exshalos.sampler.sampler.random_generator(PDF, rmin, rmax, np.int(nps), np.int32(NRs), np.int32(Neps), Tot, np.int32(Inter_log), np.int32(seed), np.int32(nthreads), np.int32(verbose))

    return x

#Find the best fit and the covariance matrix of the parameters
def Fit(k, P_data, P_terms, cov, params0 = None, MODEL = "Pgg", minimizer = "GD", Tot = 1e-5, alpha = 1e-6, nthreads = 1, verbose = False):
    """
    k: Wavenumbers of the data and theory | 1D numpy array (Nk)
    P_data: Data to be fitted | 1D numpy array (Nk)
    P_terms: Data with the basis spectra for the given model and data | 2D numpy array (Nterms, Nk)
    cov: Covariance of the data to be used in the computation of the chi2 | 2D numpy array (Nk, Nk)
    params0: Initial position of the parameters for each walker | 2D numpy array (Nwalkers, Nparams)
    MODEL: Model used to describe the data | string
    Tot: Tolerance to be used to find the best fit parameters | float
    alpha: Factor used in the suppression of the gradients | float
    nthreads: Number of threads to be used | int
    verbose: Print or not some information | boolean

    return: Parameters in the best fit, theoretical function at this point and estimative of the covariance matrix at this point | python dictionary with the best fit ("params"), theoretical function at this point ("theory",) and covariance matrix ("Sigma")
    """   

    #Check the precision
    precision = exshalos.sampler.sampler.check_precision()

    #Compute the number of parameters
    Nparameters = int((np.sqrt(8*len(P_terms) + 1) - 1)/2)
    if(MODEL == "Pgg" or MODEL == 0):
        Nparameters += 1
        MODEL = 0
    elif(MODEL == "Pgm" or MODEL == "Pmg" or MODEL == 1):
        Nparameters += 0
        MODEL = 1

    #Set params0 if not given
    if(params0 is None):
        params0 = np.random.random(Nparameters)

    #Set the value of minimizer
    if(minimizer == "GD" or minimizer == "gd" or minimizer == 0):
        minimizer = 0
    elif(minimizer == "GN" or minimizer == "gn" or minimizer == 1):
        minimizer = 1
    else:
        raise ValueError("Invalide choice of the minimizer to be used")

    #Convert the floats to the correct precision
    k = np.float64(k)
    P_data = np.float64(P_data)
    P_terms = np.float64(P_terms)
    inv_cov = np.float64(np.linalg.inv(cov))
    params0 = np.float64(params0)
    Tot = np.float64(Tot)
    alpha = np.float64(alpha)

    #Find the best fit
    x = exshalos.sampler.sampler.minimize(k, P_data, P_terms, params0, inv_cov, np.int32(Nparameters), np.int32(MODEL), np.int32(minimizer), Tot, alpha, np.int32(nthreads), np.int32(verbose))

    return x

#Sample the posterior using HMC
def HMC(k, P_data, P_terms, cov, params_priors, params0 = None, Sigma0 = None, prior_type = "flat", Nwalkers = 8, Nsteps = 1000, MODEL = "Pgg", L = 10, eps = np.pi/10, seed = 12345, pre_warmup = False, minimizer = "GD", Tot = 1e-6, alpha = 1e-6, nthreads = 1, verbose = False):
    """
    k: Wavenumbers of the data and theory | 1D numpy array (Nk)
    P_data: Data to be fitted | 1D numpy array (Nk)
    P_terms: Data with the basis spectra for the given model and data | 2D numpy array (Nterms, Nk)
    cov: Covariance of the data to be used in the computation of the chi2 | 2D numpy array (Nk, Nk)
    params_priors: Priors to be used in the sampling | 2D numpy array (Nparams, 2)
    params0: Initial position of the parameters for each walker | 2D numpy array (Nwalkers, Nparams)
    Nwalkers: Number of independent chains | int
    Nsteps: Number of total points to be sampled for each walker | int
    MODEL: Model used to describe the data | string
    L: Number of steps to be used in the leap-froag integrator | int
    eps: Size of each step of the leap-froag integrator | float
    seed: Seed for the random generator | int
    pre_warmup: Do the pre warm-up or not | boolean
    minimizer: Method to be used in the minimization in the pre warm-up phase | string
    Tot: Initial tolerance in the pre warm-up | float
    alpha: Initial step used in the minimizer in the pre warm-up phase | float
    nthreads: Number of threads to be used | int
    verbose: Print or not some information | boolean

    return: Positions of each walker in the parameter space and the respective log posterior | python dictionary with the chains ("chains") and log posterior ("logP")
    """
  
    #Check the precision
    precision = exshalos.sampler.sampler.check_precision()

    #Compute the number of parameters
    Nparameters = int((np.sqrt(8*len(P_terms) + 1) - 1)/2)
    if(MODEL == "Pgg" or MODEL == 0):
        Nparameters += 1
        MODEL = 0
    elif(MODEL == "Pgm" or MODEL == "Pmg" or MODEL == 1):
        Nparameters += 0
        MODEL = 1
    
    #Set the inital position if not given
    if(params0 is None):
        if(prior_type == "flat" or prior_type == "Flat" or prior_type == "FLAT" or prior_type == 0):
            params0 = np.random.random([Nwalkers, Nparameters])
            for i in range(Nwalkers):
                params0[i,:] = params_priors[:,0] + (params_priors[:,1] - params_priors[:,0])*params0[i,:]
        elif(prior_type == "Gaussian" or prior_type == "gaussian" or prior_type == "GAUSSIAN" or prior_type == "Gauss" or prior_type == "gauss" or prior_type == "GAUSS" or prior_type == 1):
            params0 = np.random.normal(params_priors[:,0], params_priors[:,1], [Nwalkers, Nparameters])

    #Set the prior_type variable
    if(prior_type == "flat" or prior_type == "Flat" or prior_type == "FLAT" or prior_type == 0):
        prior_type = 0
    elif(prior_type == "Gaussian" or prior_type == "gaussian" or prior_type == "GAUSSIAN" or prior_type == "Gauss" or prior_type == "gauss" or prior_type == "GAUSS" or prior_type == 1):
        prior_type = 1
    else:
        raise ValueError("Invalide type for the prior! Try prior_type = 0 (flat prior) or prior_type = 1 (gaussian prior)")

    #Set the initial Sigma
    if(Sigma0 is None):
        Sigma0 = np.diag(np.ones(Nparameters))

    #Do the pre warm-up to find the best-fit and the estimation of the covariance matrix (Sigma)
    if(pre_warmup == True):
        chi2_best = np.inf
        for i in range(Nwalkers):
            pre = exshalos.samplers.Fit(k = k, P_data = P_data, P_terms = P_terms, cov = cov, params0 = params0[i,:], MODEL = MODEL, minimizer = minimizer, Tot = Tot, alpha = alpha, nthreads = nthreads, verbose = verbose)
            if(np.invert(np.isnan(pre["chi2"]))):
                params0[i,:] = pre["params"]
                if(pre["chi2"] < chi2_best):
                    chi2_best = pre["chi2"]
                    Sigma0 = pre["Sigma"]
            print(pre["chi2"])

    print(params0)
      
    #Convert the floats to the correct precision
    k = np.float64(k)
    P_data = np.float64(P_data)
    P_terms = np.float64(P_terms)
    inv_cov = np.float64(np.linalg.inv(cov))
    params_priors = np.float64(params_priors)
    eps = np.float64(eps)
    params0 = np.float64(params0)
    Sigma0 = np.float64(Sigma0)

    #Sample the posterior using HMC
    x = exshalos.sampler.sampler.hmc(k, P_data, P_terms, params0, inv_cov, params_priors, Sigma0, np.int32(Nwalkers), np.int32(Nsteps), np.int32(Nparameters), np.int32(MODEL), np.int32(prior_type), np.int32(L), eps, np.int32(seed), np.int32(nthreads), np.int32(verbose))

    return x

