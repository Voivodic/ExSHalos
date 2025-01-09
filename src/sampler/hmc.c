#include "hmc.h"

/*Compute one leap-frog step integration*/
double Leap_Frog(double *k, void (*Model)(double *, double *, double **, double *, double *, int, int), double *P_theory, double **dP_theory, double *P_data, double *inv_cov, double *params, double *p, double *P_terms, double *dchi2, int Nk, int NO, int Nparams, double eps, int L, double *params_priors, double *invMass){
    int i, j, l;
    double chi2;

    /*First eps/2 step in the momenta*/
    chi2 = log_like(k, Model, P_theory, dP_theory, P_data, inv_cov, params, P_terms, dchi2, NULL, Nk, NO, Nparams, FALSE);
    for(j=0;j<Nparams;j++)
        p[j] += 0.5*eps*dchi2[j];

    /*Run the L steps of size eps*/
    for(i=0;i<L;i++){
        /*Eps step in the positions*/
        for(j=0;j<Nparams;j++){
            for(l=0;l<Nparams;l++)
                params[j] += eps*invMass[j*Nparams + l]*p[l];

            if(Prior_type == 0 && (params[j] < params_priors[2*j])){
                params[j] = 2.0*params_priors[2*j] - params[j];
                p[j] = -p[j];
            }
            else if(Prior_type == 0 && (params[j] > params_priors[2*j+1])){
                params[j] = 2.0*params_priors[2*j+1] - params[j];
                p[j] = -p[j];                
            }
        }

        /*Eps step in the momenta*/
        chi2 = log_like(k, Model, P_theory, dP_theory, P_data, inv_cov, params, P_terms, dchi2, NULL, Nk, NO, Nparams, FALSE);
        for(j=0;j<Nparams;j++)
            p[j] += eps*dchi2[j];
    }

    /*Go back eps/2 in the momenta*/
    for(j=0;j<Nparams;j++)
        p[j] -= 0.5*eps*dchi2[j];

    /*Return the final chi2*/
    return chi2;
}

/*Sample a given posterior with HMC*/
void Sample_HMC(double *k_data, double *P_data, double *inv_cov, double *params0, double *params_priors, double *P_terms, int Nk, int NO, int Nparams, double eps, int L, fft_real *chains, double *log_P, int Nsteps, int Nwalkers, double *mu, double *Sigma, double *invMass){
    int a, b, c, i, j, *acceptance;
    size_t ind;
    double *params, *p, *P_theory, **dP_theory, chi2, *dchi2, Hini, H, R;
    void (*Model)(double *, double *, double **, double *, double *, int, int);

    /*Alloc the array for the number of acceptances*/
    acceptance = (int *)malloc(Nwalkers*sizeof(int));
    for(i=0;i<Nwalkers;i++)
        acceptance[i] = 1;

    /*Alloc the arrays for the current params and momenta*/
    params = (double *)malloc(Nparams*sizeof(double));
    p = (double *)malloc(Nparams*sizeof(double));
    dchi2 = (double *)malloc(Nparams*sizeof(double));

    /*Alloc the arrays for the theoretical prediction and its derivative*/
    P_theory = (double *)malloc(Nk*sizeof(double));
    dP_theory = (double **)malloc(Nparams*sizeof(double *));
    for(a=0;a<Nparams;a++)
        dP_theory[a] = (double *)malloc(Nk*sizeof(double));

    /*Define the inverse of the mass matrix using GSL*/
    gsl_matrix *Mass;
    Mass = gsl_matrix_alloc(Nparams, Nparams);
    for(i=0;i<Nparams;i++)
        for(j=0;j<Nparams;j++)
            gsl_matrix_set(Mass, i, j, invMass[i*Nparams + j]);
    Mass = invert_matrix(Mass, Nparams);
    gsl_linalg_cholesky_decomp(Mass);

    /*Define the GSL arrays used in the generation of the random number*/
    gsl_vector *mean, *result;
    mean = gsl_vector_alloc(Nparams);
    result = gsl_vector_alloc(Nparams);
    gsl_vector_set_zero(mean);

    /*Alloc the arrays for the mean and covariance of the parameters for the last chunck*/
    for(a=0;a<Nwalkers;a++)
        for(b=0;b<Nparams;b++){
            mu[a*Nparams + b] = 0.0;
            for(c=0;c<Nparams;c++)
                Sigma[(a*Nparams + b)*Nparams + c] = 0.0;
    }

    /*Define the function to be used in the theoretical computations*/
    switch(MODEL){
        case 0: 
            Model = &Pgg_theory;
            break;
        case 1: 
            Model = &Pgm_theory;
            break;
    }

	/*Alloc the needed quantities for the random generator*/
	gsl_rng *rng_ptr;
	rng_ptr = gsl_rng_alloc(gsl_rng_taus);
	gsl_rng_set(rng_ptr, seed);

    /*Run for the Nwalkers walkers*/
    for(b=0;b<Nwalkers;b++){

        /*Choose the initial point*/
        gsl_ran_multivariate_gaussian(rng_ptr, mean, Mass, result);
        for(i=0;i<Nparams;i++){
            chains[((size_t) b)*Nsteps*Nparams + i] = (fft_real) params0[b*Nparams + i];
            params[i] = params0[b*Nparams + i];
            p[i] = (double) gsl_vector_get(result, i);
        }

        /*Compute the initial chi2*/
        chi2 = log_like(k_data, Model, P_theory, dP_theory, P_data, inv_cov, params, P_terms, dchi2, NULL, Nk, NO, Nparams, FALSE);
        log_P[(size_t) b*Nsteps] = chi2;

        /*Compute Nsteps new points for the chains*/
        for(c=1;c<Nsteps;c++){
            /*Compute the initial hamiltonian*/
            Hini = -chi2;
            for(i=0;i<Nparams;i++)
                for(j=0;j<Nparams;j++)
                    Hini += p[i]*p[j]/2.0*invMass[i*Nparams + j];
            if(Prior_type == 1)
                Hini += -log_prior_gauss(params, params_priors, Nparams);

            /*Evolve the system using leap-frog and compute the final hamiltonian*/
            chi2 = Leap_Frog(k_data, Model, P_theory, dP_theory, P_data, inv_cov, params, p, P_terms, dchi2, Nk, NO, Nparams, eps, L, params_priors, invMass);
            H = -chi2;
            for(i=0;i<Nparams;i++)
                for(j=0;j<Nparams;j++)
                    Hini += p[i]*p[j]/2.0*invMass[i*Nparams + j];
            if(Prior_type == 1)
                Hini += -log_prior_gauss(params, params_priors, Nparams);

            /*Add the new params to the chain or keep the last one*/
            R = exp(Hini - H);
            ind = ((size_t) b*Nsteps + c)*Nparams;
            if(R >= 1.0){
                for(i=0;i<Nparams;i++)
                    chains[ind + i] = (fft_real) params[i];
                log_P[(size_t) b*Nsteps + c] = chi2;
                acceptance[b] ++;
            }
            else if(R > gsl_rng_uniform(rng_ptr)){
                for(i=0;i<Nparams;i++)
                    chains[ind + i] = (fft_real) params[i]; 
                log_P[(size_t) b*Nsteps + c] = chi2;
                acceptance[b] ++;
            }              
            else{
                for(i=0;i<Nparams;i++){
                    chains[ind + i] = chains[ind - Nparams + i]; 
                    params[i] = (double) chains[ind + i];
                }
                log_P[(size_t) b*Nsteps + c] = log_P[(size_t) b*Nsteps + c - 1];
                chi2 = log_P[(size_t) b*Nsteps + c - 1];
            }

            /*Define the mometum of the next run*/
            gsl_ran_multivariate_gaussian(rng_ptr, mean, Mass, result);
            for(i=0;i<Nparams;i++)
                p[i] = (double) gsl_vector_get(result, i);

            /*Compute the contribution to the mean and covariance of the parameters*/
            for(i=0;i<Nparams;i++){
                mu[b*Nparams + i] += params[i];
                for(j=0;j<Nparams;j++)
                    Sigma[(b*Nparams + i)*Nparams + j] += params[i]*params[j];
            }
        }
    }

    for(i=0;i<Nwalkers;i++)
        printf("%d - %d of %d\n", i, acceptance[i], Nsteps);

    /*Compute the mean and the covariance of the parameters*/
    for(a=0;a<Nwalkers;a++){
        for(i=0;i<Nparams;i++)
            mu[a*Nparams + i] = mu[a*Nparams + i]/Nsteps;
        for(i=0;i<Nparams;i++)
            for(j=0;j<Nparams;j++)
                Sigma[(a*Nparams + i)*Nparams + j] = Sigma[(a*Nparams + i)*Nparams + j]/(Nsteps - 1) - (Nsteps)/(Nsteps - 1)*mu[a*Nparams + i]*mu[a*Nparams + j];
    }
   
    /*Free the arrays*/
    free(params);
    free(p);
    free(P_theory);
    for(a=0;a<Nparams;a++)
        free(dP_theory[a]);
    free(dP_theory);
    gsl_vector_free(mean);
    gsl_vector_free(result);
    gsl_matrix_free(Mass);
}