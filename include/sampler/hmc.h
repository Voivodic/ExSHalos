#ifndef HMC_H
#define HMC_H

#include "sampler_h.h"
#include "posteriors.h"

/*Compute one leap-frog step integration*/
double Leap_Frog(double *k, void (*Model)(double *, double *, double **, double *, double *, int, int), double *P_theory, double **dP_theory, double *P_data, double *inv_cov, double *params, double *p, double *P_terms, double *dchi2, int Nk, int NO, int Nparams, double eps, int L, double *params_priors, double *invMass);

/*Sample a given posterior with HMC*/
void Sample_HMC(double *k_data, double *P_data, double *inv_cov, double *params0, double *params_priors, double *P_terms, int Nk, int NO, int Nparams, double eps, int L, fft_real *chains, double *log_P, int Nsteps, int Nwalkers, double *mu, double *Sigma, double *invMass);

#endif