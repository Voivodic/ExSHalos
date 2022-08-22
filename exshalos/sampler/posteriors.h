#ifndef POSTERIORS_H
#define POSTERIORS_H

#include "sampler_h.h"

/*Define some global variables*/
int MODEL;      //Variable with the model to be used in the sampling
int Prior_type; //Variable with the type of prior: Flat (0) or Gaussian (1)

/*Compute the theoretical prediction for Pgg and its derivative*/
void Pgg_theory(double *k, double *P_theory, double **dP_theory, double *params, double *P_terms, int Nk, int NO);

/*Compute the theoretical prediction for Pgm*/
void Pgm_theory(double *k, double *P_theory, double **dP_theory, double *params, double *P_terms, int Nk, int NO);

/*Compute the log of the likelihood*/
double log_like(double *k, void (*Model)(double *, double *, double **, double *, double *, int, int), double *P_theory, double **dP_theory, double *P_data, double *inv_cov, double *params, double *P_terms, double *dchi2, double **ddchi2, int Nk, int NO, int Nparams, int get_H);

/*Compute the flat log of the prior*/
double log_prior_flat(double *params, double *params_priors, int Nparams);

/*Compute the exp log of the prior*/
double log_prior_gauss(double *params, double *params_priors, int Nparams);


#endif