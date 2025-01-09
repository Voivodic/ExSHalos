#ifndef MINIMIZATION_H
#define MINIMIZATION_H

#include "sampler_h.h"
#include "posteriors.h"

/*Compute the maximum of the posterior for a given model*/
double Gauss_Newton(double *k_data, double *P_theory, double *P_data, double *inv_cov, double *params, double *Sigma, double *P_terms, int Nk, int NO, int Nparams, double Tot, double alpha);

/*Compute the maximum of the posterior for a given model using Gradient Descent*/
double Gradient_Descent(double *k_data, double *P_theory, double *P_data, double *inv_cov, double *params, double *Sigma, double *P_terms, int Nk, int NO, int Nparams, double Tot, double alpha);

#endif