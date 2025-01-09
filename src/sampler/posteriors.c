#include "posteriors.h"

/*Defining the global variables*/
int MODEL;      
int Prior_type; 

/*Compute the theoretical prediction for Pgg and its derivative*/
void Pgg_theory(double *k, double *P_theory, double **dP_theory, double *params, double *P_terms, int Nk, int NO){
    int i, j, l, count;

    /*Initialize the output*/
    for(i=0;i<Nk;i++){
        P_theory[i] = P_terms[i];
        for(j=0;j<NO+1;j++)
            dP_theory[j][i] = 0.0;
    }

    /*Compute the theoretical prediction*/
    for(i=0;i<Nk;i++){
        /*Deterministic part*/
        /*Terms proportional to the operator 1*/
        count = 1;
        for(j=1;j<NO;j++){
            P_theory[i] += 2.0*params[j-1]*P_terms[count*Nk + i];
            dP_theory[j-1][i] = P_terms[count*Nk + i];
            count += j+1;
        }

        /*Other terms*/
        count = 1;
        for(j=1;j<NO;j++){
            count ++;
            for(l=1;l<j;l++){
                P_theory[i] += 2.0*params[j-1]*params[l-1]*P_terms[count*Nk + i];    
                dP_theory[j-1][i] += 2.0*params[l-1]*P_terms[count*Nk + i];   
                dP_theory[l-1][i] += 2.0*params[j-1]*P_terms[count*Nk + i]; 
                count ++;
            }
            P_theory[i] += params[j-1]*params[j-1]*P_terms[count*Nk + i];
            dP_theory[j-1][i] += 2.0*params[j-1]*P_terms[count*Nk + i];
            count ++;
        }

        /*Stochastic part*/
        P_theory[i] += params[NO-1] + k[i]*k[i]*params[NO];
        dP_theory[NO-1][i] = 1.0;
        dP_theory[NO][i] = k[i]*k[i];
    }
}

/*Compute the theoretical prediction for Pgm*/
void Pgm_theory(double *k, double *P_theory, double **dP_theory, double *params, double *P_terms, int Nk, int NO){
    int i, j, l, count;

    /*Initialize the output*/
    for(i=0;i<Nk;i++)
        P_theory[i] = P_terms[i];

    /*Compute the theoretical prediction*/
    for(i=0;i<Nk;i++){
        /*Deterministic part*/
        count = 1;
        for(j=1;j<NO;j++){
            P_theory[i] += params[j-1]*P_terms[count*Nk + i];
            dP_theory[j-1][i] = P_terms[count*Nk + i];
            count += j+1;
        }

        /*Stochastic part*/
        P_theory[i] += k[i]*k[i]*params[NO-1];
        dP_theory[NO-1][i] = k[i]*k[i];
    }
}

/*Compute the log of the likelihood*/
double log_like(double *k, void (*Model)(double *, double *, double **, double *, double *, int, int), double *P_theory, double **dP_theory, double *P_data, double *inv_cov, double *params, double *P_terms, double *dchi2, double **ddchi2, int Nk, int NO, int Nparams, int get_H){
    int a, b, i, j, l;
    double chi2;

    /*Compute the theoretical prediction*/
    (*Model)(k, P_theory, dP_theory, params, P_terms, Nk, NO);

    /*Initialize the array with the derivatives of chi2*/
    for(l=0;l<Nparams;l++)
        dchi2[l] = 0.0;

    /*Compute the chi^2*/
    chi2 = 0.0;
    for(i=0;i<Nk;i++)
        for(j=0;j<Nk;j++){
            chi2 += (P_data[i] - P_theory[i])*(P_data[j] - P_theory[j])*inv_cov[i*Nk + j];
            for(l=0;l<Nparams;l++)
                dchi2[l] += -dP_theory[l][i]*(P_theory[j] - P_data[j])*inv_cov[i*Nk + j];
        }

    /*Compute the Hessian of chi2*/
    if(get_H == TRUE){
        /*Compute the Hessian matrix*/       
        for(a=0;a<Nparams;a++)
            for(b=0;b<Nparams;b++){
                ddchi2[a][b] = 0.0;
                for(i=0;i<Nk;i++)
                    for(j=0;j<Nk;j++)
                        ddchi2[a][b] += (double) -(dP_theory[a][i]*dP_theory[b][j])*inv_cov[i*Nk + j]; //+ ddP_theory[a][b][i]*(P_theory[j] - P_data[j])
            }
    }

    return -0.5*chi2;
}

/*Compute the flat log of the prior*/
double log_prior_flat(double *params, double *params_priors, int Nparams){
    int i;
    double prior;

    /*Return inf if it is outside the bounds*/
    prior = 0.0;
    for(i=0;i<Nparams;i++)
        if(params[i] < params_priors[2*i] || params[i] > params_priors[2*i+1]){
            prior = (double) -INFINITY;
            break;
        }

    return prior;
}

/*Compute the exp log of the prior*/
double log_prior_gauss(double *params, double *params_priors, int Nparams){
    int i;
    double prior;

    /*Compute the gaussian prior*/
    prior = 0.0;
    for(i=0;i<Nparams;i++)
        prior += pow((params[i] - params_priors[2*i])/params_priors[2*i+1], 2.0);

    return -0.5*prior;
}