#include "minimization.h"

/*Compute the maximum of the posterior for a given model Gauss Newton*/
double Gauss_Newton(double *k_data, double *P_theory, double *P_data, double *inv_cov, double *params, double *Sigma, double *P_terms, int Nk, int NO, int Nparams, double Tot, double alpha){
    int i, j, k, count;
    double chi2, chi2_tmp, *dchi2, **ddchi2, **dP_theory, diff, *params_tmp, **inv_H, tmp;
    void (*Model)(double *, double *, double **, double *, double *, int, int);

    /*Alloc the GSL vector and matrix*/
    gsl_matrix *H;
    H = gsl_matrix_alloc(Nparams, Nparams);

    /*Alloc the first and second derivatives of the chi2 and theory*/
    inv_H = (double **)malloc(Nparams*sizeof(double *));
    params_tmp = (double *)malloc(Nparams*sizeof(double));
    dchi2 = (double *)malloc(Nparams*sizeof(double));
    ddchi2 = (double **)malloc(Nparams*sizeof(double *));
    dP_theory = (double **)malloc(Nparams*sizeof(double *));
    for(i=0;i<Nparams;i++){
        inv_H[i] = (double *)malloc(Nparams*sizeof(double));
        ddchi2[i] = (double *)malloc(Nparams*sizeof(double));
        dP_theory[i] = (double *)malloc(Nk*sizeof(double));     
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

    /*Iterate over the Gauss-Newton method to find the best fit and the covariance matrix of the parameters*/
    diff = 2.0;
    count = 0;
    chi2 = -1e+6;
    for(i=0;i<Nparams;i++)
        params_tmp[i] = params[i];

    while(diff > Tot || diff < -Tot || -2.0*chi2/(Nk - Nparams - 1) > 3.0){   
        /*Compute the chi2 and its derivatives up to second order*/
        chi2_tmp = chi2;
        chi2 = log_like(k_data, Model, P_theory, dP_theory, P_data, inv_cov, params, P_terms, dchi2, ddchi2, Nk, NO, Nparams, TRUE);

        /*Compute the inverse of the hessian matrix*/
        for(i=0;i<Nparams;i++)
            for(j=0;j<Nparams;j++)
                gsl_matrix_set(H, i, j, (double) ddchi2[i][j]);
        H = invert_matrix(H, Nparams);
        for(i=0;i<Nparams;i++)
            for(j=0;j<Nparams;j++)
                inv_H[i][j] = gsl_matrix_get(H, i, j);

        /*Update the current position*/
        for(i=0;i<Nparams;i++){
            params_tmp[i] = params[i];
            tmp = 0.0;
            for(j=0;j<Nparams;j++)
                tmp += inv_H[i][j]*dchi2[j];
            params[i] += -alpha*tmp;
        }

        /*Compute the relative difference between the new positoin and the old one*/
        diff = chi2/chi2_tmp - 1.0;
        count ++;
        printf("diff = %f, chi2 = %f\n", diff, chi2);
    }
    printf("count = %d\n", count);

    /*Save the values of the covariance matrix of the parameters*/
    for(i=0;i<Nparams;i++)
        for(j=0;j<Nparams;j++)
            gsl_matrix_set(H, i, j, (double) -ddchi2[i][j]);
    H = invert_matrix(H, Nparams);
    for(i=0;i<Nparams;i++)
        for(j=0;j<Nparams;j++)
            Sigma[i*Nparams + j] = gsl_matrix_get(H, i, j);

    /*Free the arrays*/
    gsl_matrix_free(H);
    free(dchi2);
    for(i=0;i<Nparams;i++){
        free(inv_H[i]);
        free(ddchi2[i]);
        free(dP_theory[i]);
    }
    free(inv_H);
    free(ddchi2);
    free(dP_theory);

    return chi2;
}

/*Compute the maximum of the posterior for a given model using Gradient Descent*/
double Gradient_Descent(double *k_data, double *P_theory, double *P_data, double *inv_cov, double *params, double *Sigma, double *P_terms, int Nk, int NO, int Nparams, double Tot, double alpha){
    int i, j, k, count;
    double chi2, chi2_tmp, *dchi2, **dP_theory, diff, tmp, **ddchi2;
    void (*Model)(double *, double *, double **, double *, double *, int, int);

    /*Alloc the first and second derivatives of the chi2 and theory*/
    dchi2 = (double *)malloc(Nparams*sizeof(double));
    ddchi2 = (double **)malloc(Nparams*sizeof(double *));
    dP_theory = (double **)malloc(Nparams*sizeof(double *));
    for(i=0;i<Nparams;i++){
        ddchi2[i] = (double *)malloc(Nparams*sizeof(double));
        dP_theory[i] = (double *)malloc(Nk*sizeof(double));      
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

    /*Iterate over the gradient descent method to find the best fit and the covariance matrix of the parameters*/
    diff = 2.0*Tot;
    count = 0;
    chi2 = log_like(k_data, Model, P_theory, dP_theory, P_data, inv_cov, params, P_terms, dchi2, NULL, Nk, NO, Nparams, FALSE);

    while((diff > Tot || diff < -Tot)){   
        /*Update the current position*/
        for(i=0;i<Nparams;i++)
            params[i] += alpha*dchi2[i];

        /*Compute the chi2 and its derivatives up to second order*/
        chi2_tmp = chi2;
        chi2 = log_like(k_data, Model, P_theory, dP_theory, P_data, inv_cov, params, P_terms, dchi2, NULL, Nk, NO, Nparams, FALSE);

        /*Change alpha and do not update the position if the first guess was bad*/
        if(chi2 < 100.0*chi2_tmp){
            alpha /= 10.0;
            diff = 2.0*Tot;
            count ++;
            continue;
        }

        /*Change alpha if it is crossing the peak*/
        if(chi2 < chi2_tmp)
            alpha /= 2.0;

        /*Compute the relative difference between the new position and the old one*/
        diff = chi2/chi2_tmp - 1.0;
        count ++;
    }

    /*Compute the curvature in the best-fit point*/
    chi2 = log_like(k_data, Model, P_theory, dP_theory, P_data, inv_cov, params, P_terms, dchi2, ddchi2, Nk, NO, Nparams, TRUE);
   
    /*Alloc the GSL matrix*/
    gsl_matrix *H;
    H = gsl_matrix_alloc(Nparams, Nparams);

    /*Save the values of the covariance matrix of the parameters*/
    for(i=0;i<Nparams;i++)
        for(j=0;j<Nparams;j++)
            gsl_matrix_set(H, i, j, -ddchi2[i][j]);
    H = invert_matrix(H, Nparams);

    for(i=0;i<Nparams;i++)
        for(j=0;j<Nparams;j++)
            Sigma[i*Nparams + j] = gsl_matrix_get(H, i, j);

    /*Free the arrays*/
    gsl_matrix_free(H);
    free(dchi2);
    for(i=0;i<Nparams;i++){
        free(ddchi2[i]);
        free(dP_theory[i]);
    }
    free(ddchi2);
    free(dP_theory);

    return chi2;
}