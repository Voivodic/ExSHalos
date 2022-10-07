#include "clpt.h"

/*Supress the linear power spectrum on small scales*/
void P_smooth(double *k, double *Plin, double *P, int N, double Lambda){
    int i;

    for(i=0;i<N;i++){
        if(k[i] > Lambda)
            P[i] = 0.0;
        else
            P[i] = Plin[i];
    }
}

/*Define the generic function to be integrated*/
double func(double x, void *p){
    struct func_params *params = (struct func_params *) p;
    gsl_interp_accel *facc = (params->facc);
    gsl_spline *fspline = (params->fspline);

    double f = gsl_spline_eval(fspline, x, facc);
    return f;
}

/*Compute the matter-matter power spectrum for 1LPT*/
void CLPT_P11(double *k, double *Plin, int N, double *P11, int nmin, int nmax, double kmax){
    int i, j, n;
    double I0, I0err, *xi1m1, *xi0p0, *xi0p2, *A1, *A2, *r, *Intr, *Intk;

    /*Alloc the array with the real space positions*/
    r = (double *)malloc(N*sizeof(double));

    /*Integrate the r-independent part*/
    gsl_interp_accel *facc = gsl_interp_accel_alloc();
    gsl_spline *fspline = gsl_spline_alloc(gsl_interp_cspline, N);

    gsl_spline_init(fspline, k, Plin, N);

    gsl_integration_workspace *w = gsl_integration_workspace_alloc(1000);
    struct func_params params = {facc, fspline};

    gsl_function F;
    F.function = &func;
    F.params = &params;

    gsl_integration_qags(&F, k[0], k[N-1], 0, 1e-4, 1000, w, &I0, &I0err);

    gsl_integration_workspace_free(w);
    gsl_spline_free(fspline);
    gsl_interp_accel_free(facc);

    /*Compute the scalar functions used to compute the final spectrum*/
    xi1m1 = (double *)malloc(N*sizeof(double));
    xi0p0 = (double *)malloc(N*sizeof(double));
    xi0p2 = (double *)malloc(N*sizeof(double));

    fftlog_ComputeXiLM(1, -1, N, k, Plin, r, xi1m1);
    for(i=0;i<N;i++)
        xi1m1[i] = xi1m1[i]/r[i];       
    fftlog_ComputeXiLM(0, 0, N, k, Plin, r, xi0p0);
    fftlog_ComputeXiLM(0, 2, N, k, Plin, r, xi0p2);

    /*Compute the two functions that enter in the final spectrum*/
    A1 = (double *)malloc(N*sizeof(double));
    A2 = (double *)malloc(N*sizeof(double));
    
    for(i=0;i<N;i++){
        A1[i] = 2.0*(I0/3.0 - I1[i]);
        A2[i] = 2.0*(3.0*I1[i] - I2[i]);
        printf("%d - %lf - (%e, %e, %e, %e) - (%e, %e) - %e\n", i, r[i], I0, I1[i], I2[i],  I3[i], A1[i], A2[i], A1[i] + A2[i]);
    }

    /*Interpolate both functions for the integral*/
    gsl_interp_accel *A1acc = gsl_interp_accel_alloc();
    gsl_interp_accel *A2acc = gsl_interp_accel_alloc();
    gsl_spline *A1spline = gsl_spline_alloc(gsl_interp_cspline, N);
    gsl_spline *A2spline = gsl_spline_alloc(gsl_interp_cspline, N);

    gsl_spline_init(A1spline, r, A1, N);
    gsl_spline_init(A2spline, r, A2, N);

    struct func_params A1params = {A1acc, A1spline};    
    struct func_params A2params = {A2acc, A2spline};    

    /*Compute the final power spectrum for k<kmax*/
    for(i=0;i<N;i++)
        P11[i] = 0.0;
    Intr = (double *)malloc(N*sizeof(double));
    Intk = (double *)malloc(N*sizeof(double));
    for(i=0;i<N;i++){
        if(k[i] > kmax)
            break;

        /*First loop computing the full integrals*/
        for(n=0;n<nmin;n++){
            for(j=0;j<N;j++){
                Intr[j] = exp(-0.5*(A1[j] + A2[j])*k[i]*k[i])*pow(A2[j], n);
                //printf("(%lf, %lf)  - %e\n", k[i], r[j], Intr[j]);
            }
            fftlog_ComputeXiLM(n, 2-n, N, r, Intr, k, Intk);

            P11[i] += pow(2.0*M_PI, 3.0)*pow(k[i], n)*Intk[i];
            //printf("(%d, %d) - %lf\n", i, n, P11[i]);
        }

        /*Second loop using the Limber approximation*/
        for(n=nmin;n<nmax;n++){
            if((n+0.5)/k[i] > r[N-1])   continue;
            P11[i] += 4.0*M_PI*sqrt(M_PI/(2*n+1))*pow(n+0.5, 2-n)*k[i]*k[i]*pow(func((n+0.5)/k[i], &A2params), n)*exp(-0.5*(func((n+0.5)/k[i], &A1params) + func((n+0.5)/k[i], &A2params))*k[i]*k[i]);            
        }
    }

    /*Free the interpolation stuff*/
    gsl_spline_free(A1spline);
    gsl_spline_free(A2spline);
    gsl_interp_accel_free(A1acc);
    gsl_interp_accel_free(A2acc);
}