#include "clpt.h"

/*Compute xi_l,m for a given power spectrum using a gaussian smoothing and the interpolation in the r->0 limit*/
void Xi_lm(double *k, double *P, int Nk, double *rlm, double *xilm, int Nr, int l, int mk, int mr, int K, double alpha, double Rmax){
    double *r, *xi, Int, Ierr, *Ptmp, *rtmp, *xitmp, R, rmin;
    int i, imin;

    /*Alloc the arrays for the fftlog*/
    r = (double *)malloc(Nk*sizeof(double));
    xi = (double *)malloc(Nk*sizeof(double));

    /*Compute the integral using fftlog*/
    fftlog_ComputeXiLM(l, mk, Nk, k, P, r, xi);
    for(i=0;i<Nk;i++)
        xi[i] = xi[i]*pow(r[i], mr);

    /*Smooth the results using a Gaussian filter*/
    gsl_vector *xi_nonsmooth = gsl_vector_alloc(Nk);    //Convert to a gsl vector
    gsl_vector *xi_smooth = gsl_vector_alloc(Nk);    //Convert to a gsl vector
    for(i=0;i<Nk;i++)
        gsl_vector_set(xi_nonsmooth, i, xi[i]);
    free(xi);

    gsl_filter_gaussian_workspace *gauss_p = gsl_filter_gaussian_alloc(K);  //Alloc the workspace
    gsl_filter_gaussian(GSL_FILTER_END_PADVALUE, alpha, 0, xi_nonsmooth, xi_smooth, gauss_p);  //Apply the gaussian filter

    gsl_filter_gaussian_free(gauss_p);

    /*Compute the integral in r->0 limit*/
    if(l + mr > 0){
        Int = 0.0;
        Ierr = 0.0;
    }
    else if (l - mr < 0){
        printf("Divergent integral! l - mr = %d!\n", l-mr);
        exit(0);
    }
    else{
        /*Compute the function to be integrated*/
        Ptmp = (double *)malloc(Nk*sizeof(double));
        for(i=0;i<Nk;i++){
            Ptmp[i] = P[i]*pow(k[i], mk+l);
        }

        gsl_interp_accel *facc = gsl_interp_accel_alloc();
        gsl_spline *fspline = gsl_spline_alloc(gsl_interp_cspline, Nk);

        gsl_spline_init(fspline, k, Ptmp, Nk);  //Interpolate the power spectrum
        free(Ptmp);

        gsl_integration_workspace *w = gsl_integration_workspace_alloc(1000);
        struct finterp_params params = {facc, fspline};

        gsl_function F;     //Define the gsl function
        F.function = &finterp;
        F.params = &params;

        gsl_integration_qags(&F, k[0], k[Nk-1], 0, 1e-4, 1000, w, &Int, &Ierr);    //Compute the integral

        gsl_integration_workspace_free(w);
        gsl_spline_free(fspline);
        gsl_interp_accel_free(facc);

        /*Compute the double factorial part*/
        for(i=0;i<=l;i++)   
            Int /= (2*i+1);
        Int /= (2.0*M_PI*M_PI);
    }

    /*Compute the value of rmin where to use the fftlog result*/
    imin = 0;
    for(i=0;i<Nk;i++){
        R = fabs(gsl_vector_get(xi_nonsmooth, i)/gsl_vector_get(xi_smooth, i) - 1.0);
        if(R > Rmax){
            rmin = r[i];
            imin = i;
        }
    }

    /*Do the interpolation*/
    rtmp = (double *)malloc((Nk-imin)*sizeof(double));
    xitmp = (double *)malloc((Nk-imin)*sizeof(double));

    for(i=0;i<Nk-imin;i++){
        rtmp[i] = r[imin+i];
        xitmp[i] = gsl_vector_get(xi_smooth, imin+i);        
    }

    gsl_interp_accel *xi_acc = gsl_interp_accel_alloc();
    gsl_spline *xi_spline = gsl_spline_alloc(gsl_interp_cspline, Nk-imin);
    gsl_spline_init(xi_spline, rtmp, xitmp, Nk-imin);  

    /*Construct the final correlation function*/
    if(Int != 0.0)  R = Int/gsl_spline_eval(xi_spline, r[imin], xi_acc);
    else    R = 1.0;
    for(i=0;i<Nr;i++){
        if(rlm[i] < rmin)
            xilm[i] = Int;
        else
            xilm[i] = R*gsl_spline_eval(xi_spline, rlm[i], xi_acc);
    }

    /*Free some memory*/
    gsl_vector_free(xi_smooth);
    gsl_vector_free(xi_nonsmooth);    
    free(r);
    free(rtmp);
    free(xitmp);
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
    struct finterp_params params = {facc, fspline};

    gsl_function F;
    F.function = &finterp;
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
        A1[i] = 2.0*(I0/3.0 - xi1m1[i]);
        A2[i] = 2.0*(3.0*xi1m1[i] - xi0p0[i]);
        printf("%d - %lf - (%e, %e, %e, %e) - (%e, %e) - %e\n", i, r[i], I0, xi1m1[i], xi0p0[i],  xi0p2[i], A1[i], A2[i], A1[i] + A2[i]);
    }

    /*Interpolate both functions for the integral*/
    gsl_interp_accel *A1acc = gsl_interp_accel_alloc();
    gsl_interp_accel *A2acc = gsl_interp_accel_alloc();
    gsl_spline *A1spline = gsl_spline_alloc(gsl_interp_cspline, N);
    gsl_spline *A2spline = gsl_spline_alloc(gsl_interp_cspline, N);

    gsl_spline_init(A1spline, r, A1, N);
    gsl_spline_init(A2spline, r, A2, N);

    struct finterp_params A1params = {A1acc, A1spline};    
    struct finterp_params A2params = {A2acc, A2spline};    

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
            P11[i] += 4.0*M_PI*sqrt(M_PI/(2*n+1))*pow(n+0.5, 2-n)*k[i]*k[i]*pow(finterp((n+0.5)/k[i], &A2params), n)*exp(-0.5*(finterp((n+0.5)/k[i], &A1params) + finterp((n+0.5)/k[i], &A2params))*k[i]*k[i]);            
        }
    }

    /*Free the interpolation stuff*/
    gsl_spline_free(A1spline);
    gsl_spline_free(A2spline);
    gsl_interp_accel_free(A1acc);
    gsl_interp_accel_free(A2acc);
}