#include "random_generator.h"

/*Generate an array of random number following a given PDF*/
void Generate_Random_Array(gsl_spline *spline_rho, unsigned long long nps, double rmin, double rmax, fft_real *rng, gsl_rng *rng_ptr, int Inter_log, int NRs, int Neps, double Tot){
	unsigned long long i;
	double A, rtmp, pdf0;
	double *x, *I, *Eps, *r, *rho;

    /*Alloc the arrays used for the interpolations*/
    x = (double *) malloc(NRs*sizeof(double));
    check_memory(x, "x")
    I = (double *) malloc(NRs*sizeof(double));
    check_memory(I, "I")

	/*Construct the interpolator used to generate the random coordinate*/
	gsl_interp_accel *acc = gsl_interp_accel_alloc();
	gsl_spline *spline_r, *spline_I;
	spline_r = gsl_spline_alloc(gsl_interp_cspline, Neps);
	spline_I = gsl_spline_alloc(gsl_interp_cspline, NRs);

    /*Contruct the array of rs where to compute the CDF and interpolate it*/
    if(Inter_log == TRUE)
        for(i=0;i<NRs;i++)
            x[i] = pow(10.0, log10(rmin) + (log10(rmax) - log10(rmin))*i/(NRs - 1));
    else
        for(i=0;i<NRs;i++)
            x[i] = rmin + (rmax - rmin)*i/(NRs - 1);

    /*Compute the integral of the PDF (the CDF) and interpolate it*/
    I[0] = 0.0;
    for(i=1;i<NRs;i++)
        I[i] = I[i-1] + gsl_spline_eval_integ(spline_rho, x[i-1], x[i], acc);
    gsl_spline_init(spline_I, x, I, NRs);
    A = (double) gsl_spline_eval(spline_I, rmax, acc);

    /*Free the arrays used in the interpolation*/
    free(x);
    free(I);

    /*Alloc the arrays used in the solution of the non-linear equation*/
    Eps = (double *) malloc(Neps*sizeof(double));
    check_memory(Eps, "Eps")
    r = (double *) malloc(Neps*sizeof(double));
    check_memory(r, "r")

	/*Define the arrays of Eps used to generate the r*/
	for(i=0;i<Neps;i++)
		Eps[i] = ((double) i)/(Neps - 1);

    /*Run over different values of epsilon*/
    r[0] = rmin;
    for(i=1;i<Neps-1;i++){
        r[i] = r[i-1];      //First guess
        rtmp = r[i]/2.0;

        /*Find the solution of the equation F^{-1}(r) = Eps*/
        while(fabs((r[i] - rtmp)/rtmp) > Tot){
            rtmp = r[i];
            pdf0 = (double) gsl_spline_eval(spline_rho, r[i], acc);
            r[i] = r[i] - (gsl_spline_eval(spline_I, r[i], acc)/A - Eps[i])/(pdf0/A);

            if(r[i] >= rmax || r[i] <= rmin)	r[i] = Eps[i];
        }
    }
    gsl_spline_free(spline_I);	
    r[Neps-1] = rmax;

    /*Interpolate the r(Eps) relation*/
    gsl_spline_init(spline_r, Eps, r, Neps);

    /*Free the arrays used in the interpolation*/
    free(r);
    free(Eps);

	/*Generate the random points*/
    #pragma omp for private(i)
    for(i=0;i<nps;i++)
        rng[i] = (fft_real) gsl_spline_eval(spline_r, gsl_rng_uniform(rng_ptr), acc);

    /*Free the gsl objects*/
	gsl_spline_free(spline_r);	
	gsl_interp_accel_free(acc);
}