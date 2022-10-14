#include "analytical_h.h"

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
double finterp(double x, void *p){
    struct finterp_params *params = (struct finterp_params *) p;
    gsl_interp_accel *facc = (params->facc);
    gsl_spline *fspline = (params->fspline);

    double f = gsl_spline_eval(fspline, x, facc);
    return f;
}