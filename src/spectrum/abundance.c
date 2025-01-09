#include "abundance.h"

/*Measure the abundance of a given halo catalogue*/
void Measure_Abundance(fft_real *Mh, size_t nh, fft_real Mmin, fft_real Mmax, int Nm, fft_real *Mmean, fft_real *dn, fft_real *dn_err, fft_real Lx, fft_real Ly, fft_real Lz){
    int *N, ind;
    size_t i;
    float dlnM;

    /*Define some arrays and numbers*/
    dlnM = (log(Mmax) - log(Mmin))/Nm;
    N = (int *)malloc(Nm*sizeof(int));

    /*Set the arrays*/
    for(i=0;i<Nm;i++){
        N[i] = 0;
        Mmean[i] = 0.0;
    }

    /*Run over all halos*/
    for(i=0;i<nh;i++){
        ind = floor((log(Mh[i]) - log(Mmin))/dlnM);
        if(ind > 0 && ind < Nm){
            N[ind] += 1;
            Mmean[ind] += Mh[i];
        }	
    }

    /*Compute the differential mass function*/
    for(i=0;i<Nm;i++){
        if(N[i] > 0) Mmean[i] = Mmean[i]/N[i];
        dn[i] = N[i]/(Lx*Ly*Lz)/dlnM;
        dn_err[i] = sqrt(N[i])/(Lx*Ly*Lz)/dlnM;
    }

    /*Free the used arrays*/
    free(N);
}