#include "bias.h"

/*Measure the masked and unmasked histograms of the initial density field*/
void Measure_Histogram(fft_real *delta, fft_real *Mh, long *flag, fft_real Mmin, fft_real Mmax, int Nm, fft_real dmin, fft_real dmax, int Nbins, size_t ng, size_t nh, long *hist_unmasked, long *hist_masked, int Central){
    size_t i;
    int j, k, indm, indd;
    long h;
    fft_real dlnM, ddelta, Mmin2;

    /*Find the size of both bins*/
    dlnM = (log10(Mmax) - log10(Mmin))/Nm;
    ddelta = (dmax - dmin)/Nbins;

    /*Set the arrays to zero*/
    for(j=0;j<Nbins;j++){
        hist_unmasked[j] = 0;
        for(k=0;k<Nm;k++)
            hist_masked[k*Nbins + j] = 0;
    }

    /*Run over all cells in the grid*/
    for(i=0;i<ng;i++){
        indd = (int) floor((delta[i] - dmin)/ddelta);
        if(indd >= 0 && indd < Nbins){
            /*Count the delta field*/
            hist_unmasked[indd] += 1;

            /*Count the delta field that is a halo*/
            h = flag[i];
            if((h >= 0 && Central == TRUE) || h >= (long) nh) continue;
            h = labs(h);
            indm = (int) floor((log10(Mh[h]) - log10(Mmin))/dlnM);
            if(indm >= 0 && indm < Nm)
                hist_masked[indm*Nbins + indd] += 1;
        }
    }
}