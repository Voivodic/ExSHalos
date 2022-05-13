#include "powermodule.h"

/*Compute all the cross and auto spectra*/
void Power_Spectrum(fft_real *grid, int nd, fft_real L, int ntype, int window, fft_real R, int interlacing, int Nk, fft_real k_min, fft_real k_max, long double *Kmean, long double *P, long *count_k, int l_max){
    int i, j, k, l, m, n, count, ind, NPs;
    size_t tmp;
    fft_real dk, k_mod, kx, ky, kz, kn = 2*M_PI/L, mu;
    fft_complex **gridk;

    NPs = (ntype*(ntype+1))/2;           /*Number of cross-spectra between different tracers*/
    dk = (k_max - k_min)/Nk;            /*Size of each bin of k*/

    /*Alloc the FFTW stuff*/
    gridk = (fft_complex **) FFTW(malloc)(ntype*sizeof(fft_complex *));
    for(i=0;i<ntype;i++)
        gridk[i] = (fft_complex *) FFTW(malloc)(nd*nd*(nd/2+1)*sizeof(fft_complex));

    /*Compute the density grids in Fourier space*/
    Compute_gridk(grid, gridk, nd, L, ntype, interlacing, R, window);

    /*Compute the power spectra*/
    #pragma omp parallel private(i, j, k, l, m, n, kx, ky, kz, k_mod, ind, tmp, count, mu) 
	{
        long double *P_private = (long double *) malloc(NPs*(l_max+1)*Nk*sizeof(long double));
        long double *k_private = (long double *) malloc(Nk*sizeof(long double));
        long *count_k_private = (long *) malloc(Nk*sizeof(long));

        for(i=0;i<Nk;i++){
            k_private[i] = 0.0;
            count_k_private[i] = 0;
            for(j=0;j<NPs;j++)
                for(k=0;k<=l_max;k++)
                    P_private[(j*(l_max+1) + k)*Nk + i] = 0.0;
        }

        #pragma omp for
        for(i=0;i<nd;i++){
            if(2*i<nd) kx = i*kn;
            else kx = (i-nd)*kn;

            for(j=0;j<nd;j++){
                if(2*j<nd) ky = j*kn;
                else ky = (j-nd)*kn;

                for(k=0;k<nd/2+1;k++){
                    if(2*k<nd) kz = k*kn;
                    else kz = (k-nd)*kn;

                    if((kz==0.0 && ky<0.0) || (kz==0.0 && ky==0.0 && kx<0.0))
                        continue;

                    k_mod = sqrt(kx*kx + ky*ky + kz*kz);
                    ind = Indice(k_mod, k_min, dk);
                    mu = kz/k_mod;

                    if(ind < Nk && ind >= 0){
                        tmp = (size_t) i*nd*(nd/2 + 1) + j*(nd/2 + 1) + k;

                        count = 0;
                        for(l=0;l<ntype;l++)
                            for(m=0;m<=l;m++){
                                for(n=0;n<=l_max;n++)
                                    P_private[(count*(l_max+1) + n)*Nk + ind] += (long double) (gridk[l][tmp][0]*gridk[m][tmp][0] + gridk[l][tmp][1]*gridk[m][tmp][1])*gsl_sf_legendre_Pl(n, mu);
                                count++;
                                }

                        k_private[ind] += (long double) k_mod;
                        count_k_private[ind] += 1;
                    }
                }
            }
        }

        #pragma omp critical
        {
            for(i=0;i<Nk;i++){
                Kmean[i] += k_private[i];
                count_k[i] += count_k_private[i];
                for(j=0;j<NPs;j++)
                    for(k=0;k<=l_max;k++)
                        P[(j*(l_max+1) + k)*Nk + i] += P_private[(j*(l_max+1) + k)*Nk + i];
            }

            free(P_private);
            free(k_private);
            free(count_k_private);
        }
    }

    for(i=0;i<ntype;i++)
        FFTW(free)(gridk[i]);
    FFTW(free)(gridk);

    /*Take the mean*/
    for(i=0;i<Nk;i++)
        if(count_k[i]>0){
            for(j=0;j<NPs;j++)
                for(k=0;k<=l_max;k++)
                    P[(j*(l_max+1) + k)*Nk + i] = P[(j*(l_max+1) + k)*Nk + i]/count_k[i];
            Kmean[i] = Kmean[i]/count_k[i];	
            count_k[i] = 2.0*count_k[i];
        }
}