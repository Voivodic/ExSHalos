#include "trimodule.h"

/*Compute all the cross trispectra for the covariance*/
int Tri_Spectrum(fft_real *grid, int nd, fft_real L, int ntype, int window, fft_real R, int interlacing, int Nk, fft_real k_min, fft_real k_max, long double *K1, long double *K2, long double **T, long double **Tu, long double *IT, long double *KP, long double **P, long double *IP){
    int i, j, k, l, a, b, f1, f2, f3, f4, count_sq, count_pk, ind, NPs, NTs, countk1, countk2;
    size_t tmp, ng;
    fft_real dk, kmod, kx, ky, kz, kn = 2*M_PI/L, km1, km2;
    fft_complex **gridk;
    long double **Tu1, **Tu2;

    NPs = ntype*(ntype + 1)/2;   /*Number of cross-spectra between different tracers*/
    NTs = pow(NPs, 2);           /*Number of cross-trispectra between different tracers*/
    dk = (k_max - k_min)/Nk;     /*Size of each bin of k*/
    ng = (size_t) pow(nd, 3);     /*Total number of celss*/

    /*Alloc the FFTW stuff*/
    gridk = (fft_complex **) FFTW(malloc)(ntype*sizeof(fft_complex *));
    for(l=0;l<ntype;l++)
        gridk[l] = (fft_complex *) FFTW(malloc)(nd*nd*(nd/2+1)*sizeof(fft_complex));

    /*Compute the density grids in Fourier space*/
    Compute_gridk(grid, gridk, nd, L, ntype, interlacing, R, window);

    /*Alloc the arrays used to compute the unconected part of the trispectrum*/
    Tu1 = (long double **)malloc(NTs*sizeof(long double *));
    Tu2 = (long double **)malloc(NTs*sizeof(long double *));
    for(i=0;i<NTs;i++){
        Tu1[i] = (long double *)malloc((size_t) pow(Nk, 2)*sizeof(long double));
        Tu2[i] = (long double *)malloc((size_t) pow(Nk, 2)*sizeof(long double));
    }

    /*Set the arrays for the FFTs*/
    fft_complex **in_Tm, *in_I;
    fft_real **out_Tm1,  *out_I1, **out_Tm2, *out_I2;
    FFTW(plan) p_Tm, p_I;
        
    /*Alloc the arrays for the FFTs*/
    in_Tm = (fft_complex **)FFTW(malloc)(ntype*sizeof(fft_complex *));
    for(i=0;i<ntype;i++)
        in_Tm[i] = (fft_complex *)FFTW(malloc)(nd*nd*(nd/2+1)*sizeof(fft_complex));
    in_I = (fft_complex *)FFTW(malloc)(nd*nd*(nd/2+1)*sizeof(fft_complex));

    out_Tm1 = (fft_real **)FFTW(malloc)(ntype*sizeof(fft_real *));
    out_Tm2 = (fft_real **)FFTW(malloc)(ntype*sizeof(fft_real *));
    for(i=0;i<ntype;i++){
        out_Tm1[i] = (fft_real *)FFTW(malloc)(ng*sizeof(fft_real));
        out_Tm2[i] = (fft_real *)FFTW(malloc)(ng*sizeof(fft_real));
    }
    out_I1 = (fft_real *)FFTW(malloc)(ng*sizeof(fft_real));
    out_I2 = (fft_real *)FFTW(malloc)(ng*sizeof(fft_real));

    /*Count the total number of configurations computed so far (<Ntri)*/
    count_sq = 0;
    count_pk = 0;

    /**************************/
    /* Compute the Bispectrum */
    /**************************/
    /*Compute the density grids for the first bin of k*/
    for(a=0;a<(int) Nk;a++){
        km1 = 0.0;
        countk1 = 0;
        #pragma omp parallel for private(i, j, k, l, kx, ky, kz, kmod, ind, tmp) reduction(+:km1,countk1)
        for(i=0;i<nd;i++){
            if(2*i<nd) kx = i*kn;
            else kx = (i-nd)*kn;

            for(j=0;j<nd;j++){
                if(2*j<nd) ky = j*kn;
                else ky = (j-nd)*kn;

                for(k=0;k<nd/2+1;k++){
                    if(2*k<nd) kz = k*kn;
                    else kz = (k-nd)*kn;

                    kmod = sqrt(kx*kx + ky*ky + kz*kz);
                    ind = Indice(kmod, k_min, dk);
                    tmp = (size_t) i*nd*(nd/2 + 1) + j*(nd/2 + 1) + k;

                    if(ind == a){
                        km1 += kmod;
                        countk1 += 1;

                        for(l=0;l<ntype;l++){
                            in_Tm[l][tmp][0] = gridk[l][tmp][0];
                            in_Tm[l][tmp][1] = gridk[l][tmp][1];
                        }

                        in_I[tmp][0] = 1.0;
                        in_I[tmp][1] = 0.0;
                    }
                    else{
                        for(l=0;l<ntype;l++){
                            in_Tm[l][tmp][0] = 0.0;
                            in_Tm[l][tmp][1] = 0.0;
                        }                  
        
                        in_I[tmp][0] = 0.0;
                        in_I[tmp][1] = 0.0;
                    }				
                }
            }
        }
        km1 = km1/countk1;

        /*Compute the FFTs*/
        for(l=0;l<ntype;l++){
            p_Tm = FFTW(plan_dft_c2r_3d)(nd, nd, nd, in_Tm[l], out_Tm1[l], FFTW_ESTIMATE); 
            FFTW(execute)(p_Tm);
        }
        p_I = FFTW(plan_dft_c2r_3d)(nd, nd, nd, in_I, out_I1, FFTW_ESTIMATE);
        FFTW(execute)(p_I);	
        
        /*Compute the density grids for the second bin of k*/
        for(b=0;b<a;b++){
            km2 = 0.0;
            countk2 = 0;

            #pragma omp parallel for private(i, j, k, l, kx, ky, kz, kmod, ind, tmp) reduction(+:km2,countk2)
            for(i=0;i<nd;i++){
                if(2*i<nd) kx = i*kn;
                else kx = (i-nd)*kn;

                for(j=0;j<nd;j++){
                    if(2*j<nd) ky = j*kn;
                    else ky = (j-nd)*kn;

                    for(k=0;k<nd/2+1;k++){
                        if(2*k<nd) kz = k*kn;
                        else kz = (k-nd)*kn;

                        kmod = sqrt(kx*kx + ky*ky + kz*kz);
                        ind = Indice(kmod, k_min, dk);
                        tmp = i*nd*(nd/2 + 1) + j*(nd/2 + 1) + k;
                    
                        if(ind == b){
                            km2 += kmod;
                            countk2 += 1;

                            for(l=0;l<ntype;l++){
                                in_Tm[l][tmp][0] = gridk[l][tmp][0];
                                in_Tm[l][tmp][1] = gridk[l][tmp][1];
                            }

                            in_I[tmp][0] = 1.0;
                            in_I[tmp][1] = 0.0;
                        }
                        else{
                            for(l=0;l<ntype;l++){
                                in_Tm[l][tmp][0] = 0.0;
                                in_Tm[l][tmp][1] = 0.0;
                            }                       
        
                            in_I[tmp][0] = 0.0;
                            in_I[tmp][1] = 0.0;
                        }				
                    }
                }
            }
            km2 = km2/countk2;

            /*Compute the FFTs*/
            for(l=0;l<ntype;l++){
                p_Tm = FFTW(plan_dft_c2r_3d)(nd, nd, nd, in_Tm[l], out_Tm2[l], FFTW_ESTIMATE); 
                FFTW(execute)(p_Tm);
            }
            p_I = FFTW(plan_dft_c2r_3d)(nd, nd, nd, in_I, out_I2, FFTW_ESTIMATE);
            FFTW(execute)(p_I);	

            /*Compute the sum of the grids in real space and save the unormalized Bispectrum and the number of triangles*/
            for(l=0;l<NTs;l++){
                T[l][count_sq] = 0.0;
                Tu1[l][count_sq] = 0.0;
                Tu2[l][count_sq] = 0.0;
            }
            IT[count_sq] = 0.0;
            for(i=0;i<nd;i++)
                for(j=0;j<nd;j++)
                    for(k=0;k<nd;k++){
                        tmp = (size_t) i*nd*nd + j*nd + k;
                        
                        ind = 0;
                        for(f1=0;f1<ntype;f1++)
                            for(f2=0;f2<=f1;f2++)
                                for(f3=0;f3<ntype;f3++)
                                    for(f4=0;f4<=f3;f4++){
                                        T[ind][count_sq] += (long double) out_Tm1[f1][tmp]*out_Tm1[f2][tmp]*out_Tm2[f3][tmp]*out_Tm2[f4][tmp];
                                        Tu1[ind][count_sq] += (long double) out_Tm1[f1][tmp]*out_Tm1[f2][tmp];
                                        Tu2[ind][count_sq] += (long double) out_Tm2[f3][tmp]*out_Tm2[f4][tmp];

                                        ind ++;
                                    }
                        IT[count_sq] += (long double) out_I1[tmp]*out_I1[tmp]*out_I2[tmp]*out_I2[tmp];
                    }
            K1[count_sq] = (long double) km1;
            K2[count_sq] = (long double) km2;
            count_sq ++;
        }/*Close the loop in the second field*/

        /*Compute the sum of the grids in real space and save the unormalized Bispectrum and the number of triangles*/
        for(l=0;l<NTs;l++){
            T[l][count_sq] = 0.0;
            Tu1[l][count_sq] = 0.0;
            Tu2[l][count_sq] = 0.0;
        }
        for(l=0;l<NPs;l++)
            P[l][count_pk] = 0.0;
        IT[count_sq] = 0.0;
        IP[count_pk] = 0.0;
        for(i=0;i<nd;i++)
            for(j=0;j<nd;j++)
                for(k=0;k<nd;k++){
                    tmp = (size_t) i*nd*nd + j*nd + k;
                    
                    ind = 0;
                    for(f1=0;f1<ntype;f1++)
                        for(f2=0;f2<=f1;f2++)
                            for(f3=0;f3<ntype;f3++)
                                for(f4=0;f4<=f3;f4++){
                                    T[ind][count_sq] += (long double) out_Tm1[f1][tmp]*out_Tm1[f2][tmp]*out_Tm1[f3][tmp]*out_Tm1[f4][tmp];
                                    Tu1[ind][count_sq] += (long double) out_Tm1[f1][tmp]*out_Tm1[f2][tmp];
                                    Tu2[ind][count_sq] += (long double) out_Tm1[f3][tmp]*out_Tm1[f4][tmp];

                                    ind ++;
                                }
                    IT[count_sq] += (long double) out_I1[tmp]*out_I1[tmp]*out_I1[tmp]*out_I1[tmp];

                    ind = 0;
                    for(f1=0;f1<ntype;f1++)
                        for(f2=0;f2<=f1;f2++){
                            P[ind][count_pk] += (long double) out_Tm1[f1][tmp]*out_Tm1[f2][tmp];
                            ind ++;
                        }
                    IP[count_pk] += (long double) out_I1[tmp]*out_I1[tmp];
                }
        K1[count_sq] = (long double) km1;
        K2[count_sq] = (long double) km1;
        count_sq ++;
        KP[count_pk] = (long double) km1;
        count_pk ++;
    }/*Close the loop in the first field*/

    /*Free the FFTW3 arrays*/
    for(i=0;i<ntype;i++){
        FFTW(free)(in_Tm[i]);
        FFTW(free)(out_Tm1[i]);     
        FFTW(free)(out_Tm2[i]); 
        FFTW(free)(gridk[i]);
    }
    FFTW(free)(in_Tm); FFTW(free)(in_I);
    FFTW(free)(out_Tm1); FFTW(free)(out_Tm2); FFTW(free)(out_I1); FFTW(free)(out_I2); 
    FFTW(free)(p_Tm); FFTW(free)(p_I);
    FFTW(free)(gridk);

    /*Fill Compute the normalized quantities*/
    for(i=0;i<count_sq;i++){
        for(j=0;j<NTs;j++){
            T[j][i] = T[j][i]/IT[i]*pow(L, 3.0);
            Tu[j][i] = Tu1[j][i]*Tu2[j][i]/IT[i]*pow(L, 3.0)/ng;
        }
        IT[i] = IT[i]/ng;
    }

    for(i=0;i<count_pk;i++){
        for(j=0;j<NPs;j++)
            P[j][i] = P[j][i]/IP[i];
        IP[i] = IP[i]/ng;
    }

    for(i=0;i<NTs;i++){
        free(Tu1[i]);
        free(Tu2[i]);
    }
    free(Tu1); free(Tu2);

    return count_sq;
}
