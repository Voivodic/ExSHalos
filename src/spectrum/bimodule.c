#include "bimodule.h"

/*Compute all the cross bispectra*/
int Bi_Spectrum(fft_real *grid, int nd, fft_real L, int ntype, int window, fft_real R, int interlacing, int Nk, fft_real k_min, fft_real k_max, long double *K1, long double *K2, long double *K3, long double **B, long double *IB, long double *KP, long double **P, long double *IP, int verbose){
    int i, j, k, l, a, b, c, f1, f2, f3, tmpc, count_tri, count_pk, ind, NPs, NBs, countk1, countk2, countk3;
    size_t tmp, ng;
    fft_real dk, kmod, kx, ky, kz, kn = 2*M_PI/L, km1, km2, km3;
    fft_complex **gridk;

    NPs = ntype*(ntype+1)/2;       /*Number of cross-spectra between different tracers*/
    NBs = pow(ntype, 3);           /*Number of cross-bispectra between different tracers*/
    dk = (k_max - k_min)/Nk;       /*Size of each bin of k*/
    ng = (size_t) pow(nd, 3);               /*Total number of celss*/

    /*Alloc the FFTW stuff*/
    gridk = (fft_complex **) FFTW(malloc)(ntype*sizeof(fft_complex *));
    for(l=0;l<ntype;l++)
        gridk[l] = (fft_complex *) FFTW(malloc)(nd*nd*(nd/2+1)*sizeof(fft_complex));

    /*Compute the density grids in Fourier space*/
    Compute_gridk(grid, gridk, nd, L, ntype, interlacing, R, window);

    /*Set the arrays for the FFTs*/
    fft_complex **in_Bm, *in_I;
    fft_real **out_Bm1,  *out_I1, **out_Bm2, *out_I2, **out_Bm3, *out_I3;
    FFTW(plan) p_Bm, p_I;
        
    /*Alloc the arrays for the FFTs*/
    in_Bm = (fft_complex **)FFTW(malloc)(ntype*sizeof(fft_complex *));
    for(i=0;i<ntype;i++)
        in_Bm[i] = (fft_complex *)FFTW(malloc)(nd*nd*(nd/2+1)*sizeof(fft_complex));
    in_I = (fft_complex *)FFTW(malloc)(nd*nd*(nd/2+1)*sizeof(fft_complex));

    out_Bm1 = (fft_real **)FFTW(malloc)(ntype*sizeof(fft_real *));
    out_Bm2 = (fft_real **)FFTW(malloc)(ntype*sizeof(fft_real *));
    out_Bm3 = (fft_real **)FFTW(malloc)(ntype*sizeof(fft_real *));
    for(i=0;i<ntype;i++){
        out_Bm1[i] = (fft_real *)FFTW(malloc)(ng*sizeof(fft_real));
        out_Bm2[i] = (fft_real *)FFTW(malloc)(ng*sizeof(fft_real));
        out_Bm3[i] = (fft_real *)FFTW(malloc)(ng*sizeof(fft_real));
    }
    out_I3 = (fft_real *)FFTW(malloc)(ng*sizeof(fft_real));
    out_I1 = (fft_real *)FFTW(malloc)(ng*sizeof(fft_real));
    out_I2 = (fft_real *)FFTW(malloc)(ng*sizeof(fft_real));

    /*Count the total number of configurations computed so far (<Ntri)*/
    count_tri = 0;
    count_pk = 0;

    /**************************/
    /* Compute the Bispectrum */
    /**************************/
    /*Compute the density grids for the first bin of k*/
    for(a=0;a<(int) Nk;a++){
        if(verbose == TRUE)
            printf("a = %d of %d. %d Bispectra computed so far\n", a+1, Nk, count_tri);

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
                            in_Bm[l][tmp][0] = gridk[l][tmp][0];
                            in_Bm[l][tmp][1] = gridk[l][tmp][1];
                        }

                        in_I[tmp][0] = 1.0;
                        in_I[tmp][1] = 0.0;
                    }
                    else{
                        for(l=0;l<ntype;l++){
                            in_Bm[l][tmp][0] = 0.0;
                            in_Bm[l][tmp][1] = 0.0;
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
            p_Bm = FFTW(plan_dft_c2r_3d)(nd, nd, nd, in_Bm[l], out_Bm1[l], FFTW_ESTIMATE); 
            FFTW(execute)(p_Bm);
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
                                in_Bm[l][tmp][0] = gridk[l][tmp][0];
                                in_Bm[l][tmp][1] = gridk[l][tmp][1];
                            }

                            in_I[tmp][0] = 1.0;
                            in_I[tmp][1] = 0.0;
                        }
                        else{
                            for(l=0;l<ntype;l++){
                                in_Bm[l][tmp][0] = 0.0;
                                in_Bm[l][tmp][1] = 0.0;
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
                p_Bm = FFTW(plan_dft_c2r_3d)(nd, nd, nd, in_Bm[l], out_Bm2[l], FFTW_ESTIMATE); 
                FFTW(execute)(p_Bm);
            }
            p_I = FFTW(plan_dft_c2r_3d)(nd, nd, nd, in_I, out_I2, FFTW_ESTIMATE);
            FFTW(execute)(p_I);	

            /*Compute the configuration aab if b<a/2*/
            if(b < floor(a/2)){
                /*Compute the sum of the grids in real space and save the unormalized Bispectrum and the number of triangles*/
                for(l=0;l<NBs;l++)
                    B[l][count_tri] = 0.0;
                IB[count_tri] = 0.0;
                for(i=0;i<nd;i++)
                    for(j=0;j<nd;j++)
                        for(k=0;k<nd;k++){
                            tmp = (size_t) i*nd*nd + j*nd + k;
                            
                            for(f1=0;f1<ntype;f1++)
                                for(f2=0;f2<ntype;f2++)
                                    for(f3=0;f3<ntype;f3++){
                                        ind = f1*ntype*ntype + f2*ntype + f3;
                                        B[ind][count_tri] += (long double) out_Bm1[f1][tmp]*out_Bm1[f2][tmp]*out_Bm2[f3][tmp];
                                    }
                            IB[count_tri] += (long double) out_I1[tmp]*out_I1[tmp]*out_I2[tmp];
                        }
                K1[count_tri] = (long double) km1;
                K2[count_tri] = (long double) km1;
                K3[count_tri] = (long double) km2;
                count_tri ++;

                continue;
            }
                
            /*Compute the density grids for the third bin of k*/
            tmpc = (int) a - (int) b - 1;
            if(tmpc < 0) tmpc = 0;
            for(c=(int) tmpc;c<b;c++){
                km3= 0.0;
                countk3 = 0;

                #pragma omp parallel for private(i, j, k, l, kx, ky, kz, kmod, ind, tmp) reduction(+:km3,countk3)
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
                    
                            if(ind == c){
                                km3 += kmod;
                                countk3 += 1;

                                for(l=0;l<ntype;l++){
                                    in_Bm[l][tmp][0] = gridk[l][tmp][0];
                                    in_Bm[l][tmp][1] = gridk[l][tmp][1];
                                }

                                in_I[tmp][0] = 1.0;
                                in_I[tmp][1] = 0.0;
                            }
                            else{
                                for(l=0;l<ntype;l++){
                                    in_Bm[l][tmp][0] = 0.0;
                                    in_Bm[l][tmp][1] = 0.0;
                                }                           
        
                                in_I[tmp][0] = 0.0;
                                in_I[tmp][1] = 0.0;
                            }				
                        }
                    }
                }
                km3 = km3/countk3;

                /*Compute the FFTs*/
                for(l=0;l<ntype;l++){
                    p_Bm = FFTW(plan_dft_c2r_3d)(nd, nd, nd, in_Bm[l], out_Bm3[l], FFTW_ESTIMATE); 
                    FFTW(execute)(p_Bm);
                }
                p_I = FFTW(plan_dft_c2r_3d)(nd, nd, nd, in_I, out_I3, FFTW_ESTIMATE);
                FFTW(execute)(p_I);

                /*Compute the sum of the grids in real space and save the unormalized Bispectrum and the number of triangles*/
                for(l=0;l<NBs;l++)
                    B[l][count_tri] = 0.0;
                IB[count_tri] = 0.0;
                for(i=0;i<nd;i++)
                    for(j=0;j<nd;j++)
                        for(k=0;k<nd;k++){
                            tmp = (size_t) i*nd*nd + j*nd + k;
                            
                            for(f1=0;f1<ntype;f1++)
                                for(f2=0;f2<ntype;f2++)
                                    for(f3=0;f3<ntype;f3++){
                                        ind = f1*ntype*ntype + f2*ntype + f3;
                                        B[ind][count_tri] += (long double) out_Bm1[f1][tmp]*out_Bm2[f2][tmp]*out_Bm3[f3][tmp];
                                    }
                            IB[count_tri] += (long double) out_I1[tmp]*out_I2[tmp]*out_I3[tmp];
                        }
                K1[count_tri] = (long double) km1;
                K2[count_tri] = (long double) km2;
                K3[count_tri] = (long double) km3;
                count_tri ++;
            }/*Close the loop in the third field*/

            /*Compute the sum of the grids in real space and save the unormalized Bispectrum and the number of triangles*/
            for(l=0;l<NBs;l++){
                B[l][count_tri] = 0.0;
                B[l][count_tri+1] = 0.0;
            }
            IB[count_tri] = 0.0;
            IB[count_tri+1] = 0.0;
            for(i=0;i<nd;i++)
                for(j=0;j<nd;j++)
                    for(k=0;k<nd;k++){
                        tmp = (size_t) i*nd*nd + j*nd + k;
                        
                        for(f1=0;f1<ntype;f1++)
                            for(f2=0;f2<ntype;f2++)
                                for(f3=0;f3<ntype;f3++){
                                    ind = f1*ntype*ntype + f2*ntype + f3;
                                    B[ind][count_tri] += (long double) out_Bm1[f1][tmp]*out_Bm2[f2][tmp]*out_Bm2[f3][tmp];
                                    B[ind][count_tri+1] += (long double) out_Bm1[f1][tmp]*out_Bm1[f2][tmp]*out_Bm2[f3][tmp];
                                }
                        IB[count_tri] += (long double) out_I1[tmp]*out_I2[tmp]*out_I2[tmp];
                        IB[count_tri+1] += (long double) out_I1[tmp]*out_I1[tmp]*out_I2[tmp];
                    }
            K1[count_tri] = (long double) km1;
            K2[count_tri] = (long double) km2;
            K3[count_tri] = (long double) km2;
            count_tri ++;
            K1[count_tri] = (long double) km1;
            K2[count_tri] = (long double) km1;
            K3[count_tri] = (long double) km2;
            count_tri ++;
        }/*Close the loop in the second field*/

        /*Compute the sum of the grids in real space and save the unormalized Bispectrum and the number of triangles*/
        for(l=0;l<NBs;l++)
            B[l][count_tri] = 0.0;
        for(l=0;l<NPs;l++)
            P[l][count_pk] = 0.0;
        IB[count_tri] = 0.0;
        IP[count_pk] = 0.0;
        for(i=0;i<nd;i++)
            for(j=0;j<nd;j++)
                for(k=0;k<nd;k++){
                    tmp = (size_t) i*nd*nd + j*nd + k;
                    
                    for(f1=0;f1<ntype;f1++)
                        for(f2=0;f2<ntype;f2++)
                            for(f3=0;f3<ntype;f3++){
                                ind = f1*ntype*ntype + f2*ntype + f3;
                                B[ind][count_tri] += (long double) out_Bm1[f1][tmp]*out_Bm1[f2][tmp]*out_Bm1[f3][tmp];
                            }
                    IB[count_tri] += (long double) out_I1[tmp]*out_I1[tmp]*out_I1[tmp];

                    ind = 0;
                    for(f1=0;f1<ntype;f1++)
                        for(f2=0;f2<=f1;f2++){
                            P[ind][count_pk] += (long double) out_Bm1[f1][tmp]*out_Bm1[f2][tmp];
                            ind ++;
                        }
                    IP[count_pk] += (long double) out_I1[tmp]*out_I1[tmp];
                }
        K1[count_tri] = (long double) km1;
        K2[count_tri] = (long double) km1;
        K3[count_tri] = (long double) km1;
        count_tri ++;
        KP[count_pk] = (long double) km1;
        count_pk ++;
    }/*Close the loop in the first field*/

    if(verbose == TRUE)
        printf("%d bispectra computed!\n", count_tri);

    /*Free the FFTW3 arrays*/
    for(i=0;i<ntype;i++){
        FFTW(free)(in_Bm[i]);
        FFTW(free)(out_Bm1[i]);     
        FFTW(free)(out_Bm2[i]); 
        FFTW(free)(out_Bm3[i]);
        FFTW(free)(gridk[i]);
    }
    FFTW(free)(in_Bm); FFTW(free)(in_I);
    FFTW(free)(out_Bm1); FFTW(free)(out_Bm2); FFTW(free)(out_Bm3); FFTW(free)(out_I1); FFTW(free)(out_I2); FFTW(free)(out_I3);
    FFTW(free)(p_Bm); FFTW(free)(p_I);
    FFTW(free)(gridk);

    /*Fill Compute the normalized quantities*/
    for(i=0;i<count_tri;i++){
        for(j=0;j<NBs;j++)
            B[j][i] = B[j][i]/IB[i]*pow(L, 3.0/2.0);
        IB[i] = IB[i]/ng;
    }

    for(i=0;i<count_pk;i++){
        for(j=0;j<NPs;j++)
            P[j][i] = P[j][i]/IP[i];
        IP[i] = IP[i]/ng;
    }

    return count_tri;
}
