#include "box.h"

/*Generate a halo catalogue in a box from a given linear power spectrum*/
size_t Generate_Halos_Box_from_Pk(fft_real *K, fft_real *P, int Nk, fft_real R_max, fft_real k_smooth, HALOS **halos, fft_real **posh, fft_real **velh, long *flag, fft_real *delta, fft_real *S, fft_real *V, int fixed, fft_real phase){
    int i, j;
    size_t nh;
    fft_complex *deltak;

    /*Alloc the grids*/
    if(out.OUT_DEN == FALSE){
        delta = (fft_real *) malloc(box.ng*sizeof(fft_real));
        check_memory(delta, "delta")
    }
    deltak = (fft_complex *) FFTW(malloc)(((size_t) box.nd[0])*((size_t) box.nd[1])*((size_t) box.nz2)*sizeof(fft_complex));
    check_memory(deltak, "deltak")

    /*Compute the density grids*/
    Compute_Den(K, P, Nk, R_max, delta, deltak, fixed, phase, 100000.0);

    /*Find the halos in the density grid*/
    nh = Find_Halos(delta, K, P, Nk, flag, halos);

    if(out.OUT_DEN == FALSE)
        free(delta);

    /*Alloc the arrays with the positions and velocities of the halos*/
    *posh = (fft_real *)malloc(3*nh*sizeof(fft_real));
    if(out.OUT_VEL == TRUE)
	    *velh = (fft_real *)malloc(3*nh*sizeof(fft_real));
	for(i=0;i<nh;i++)
		for(j=0;j<3;j++){
			(*posh)[(size_t) 3*i+j] = 0.0;
            if(out.OUT_VEL == TRUE)
			    (*velh)[(size_t) 3*i+j] = 0.0;
		}

    /*Alloc the array for the displacements*/
    if(out.DO_2LPT == TRUE && out.OUT_LPT == FALSE)
        S = (fft_real *) malloc(3*box.ng*sizeof(fft_real));

    /*Compute the LPT*/
    if(out.DO_2LPT == TRUE){
        Compute_1LPT(deltak, *posh, *velh, S, V, flag, k_smooth);
        FFTW(free)(deltak);

        Compute_2LPT(*posh, *velh, S, V, flag, k_smooth);

        if(out.OUT_LPT == FALSE)
            free(S);
    }
    else{
        if(out.OUT_LPT == TRUE)
            Compute_1LPT(deltak, *posh, *velh, S, V, flag, k_smooth);
        else
            Compute_1LPT(deltak, *posh, *velh, NULL, NULL, flag, k_smooth);

        FFTW(free)(deltak);
    }

    /*Compute the final position and velocity of each halo*/
    Compute_Posh(*halos, *posh, *velh, nh);

    /*Compute the final position of the particles*/
    if(out.OUT_LPT == TRUE)
        Compute_Pos(S);

    return nh;
}

/*Generate a halo catalogue in a box from a given density grid*/
size_t Generate_Halos_Box_from_Grid(fft_real *K, fft_real *P, int Nk, fft_real k_smooth, HALOS **halos, fft_real **posh, fft_real **velh, long *flag, fft_real *delta, fft_real *S, fft_real *V, int IN_disp){
    int i, j;
    size_t ind, nh;
    fft_complex *deltak;

    /*Find the halos in the density grid*/
    nh = Find_Halos(delta, K, P, Nk, flag, halos);

    /*Alloc the arrays with the positions and velocities of the halos*/
    *posh = (fft_real *)malloc(3*nh*sizeof(fft_real));
    if(out.OUT_VEL == TRUE)
        *velh = (fft_real *)malloc(3*nh*sizeof(fft_real));
    for(i=0;i<nh;i++)
        for(j=0;j<3;j++){
            (*posh)[(size_t) 3*i+j] = 0.0;
            if(out.OUT_VEL == TRUE)
                (*velh)[(size_t) 3*i+j] = 0.0;
        }

    /*Attribute the inputed displacements to the halos*/
    if(IN_disp == TRUE){
        for(ind=0;ind<box.ng;ind++)
            if(flag[ind] < (long) box.ng){
                (*posh)[3*flag[ind]] += S[3*ind];
                (*posh)[3*flag[ind] + 1] += S[3*ind + 1];
                (*posh)[3*flag[ind] + 2] += S[3*ind + 2];

                if(out.OUT_VEL == TRUE){
                    (*velh)[3*flag[ind]] += V[3*ind];
                    (*velh)[3*flag[ind] + 1] += V[3*ind + 1];
                    (*velh)[3*flag[ind] + 2] += V[3*ind + 2];
                }
            }
    }

    /*Compute the displacements and attribute to the halos*/
    else{
        /*Alloc the grids*/
        deltak = (fft_complex *) FFTW(malloc)(((size_t) box.nd[0])*((size_t) box.nd[1])*((size_t) box.nz2)*sizeof(fft_complex));
        check_memory(deltak, "deltak")

        /*Compute the density grid in Forier space*/
        Compute_Denk(delta, deltak);

        /*Alloc the array for the displacements*/
        if(out.DO_2LPT == TRUE || out.OUT_LPT == FALSE)
            S = (fft_real *) malloc(3*box.ng*sizeof(fft_real));

        /*Compute the LPT*/
        if(out.DO_2LPT == TRUE){
            Compute_1LPT(deltak, *posh, *velh, S, V, flag, k_smooth);
            FFTW(free)(deltak);

            Compute_2LPT(*posh, *velh, S, V, flag, k_smooth);

            if(out.OUT_LPT == FALSE)
                free(S);
        }
        else{
            if(out.OUT_LPT == TRUE)
                Compute_1LPT(deltak, *posh, *velh, S, V, flag, k_smooth);
            else
                Compute_1LPT(deltak, *posh, *velh, NULL, NULL, flag, k_smooth);

            FFTW(free)(deltak);
        }

        /*Compute the final position of the particles*/
        if(out.OUT_LPT == TRUE)
            Compute_Pos(S);
    }

    /*Compute the final position and velocity of each halo*/
    Compute_Posh(*halos, *posh, *velh, nh);

    return nh;
}