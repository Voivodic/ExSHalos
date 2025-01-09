#include "spectrum_h.h"

/*Evaluate the ciclic sum of x and y*/
int mod(int x, int y, int nd){
	int resp;

	resp = x + y;
    while(resp >= nd)
        resp -= nd;
    while(resp < 0)
        resp += nd;

	return resp;
}

/*Define the cyclic sum for floats*/
fft_real cysumf(fft_real x, fft_real y, fft_real L){
	fft_real resp;

	resp = x + y;
    while(resp >= L)
        resp -= L;
    while(resp < 0.0)
        resp += L;

	return resp;
}

/*Define de sinc function*/
fft_real sinc(fft_real x){
	fft_real resp;
	if(x!=0.0) resp = sin(x)/x;
	else resp = 1.0;	

	return resp;
}

/*Define the bin for the mode*/
int Indice(fft_real k, fft_real kmin, fft_real dk){
	int resp;
	fft_real tmp;

	tmp = (k - kmin)/dk;

	resp = floor(tmp);

	return resp;
}

/*Define window function for NGP and CIC*/
fft_real W(fft_real k1, fft_real k2, fft_real k3, fft_real Lb, fft_real R, int window){
	fft_real resp, kmod;

	if(window == 0)
		resp = 1.0;
	if(window == 1)
		resp = sinc(k1*Lb/2.0)*sinc(k2*Lb/2.0)*sinc(k3*Lb/2.0);
	if(window == 2)
		resp = pow(sinc(k1*Lb/2.0)*sinc(k2*Lb/2.0)*sinc(k3*Lb/2.0), 2.0);
	if(window == 3){
		kmod = sqrt(k1*k1 + k2*k2 + k3*k3);	
		resp = 3.0/kmod/kmod/R/R*(sinc(kmod*R) - cos(kmod*R));
	}
	if(window == 4){
		kmod = k1*k1 + k2*k2 + k3*k3;	
		resp = exp(-kmod*R*R/2.0);
	}

	return resp;
}

/*Give a indice for the partricle*/
void ind(fft_real x[], int xt[], fft_real Ld, int nd){
	xt[0] = floor(x[0]/Ld);
	xt[1] = floor(x[1]/Ld);
	xt[2] = floor(x[2]/Ld);
	if(xt[0]==nd)	xt[0] -=1;
	if(xt[1]==nd)	xt[1] -=1;
	if(xt[2]==nd)	xt[2] -=1;
}

/*Compute the density grids in Fourier space corrected by the interlacing and window function*/
void Compute_gridk(fft_real *grid, fft_complex **gridk, int nd, fft_real L, int ntype, int interlacing, fft_real R, int window){
    int i, j, k, l;
    fft_real kx, ky, kz, kn, Lb, Normk;
    size_t ng, tmp;
    fft_complex *out1, *out2;
    FFTW(plan) p1, p2;  

    Normk = pow(L/(nd*nd), 3.0/2.0);    /*Normalization for the Fourier transforms (from x to k)*/
    ng = (size_t) nd*nd*nd;   /*Number of grid cells*/
    kn = 2*M_PI/L;  /*Fundamental frequency*/
    Lb = L/nd;      /*Size of each cell*/

    /*Alloc the temporary arrays*/
    out1 = (fft_complex*) FFTW(malloc)(nd*nd*(nd/2+1)*sizeof(fft_complex));
    if (interlacing == TRUE)
        out2 = (fft_complex*) FFTW(malloc)(nd*nd*(nd/2+1)*sizeof(fft_complex));

    for(l=0;l<ntype;l++){
        /*Create fft plans*/
        if (interlacing == TRUE){
            p1 = FFTW(plan_dft_r2c_3d)(nd, nd, nd, &grid[2*((size_t) l)*ng], out1, FFTW_ESTIMATE); 
            p2 = FFTW(plan_dft_r2c_3d)(nd, nd, nd, &grid[(2*((size_t) l) + 1)*ng], out2, FFTW_ESTIMATE);
        }
        else
            p1 = FFTW(plan_dft_r2c_3d)(nd, nd, nd, &grid[ng*((size_t) l)], out1, FFTW_ESTIMATE); 

        /*Compute the FFT of the grids*/
        FFTW(execute)(p1);

        if (interlacing == TRUE){
            FFTW(execute)(p2);

            /*Take the mean over the two grids*/
			#pragma omp parallel for private(i, j, k, kx, ky, kz, tmp) 
            for(i=0;i<nd;i++){
                if(2*i<nd) kx = i*kn;
                else kx = (i-nd)*kn;

                for(j=0;j<nd;j++){
                    if(2*j<nd) ky = j*kn;
                    else ky = (j-nd)*kn;

                    for(k=0;k<nd/2+1;k++){
                        if(2*k<nd) kz = k*kn;
                        else kz = (k-nd)*kn;

                        tmp = nd*(nd/2 + 1)*i + (nd/2 + 1)*j + k;
                        gridk[l][tmp][0] = (out1[tmp][0] + out2[tmp][0]*cos((kx + ky + kz)*Lb/2.0) + out2[tmp][1]*sin((kx + ky + kz)*Lb/2.0))/2.0*Normk/W(kx, ky, kz, Lb, R, window);
                        gridk[l][tmp][1] = (out1[tmp][1] + out2[tmp][1]*cos((kx + ky + kz)*Lb/2.0) - out2[tmp][0]*sin((kx + ky + kz)*Lb/2.0))/2.0*Normk/W(kx, ky, kz, Lb, R, window);
                    }
                }
            }
        }

        else{
			#pragma omp parallel for private(i, j, k, kx, ky, kz, tmp)
            for(i=0;i<nd;i++){
                if(2*i<nd) kx = i*kn;
                else kx = (i-nd)*kn;

                for(j=0;j<nd;j++){
                    if(2*j<nd) ky = j*kn;
                    else ky = (j-nd)*kn;

                    for(k=0;k<nd/2+1;k++){
                        if(2*k<nd) kz = k*kn;
                        else kz = (k-nd)*kn;

                        tmp = nd*(nd/2 + 1)*i + (nd/2 + 1)*j + k;
                        gridk[l][tmp][0] = (out1[tmp][0])*Normk/W(kx, ky, kz, Lb, R, window);
                        gridk[l][tmp][1] = (out1[tmp][1])*Normk/W(kx, ky, kz, Lb, R, window);
                    }
                }
            }
        }
    }

    /*Free the FFTW stuff*/
    FFTW(free)(out1);
    FFTW(free)(p1);
    if (interlacing == TRUE){
        FFTW(free)(out2);
        FFTW(free)(p2);
    }
}