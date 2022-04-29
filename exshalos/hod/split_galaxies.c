#include "split_galaxies.h"

/*Relative occupancy of central red galaxies*/
fft_real Occ_red_cen(fft_real Mh){
    fft_real resp;

    resp = split.C3*pow(log10(Mh), 3.0) + split.C2*pow(log10(Mh), 2.0) + split.C1*log10(Mh) + split.C0;

    if(resp < 0.0)  resp = 0.0;
    if(resp > 1.0)  resp = 1.0; 

	return resp;
}

/*Relative occupancy of central blue galaxies*/
fft_real Occ_blue_cen(fft_real Mh){
	return 1.0 - Occ_red_cen(Mh);
}

/*Relative occupancy of satellite red galaxies*/
fft_real Occ_red_sat(fft_real Mh){
    fft_real resp;

    resp = split.S3*pow(log10(Mh), 3.0) + split.S2*pow(log10(Mh), 2.0) + split.S1*log10(Mh) + split.S0;

    if(resp < 0.0)  resp = 0.0;
    if(resp > 1.0)  resp = 1.0; 

	return resp;
}

/*Relative occupancy of satellite blue galaxies*/
fft_real Occ_blue_sat(fft_real Mh){
	return  1.0 - Occ_red_sat(Mh);
}

/*Determine the type of each galaxy*/
void Galaxy_Types(size_t ng, fft_real *Massh, long *flag, int *type, gsl_rng *rng_ptr){
	size_t i;
	fft_real rnd;

	/*Run over all galaxies*/
	for(i=0;i<ng;i++){
		rnd = (fft_real) gsl_rng_uniform(rng_ptr);

		/*Define the type of the satellites*/
		if(flag[i] > 0){
			if(rnd < Occ_blue_sat(Massh[flag[i]]))
				type[i] = 1;
			else
				type[i] = 2;
		}
		/*Define the type of the centrals*/
		else{
			if(rnd < Occ_blue_cen(Massh[-flag[i]]))
				type[i] = -1;
			else
				type[i] = -2;
		}
	}
}
