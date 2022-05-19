#include "split_galaxies.h"

/*Relative occupancy of central galaxies for each type*/
fft_real Occ_cen(fft_real log10Mh, int type){
	int i;
    fft_real resp;

	resp = 0.0;
	for(i=0;i<split.order_cen;i++)
		resp += split.params_cen[type][i]*pow(log10Mh, i);

    if(resp < 0.0)  resp = 0.0;
    if(resp > 1.0)  resp = 1.0; 

	return resp;
}

/*Relative occupancy of satellite galaxies for each type*/
fft_real Occ_sat(fft_real log10Mh, int type){
	int i;
    fft_real resp;

	resp = 0.0;
	for(i=0;i<split.order_sat;i++)
		resp += split.params_sat[type][i]*pow(log10Mh, i);

    if(resp < 0.0)  resp = 0.0;
    if(resp > 1.0)  resp = 1.0; 

	return resp;
}


/*Determine the type of each galaxy*/
void Galaxy_Types(size_t ng, fft_real *Massh, long *flag, int *type, gsl_rng *rng_ptr){
	size_t i, j;
	fft_real rnd, Occ;

	/*Run over all galaxies*/
	for(i=0;i<ng;i++){
		rnd = (fft_real) gsl_rng_uniform(rng_ptr);
		Occ = 0.0;

		/*Define the type of the satellites*/
		if(flag[i] > 0){
			for(j=0;j<split.ntypes-1;j++){
				Occ += Occ_sat(log10(Massh[flag[i]]), j);

				if(rnd <= Occ){
					type[i] = j + 1;
					break;
				}
			}	
			if(rnd > Occ)
				type[i] = split.ntypes;	
		}

		/*Define the type of the centrals*/
		else{
			for(j=0;j<split.ntypes-1;j++){
				Occ += Occ_cen(log10(Massh[-flag[i]]), j);

				if(rnd <= Occ){
					type[i] = - (j + 1);
					break;
				}
			}	
			if(rnd > Occ)
				type[i] = -split.ntypes;	
		}
	}
}
