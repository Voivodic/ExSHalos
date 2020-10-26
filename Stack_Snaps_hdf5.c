#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>
#include "hdf5.h"

#define check_memory(p, name) if(p == NULL){printf("Problems to alloc %s.\n", name); return 0;}

int main(int argc,char *argv[])
{
FILE *collapse_cat, *snap_cat;
char lightfile[100], collapsefile[100], snapfile[100];
int Nsnaps, out_collapse, i, k;
long cont, cont_tmp, j;
hid_t light_cat, datatype;      
hsize_t dimsf[2];
herr_t status;    
float *data;                         

if (argc != 4){
	printf("arg1: Prefix of the inputs and for the outputs.\n");
	printf("arg2: Number of snapshots.\n");
	printf("arg3: Stack the files with the collapsed particles? Yes (1) or No (0).\n");

	exit(0);
}

/*Read the input parameters*/
Nsnaps = atoi(argv[2]);					//Number of snapshots
sprintf(lightfile, "%s_LightCone.h5", argv[1]);	//File for the final light cone
sprintf(collapsefile, "%s_Collapse.dat", argv[1]);	//File with the final information about the collapsed particles
out_collapse = atoi(argv[3]);				//Parameter with the information about the collapsed particles

/*Create the new HDF5 file*/    
light_cat = H5Fcreate(lightfile, H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);

/*Define datatype for the data in the file*/
datatype = H5Tcopy(H5T_NATIVE_FLOAT);
status = H5Tset_order(datatype, H5T_ORDER_LE);

/*Construct the final light cone*/
cont = 0;
for(i=0;i<Nsnaps;i++){

	/*Open this snapshot*/
	sprintf(snapfile, "%s_%d_LightCone.dat", argv[1], i);
	snap_cat = fopen(snapfile, "rb");

	if (snap_cat == NULL) {
		printf("Unable to open %s\n", snapfile);
		exit(0);
	}

	fread(&cont_tmp, sizeof(long), 1, snap_cat);

	//printf("cont = %ld\n", cont_tmp);

   /*Define the handles for the hdf5 library*/
    hid_t dataset, dataspace; 

    /*Define the size of the array*/
    dimsf[0] = cont_tmp;
    dimsf[1] = 7;
    dataspace = H5Screate_simple(2, dimsf, NULL); 

    /*Create the dataset*/
    sprintf(snapfile, "Snap_%d", i);
    dataset = H5Dcreate(light_cat, snapfile, datatype, dataspace, H5P_DEFAULT);

    /*Alloc the data file*/
    data = (float *)malloc(cont_tmp*7*sizeof(float));

	/*Read the information of each halo and save it*/
	for(j=0;j<cont_tmp;j++){

		fread(&data[j*7+0], sizeof(float), 1, snap_cat);
		fread(&data[j*7+1], sizeof(float), 1, snap_cat);
		fread(&data[j*7+2], sizeof(float), 1, snap_cat);
		fread(&data[j*7+3], sizeof(float), 1, snap_cat);
		fread(&data[j*7+4], sizeof(float), 1, snap_cat);
		fread(&data[j*7+5], sizeof(float), 1, snap_cat);
		fread(&data[j*7+6], sizeof(float), 1, snap_cat);
	}
	fclose(snap_cat);

	cont += cont_tmp;

    /*Write the data in this snapshot to the dataset*/
    status = H5Dwrite(dataset, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, data);

    /*Free the data array*/
	free(data);

    /*Close some HDF5 variables*/
    H5Sclose(dataspace);
    H5Dclose(dataset);
}

/*Close some HDF5 file*/
H5Tclose(datatype);
H5Fclose(light_cat);

return 0;
}
