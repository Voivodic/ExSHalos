# ExSHalos
ExSHalos code for the fast generation of halo catalogues.

This folder contains the ExSHalos code use to generate halo catalogues fast and using a few computational resources. This code is especially useful to compute covariance matrices for many observables and tracers.

This folder has the following codes:

-Makefile: A simple makefile used to compile the ExSHalos.c and Stack_Snaps_hdf5.c;

-ExSHalosLC4.c: An implementation of the ExSHalos algorithm itself;

-Stack_Snaps_hdf5.c: Auxiliary code use to stack many snapshots to generate a lightcone.

The complete list of input options for each code is given by the code when started without any input.

ExSHalos needs the external libraries: FFTW3, GSL, HDF5, and FFTLog. 

If you use any of the codes present here, please cite the paper: https://arxiv.org/abs/1906.06630.

Please, contact me if you have any question: rodrigo.voivodic@usp.br
