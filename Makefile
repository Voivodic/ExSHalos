EXEC = ExSHalosLC4
EXEC2 = Stack_Snaps_hdf5

#Compiler
CC = gcc

#FFTW3 
FFTW_LIBR   = -L$(HOME)/local/lib 
FFTW_INCL   = -I$(HOME)/local/include
FFTW_FLAGS  = -lfftw3f -lfftw3

#GSL
GSL_LIBR    = -L$(HOME)/local/lib 
GSL_INCL    = -I$(HOME)/local/include
GSL_FLAGS   = -lgsl -lgslcblas

#HDF5
HDF5_LIBR    = -L$(HOME)/local/lib 
HDF5_INCL    = -I$(HOME)/local/include
HDF5_FLAGS   = -lhdf5

#FFTlog
FFTLOG_LIBR = FFTLog-master/fftlog.o
FFTLOG_INCL = -IFFTLog-master/include/

FLAGS = $(FFTW_FLAGS) $(GSL_FLAGS) -lm -fopenmp #-ggdb -Wall -Wno-unknown-pragmas -O3 -g -mtune=native

ExSHalos: ExSHalosLC4.c Stack_Snaps_hdf5.c
	$(CC) ExSHalosLC4.c -o $(EXEC) $(FFTW_LIBR) $(FFTW_INCL) $(GSL_LIBR) $(GSL_INCL) $(FFTLOG_LIBR) $(FFTLOG_INCL) $(FLAGS)
	$(CC) Stack_Snaps_hdf5.c -o $(EXEC2) -DH5_USE_16_API $(HDF5_LIBR) $(HDF5_INCL) $(HDF5_FLAGS)

clean:
	\rm -f $(EXEC)
