import exshalos
import numpy as np

#Compute the gaussian density grid given the power spectrum
def Generate_Density_Grid(k, P, R_max = 100000.0, nd = 256, ndx = None, ndy = None, ndz = None, Lc = 2.0, outk = False, seed = 12345, verbose = False, nthreads = 1):
    """
    k: Wavenumbers of the power spectrum | 1D numpy array
    P: Power spectrum | 1D numpy array
    R_max: Maximum size used to compute the correlation function in Mpc/h | float
    ndx, ndy, ndz: Number of cells in each direction | ints
    Lc: Size of each cell in Mpc/h | float
    outk: Return the density field in fourier space? | Boolean
    seed: Seed used to generate the random numbers | int
    verbose: Output or do not output information in the c code | boolean
    nthreads: Number of threads used by openmp | int

    return: The 3D density grid in real space (also in Forier if outk == True) | 3D numpy array
    """

    precision = exshalos.exshalos.exshalos.check_precision()

    if(precision == 4):
        k = k.astype("float32")
        P = P.astype("float32")
        R_max = np.float32(R_max)
        Lc = np.float32(Lc)
    else:
        k = k.astype("float64")
        P = P.astype("float64")
        R_max = np.float64(R_max)
        Lc = np.float64(Lc)  

    if(ndx is None):
        ndx = nd
    if(ndy is None):
        ndy = nd
    if(ndz is None):
        ndz = nd   

    if(outk == False):
        grid = exshalos.exshalos.exshalos.density_grid_compute(k, P, R_max, np.int32(ndx), np.int32(ndy), np.int32(ndz), Lc, np.int32(outk), np.int32(seed), np.int32(verbose), np.int32(nthreads)) 

        return grid[0] 
    else:
        (grid, gridk) = exshalos.exshalos.exshalos.density_grid_compute(k, P, R_max, np.int32(ndx), np.int32(ndy), np.int32(ndz), Lc, np.int32(outk), np.int32(seed), np.int32(verbose), np.int32(nthreads)) 

        return grid, gridk