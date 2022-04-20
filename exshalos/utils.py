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

#Generate a halo catalogue (in Lagrangian space) given an initial density grid
def Find_Halos_from_Grid(grid, k, P, Lc = 2.0, Om0 = 0.31, z = 0.0, delta_c = 1.686, Nmin = 10, a = 1.0, beta = 0.0, alpha = 0.0, verbose = False):
    """
    grid: Density grid where the halos will be find | 3D numpy array (Ndx, Ndy, Ndz)
    k: Wavenumbers of the power spectrum | 1D numpy array
    P: Power spectrum | 1D numpy array
    Lc: Size of each cell in Mpc/h | float
    z: Redshift of the density grid and final halo catalogue | float
    delta_c: Critical density of the halo formation linearly extrapolated to z | float
    Nmin: Minimum number of particles in each halo | int
    a: a parameter of the ellipsoidal barrier | float
    beta: beta parameter of the ellipsoidal barrier | float
    alpha: alpha parameter of the ellipsoidal barrier | float
    verbose: Output or do not output information in the c code | boolean

    return: List with the positions and mass of each halo | 2D tuple with: 2D numpy array with the position of each halo (Nh x 3), 1D numpy arry with the mass of each halo (Nh)
    """

    precision = exshalos.exshalos.exshalos.check_precision()

    if(precision == 4):
        k = k.astype("float32")
        P = P.astype("float32")
        Lc = np.float32(Lc)
        z = np.float32(z)
        delta_c = np.float32(delta_c)
        a = np.float32(a)
        beta = np.float32(beta)
        alpha = np.float32(alpha)
    else:
        k = k.astype("float64")
        P = P.astype("float64")
        Lc = np.float64(Lc)  
        z = np.float64(z)
        delta_c = np.float64(delta_c)
        a = np.float64(a)
        beta = np.float64(beta)
        alpha = np.float64(alpha)

    (posh, Mh) = exshalos.exshalos.exshalos.find_halos(grid, k, P, Lc, Om0, z, delta_c, np.int32(Nmin), a, beta, alpha, verbose)

    return posh, Mh


#Compute the positions and velocities of particles given a grid using LPT
def Displace_LPT(grid, Lc = 2.0, Om0 = 0.31, z = 0.0, k_smooth = 0.15, DO_2LPT = False, OUT_vel = False, Input_k = False, verbose = False):
    """
    grid: Density grid where the halos will be find | 3D numpy array (Ndx, Ndy, Ndz)
    Lc: Size of each cell in Mpc/h | float
    z: Redshift of the density grid and final halo catalogue | float
    k_smooth: Scale used to smooth the displacements | float
    DO_2LPT: Use or do not the second order | boolean
    OUT_vel: Output or do not the velocities of the particles | boolean
    Inpute_k: The density grid is in real or Fourier space | boolean
    verbose: Output or do not output information in the c code | boolean

    return: Position and velocities (if ordered) of all particles in the grid | 2D (or 1D) tuple with the positions and velocies of all particles (Np x 3) + (Np x 3)
    """

    precision = exshalos.exshalos.exshalos.check_precision()

    if(precision == 4):
        grid = grid.astype("float32")
        Lc = np.float32(Lc)
        z = np.float32(z)
        k_smooth = np.float32(k_smooth)

    else:
        grid = grid.astype("float32")
        Lc = np.float64(Lc)  
        z = np.float64(z)
        k_smooth = np.float64(k_smooth)

    if(OUT_vel == False):
        S = exshalos.exshalos.exshalos.lpt_compute(grid, Lc, Om0, z, k_smooth, np.int32(DO_2LPT), np.int32(OUT_vel), np.int32(Input_k), np.int32(verbose))

        return S[0]
    else:
        (S, V) = exshalos.exshalos.exshalos.lpt_compute(grid, Lc, Om0, z, k_smooth, np.int32(DO_2LPT), np.int32(OUT_vel), np.int32(Input_k), np.int32(verbose))

        return S, V
