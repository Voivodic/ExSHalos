import exshalos
import numpy as np
from scipy.optimize import minimize
from scipy.interpolate import interp1d

#Compute the gaussian density grid given the power spectrum
def Generate_Density_Grid(k, P, R_max = 100000.0, nd = 256, ndx = None, ndy = None, ndz = None, Lc = 2.0, outk = False, seed = 12345, fixed = False, phase = 0.0, k_smooth = 100000.0, verbose = False, nthreads = 1):
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
        phase = np.float32(phase)
        k_smooth = np.float32(k_smooth)
    else:
        k = k.astype("float64")
        P = P.astype("float64")
        R_max = np.float64(R_max)
        Lc = np.float64(Lc)  
        phase = np.float64(phase)
        k_smooth = np.float64(k_smooth)


    if(ndx is None):
        ndx = nd
    if(ndy is None):
        ndy = nd
    if(ndz is None):
        ndz = nd   

    x = exshalos.exshalos.exshalos.density_grid_compute(k, P, R_max, np.int32(ndx), np.int32(ndy), np.int32(ndz), Lc, np.int32(outk), np.int32(seed), np.int32(fixed), phase, k_smooth, np.int32(verbose), np.int32(nthreads)) 

    return x

#Generate a halo catalogue (in Lagrangian space) given an initial density grid
def Find_Halos_from_Grid(grid, k, P, Lc = 2.0, Om0 = 0.31, z = 0.0, delta_c = -1.0, Nmin = 10, a = 1.0, beta = 0.0, alpha = 0.0, verbose = False):
    """
    grid: Density grid where the halos will be find | 3D numpy array (Ndx, Ndy, Ndz)
    k: Wavenumbers of the power spectrum | 1D numpy array
    P: Power spectrum | 1D numpy array
    Lc: Size of each cell in Mpc/h | float
    Om0: Value of the matter overdensity today | float
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

    x = exshalos.exshalos.exshalos.find_halos(grid, k, P, Lc, Om0, z, delta_c, np.int32(Nmin), a, beta, alpha, verbose)

    return x


#Compute the positions and velocities of particles given a grid using LPT
def Displace_LPT(grid, Lc = 2.0, Om0 = 0.31, z = 0.0, k_smooth = 10000.0, DO_2LPT = False, OUT_VEL = False, Input_k = False, OUT_POS = True, verbose = False):
    """
    grid: Density grid where the halos will be find | 3D numpy array (Ndx, Ndy, Ndz)
    Lc: Size of each cell in Mpc/h | float
    z: Redshift of the density grid and final halo catalogue | float
    k_smooth: Scale used to smooth the displacements | float
    DO_2LPT: Use or do not the second order | boolean
    OUT_VEL: Output or do not the velocities of the particles | boolean
    Inpute_k: The density grid is in real or Fourier space | boolean
    OUT_POS: Output the position or just the displacements | boolean
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

    x = exshalos.exshalos.exshalos.lpt_compute(grid, Lc, Om0, z, k_smooth, np.int32(DO_2LPT), np.int32(OUT_VEL), np.int32(Input_k), np.int32(OUT_POS), np.int32(verbose))

    return x

#Fit the parameters of the barrier given a mass function
def Fit_Barrier(k, P, M, dndlnM, grid = None, R_max = 100000.0, Mmin = -1.0, Mmax = -1.0, Nm = 25, nd = 256, Lc = 2.0, Om0 = 0.31, z = 0.0, delta_c = -1.0, Nmin = 10, seed = 12345, x0 = None, verbose = False, nthreads = 1, Max_inter = 100, tol = None):
    """
    k: Wavenumbers of the power spectrum | 1D numpy array
    P: Power spectrum | 1D numpy array
    R_max: Maximum size used to compute the correlation function in Mpc/h | float
    Mmin: Minimum mass used to construct the mass bins | float
    Mmax: Maximum mass used to construct the mass bins | float
    Nm: Number of mass bins | int
    nd: Number of cells in each direction | ints
    Lc: Size of each cell in Mpc/h | float
    Om0: Value of the matter overdensity today | float    
    z: Redshift of the density grid and final halo catalogue | float
    delta_c: Critical density of the halo formation linearly extrapolated to z | float
    Nmin: Minimum number of particles in each halo | int
    seed: Seed used to generate the random numbers | int
    verbose: Output or do not output information in the c code | boolean
    nthreads: Number of threads used by openmp | int
    Max_inter: Maximum number of interations used in the minimization | int

    return: The values of the parameters of the ellipsoidal barrier | 3 x float
    """

    #Interpolate the given mass function
    fdn = interp1d(np.log(M), dndlnM)

    #Construct the gaussian density grid
    if(grid == None):
        grid = Generate_Density_Grid(k, P, R_max, nd = nd, Lc = Lc, seed = seed, verbose = verbose, nthreads = nthreads)

    #Define the function to be minimized to find the best parameters of the barrier
    def Chi2(theta):
        a, beta, alpha = theta

        x = exshalos.utils.Find_Halos_from_Grid(grid["grid"], k, P, Lc = Lc, Om0 = Om0, z = z, delta_c = delta_c, Nmin = Nmin, a = a, beta = beta, alpha = alpha, verbose = verbose)

        dnh = exshalos.simulation.Compute_Abundance(x["Mh"], Mmin = Mmin, Mmax = Mmax, Nm = Nm, Lc = Lc, nd = nd, verbose = verbose)

        mask = dnh["dn"] > 0.0
        chi2 = np.sum(np.power((dnh["dn"][mask] - fdn(np.log(dnh["Mh"][mask])))/dnh["dn_err"][mask], 2.0))/(Nm - 4)
        print("Current try: (%f, %f, %f) with chi2 = %f" %(a, beta, alpha, chi2))

        return chi2

    #Define the inital position
    if(x0 is None):
        x0 = [0.55, 0.4, 0.7]#np.random.random(3)*[2.0, 1.0, 1.0]

    #Minimaze the Chi2 to get the best fit parameters
    bounds = [[0.0, 2.0], [0.0, 1.0], [0.0, 1.0]]
    x = minimize(Chi2, x0 = x0, bounds = bounds, method = "Nelder-Mead", options = {"maxiter" : Max_inter}, tol = tol)

    return x

#Fit the parameters of the HOD
def Fit_HOD(k, P, nbar = None, posh = None, Mh = None, velh = None, Ch = None, nd = 256, ndx = None, ndy = None, ndz = None, Lc = 2.0, Om0 = 0.31, z = 0.0, x0 = None, sigma = 0.5, Deltah = -1.0, seed = 12345, USE_VEL = False, l_max = 0, direction = "z", window = "cic", R = 4.0, R_times = 5.0, interlacing = True, Nk = 25, k_min = None, k_max = 0.3, verbose = False, nthreads = 1, Max_inter = 100, tol = None):
    """
    posh: Positions of the halos| 2D array (Nh, 3)
    velh: Velocities of the halos| 2D array (Nh, 3)
    Mh: Mass of the halos | 1D array (Nh)
    Ch: Concentration of the halos | 1D array (Nh)
    nd or ndx, ndy, ndz: Number of cells in each direction | ints
    Lc: Size of each cell in Mpc/h | float
    Om0: Value of the matter overdensity today | float
    z: Redshift of the density grid and final halo catalogue | float
    x0: Itial guess for the best fit parameters of the hod | 1D array (5)
    sigma: Parameter of the exclusion term of the halo density profile (Voivodic 2020) | float
    Deltah: Overdensity of the halos | float
    seed: Seed used to generate the density field | int
    USE_VEL: Use the power spectrum in redshift space | boolean
    verbose: Output or do not output information in the c code | boolean

    return: Return the 5 best fit parameters of the HOD | 1D numpy array (5)
    """

    #Interpolate the given power spectrum
    fP = interp1d(k, P)

    #Define the function to be minimized
    def Chi2(theta):
        logMmin, siglogM, logM0, logM1, alpha = theta

        gals = exshalos.mock.Generate_Galaxies_from_Halos(posh, Mh, velh = velh, Ch = Ch, nd = nd, ndx = ndx, ndy = ndy, ndz = ndz, Lc = Lc, Om0 = Om0, z = z, logMmin = logMmin, siglogM = siglogM, logM0 = logM0, logM1 = logM1, alpha = alpha, sigma = sigma, Deltah = Deltah, seed = seed, OUT_VEL = USE_VEL, OUT_FLAG = False, verbose = verbose)

        if(USE_VEL == True):
            grid = exshalos.simulation.Compute_Density_Grid(gals["posg"], vel = gals["velg"], mass = None, type = None, nd = nd, L = nd*Lc, Om0 = Om0, z = z, direction = direction, window = "CIC", R = R, R_times = R_times, interlacing = interlacing, verbose = verbose, nthreads = nthreads)
        else:
            grid = exshalos.simulation.Compute_Density_Grid(gals["posg"], vel = None, mass = None, type = None, nd = nd, L = nd*Lc, Om0 = Om0, z = z, direction = None, window = "CIC", R = R, R_times = R_times, interlacing = interlacing, verbose = verbose, nthreads = nthreads)

        Pk = exshalos.simulation.Compute_Power_Spectrum(grid, L = nd*Lc, window = window, R = R, Nk = Nk, k_min = k_min, k_max = k_max, l_max = l_max, verbose = verbose, nthreads = nthreads, ntype = 1, direction = direction)

        if(nbar is None):
            chi2 = (np.sum(np.power((Pk["Pk"] - fP(Pk["k"]))/(Pk["Pk"]/Pk["Nk"]), 2.0)))/(Nk - 6)
        else:
            chi2 = (np.sum(np.power((Pk["Pk"] - fP(Pk["k"]))/(Pk["Pk"]/Pk["Nk"]), 2.0)) + np.power((len(gals["posg"]) - nbar*(Lc*nd)**3)/len(gals["posg"]), 2.0))/(Nk - 6)            

        return chi2

    #Define the inital position
    if(x0 is None):
        x0 = [13.25424743, 0.26461332, 13.28383025, 14.32465146, 1.00811277]

    #Minimaze the Chi2 to get the best fit parameters
    bounds = [[9.0, 15.0], [0.0, 1.0], [9.0, 15.0], [9.0, 15.0], [0.0, 2.0]]
    x = minimize(Chi2, x0 = x0, bounds = bounds, method = "Nelder-Mead", options = {"maxiter" : Max_inter}, tol = tol)
    

    return x
