import exshalos
import numpy as np

#Generate a halo catalogue from a linear power spectrum
def Generate_Halos_Box_from_Pk(k, P, R_max = 100000.0, nd = 256, ndx = None, ndy = None, ndz = None, Lc = 2.0, Om0 = 0.31, z = 0.0, k_smooth = 10000.0, delta_c = -1.0, Nmin = 1, a = 1.0, beta = 0.0, alpha = 0.0, seed = None, fixed = False, phase = 0.0, OUT_DEN = False, OUT_LPT = False, OUT_VEL = False, DO_2LPT = False, OUT_FLAG = False, verbose = False, nthreads = 1):
    """
    k: Wavenumbers of the power spectrum | 1D numpy array
    P: Power spectrum | 1D numpy array
    R_max: Maximum size used to compute the correlation function in Mpc/h | float
    nd or ndx, ndy, ndz: Number of cells in each direction | ints
    Lc: Size of each cell in Mpc/h | float
    Om0: Value of the matter overdensity today | float
    z: Redshift of the density grid and final halo catalogue | float
    k_smooth: Scale used to smooth the LPT computations | float
    delta_c: Critical density of the halo formation linearly extrapolated to z | float
    Nmin: Minimum number of particles in each halo | int
    a: a parameter of the ellipsoidal barrier | float
    beta: beta parameter of the ellipsoidal barrier | float
    alpha: alpha parameter of the ellipsoidal barrier | float
    seed: Seed used to generate the density field | int
    OUT_DEN: Output the density field used | boolean
    OUT_LPT: Output the dispaced particles | boolean
    OUT_VEL: Output the velocities of halos and particles | boolean
    DO_2LPT: Use the second order LPT to displace the halos and particles | boolean
    OUT_FLAG: Output the flag corresponding to the host halo of each particle | boolean
    verbose: Output or do not output information in the c code | boolean
    nthreads: Number of thread to be used in some computations | int

    return: Numpy arrays with all output ordered between: halo positions (nh, 3), halo velocities (nh, 3), halo masses (nh), particle positions (np, 3), particle velocities (np, 3), particle flags (np) and density grid (ndx, ndy, ndz)
    """

    #Check the precision and convert the arrays
    precision = exshalos.exshalos.exshalos.check_precision()

    if(precision == 4):
        k = k.astype("float32")
        P = P.astype("float32")
        R_max = np.float32(R_max)
        Lc = np.float32(Lc)
        Om0 = np.float32(Om0)
        z = np.float32(z)
        k_smooth = np.float32(k_smooth)
        delta_c = np.float32(delta_c)
        a = np.float32(a)
        beta = np.float32(beta)
        alpha = np.float32(alpha)
        phase - np.float32(phase)

    else:
        k = k.astype("float64")
        P = P.astype("float64")
        R_max = np.float64(R_max)
        Lc = np.float64(Lc)
        Om0 = np.float64(Om0)
        z = np.float64(z)
        k_smooth = np.float64(k_smooth)
        delta_c = np.float64(delta_c)
        a = np.float64(a)
        beta = np.float64(beta)
        alpha = np.float64(alpha) 
        phase = np.float64(phase) 

    #Define the number of cells in each direction
    if(ndx is None):
        ndx = nd
    if(ndy is None):
        ndy = nd
    if(ndz is None):
        ndz = nd     

    #Define the seed
    if(seed is None):
        seed = np.random.randint(1e+9)

    #Run the .C program to generate the halo catalogue
    x = exshalos.exshalos.exshalos.halos_box_from_pk(k, P, R_max, np.int32(ndx), np.int32(ndy), np.int32(ndz), Lc, np.int32(seed), k_smooth, Om0, z, delta_c, np.int32(Nmin), a, beta, alpha, np.int32(fixed), phase, np.int32(OUT_DEN), np.int32(OUT_LPT), np.int32(OUT_VEL), np.int32(DO_2LPT), np.int32(OUT_FLAG), np.int32(verbose), np.int32(nthreads))

    return x

#Generate a halo catalogue from a density grid
def Generate_Halos_Box_from_Grid(grid, k, P, S = None, V = None, Lc = 2.0, Om0 = 0.31, z = 0.0, k_smooth = 10000.0, delta_c = -1.0, Nmin = 1, a = 1.0, beta = 0.0, alpha = 0.0, OUT_LPT = False, OUT_VEL = False, DO_2LPT = False, OUT_FLAG = False, verbose = False, nthreads = 1):
    """
    grid: Density grid used to generate the halos | 3D numpy array (ndx, ndy, ndz)
    S: Displacements of the particles in the grid | 2D numpy array (np, 3)
    V: Velocity of the particles in the grid | 2D numpy array (np, 3)
    k: Wavenumbers of the power spectrum | 1D numpy array
    P: Power spectrum | 1D numpy array
    Lc: Size of each cell in Mpc/h | float
    Om0: Value of the matter overdensity today | float
    z: Redshift of the density grid and final halo catalogue | float
    k_smooth: Scale used to smooth the LPT computations | float
    delta_c: Critical density of the halo formation linearly extrapolated to z | float
    Nmin: Minimum number of particles in each halo | int
    a: a parameter of the ellipsoidal barrier | float
    beta: beta parameter of the ellipsoidal barrier | float
    alpha: alpha parameter of the ellipsoidal barrier | float
    OUT_LPT: Output the dispaced particles | boolean
    OUT_VEL: Output the velocities of halos and particles | boolean
    DO_2LPT: Use the second order LPT to displace the halos and particles | boolean
    OUT_FLAG: Output the flag corresponding to the host halo of each particle | boolean
    verbose: Output or do not output information in the c code | boolean
    nthreads: Number of thread to be used in some computations | int

    return: Numpy arrays with all output ordered between: halo positions (nh, 3), halo velocities (nh, 3), halo masses (nh), particle positions (np, 3), particle velocities (np, 3) and particle flags (np)
    """

    precision = exshalos.exshalos.exshalos.check_precision()

    if(S is None):
        In_disp = False
    else:
        In_disp = True

    if(In_disp == True):
        if(S.shape[0] != grid.shape[0]*grid.shape[1]*grid.shape[2]):
            raise ValueError("The number of particles in S is different of the number of grid cells!")
        if(S.shape[1] != 3):
            raise ValueError("The number of components of S is different from 3!")
        if(V is not None):
            if(V.shape[0] != grid.shape[0]*grid.shape[1]*grid.shape[2]):
                raise ValueError("The number of particles in V is different of the number of grid cells!")   
            if(V.shape[1] != 3):
                raise ValueError("The number of components of V is different from 3!")        

    if(precision == 4):
        grid = grid.astype("float32")
        k = k.astype("float32")
        P = P.astype("float32")
        Lc = np.float32(Lc)
        Om0 = np.float32(Om0)
        z = np.float32(z)
        k_smooth = np.float32(k_smooth)
        delta_c = np.float32(delta_c)
        a = np.float32(a)
        beta = np.float32(beta)
        alpha = np.float32(alpha)
        if(S is not None):
            S = S.astype("float32")
        if(V is not None):
            V = V.astype("float32")

    else:
        grid = grid.astype("float64")
        k = k.astype("float64")
        P = P.astype("float64")
        Lc = np.float64(Lc)
        Om0 = np.float64(Om0)
        z = np.float64(z)
        k_smooth = np.float64(k_smooth)
        delta_c = np.float64(delta_c)
        a = np.float64(a)
        beta = np.float64(beta)
        alpha = np.float64(alpha)   
        if(S is not None):
            S = S.astype("float64")
        if(V is not None):
            V = V.astype("float64")

    x = exshalos.exshalos.exshalos.halos_box_from_grid(k, P, grid, S, V, Lc, k_smooth, Om0, z, delta_c, np.int32(Nmin), a, beta, alpha, np.int32(OUT_LPT), np.int32(OUT_VEL), np.int32(DO_2LPT), np.int32(OUT_FLAG), np.int32(In_disp), np.int32(verbose), np.int32(nthreads))

    return x

#Populate the halos with one type of galaxy using a HOD
def Generate_Galaxies_from_Halos(posh, Mh, velh = None, Ch = None, nd = 256, ndx = None, ndy = None, ndz = None, Lc = 2.0, Om0 = 0.31, z = 0.0, logMmin = 13.25424743, siglogM =  0.26461332, logM0 = 13.28383025, logM1 = 14.32465146, alpha = 1.00811277, sigma = 0.5, Deltah = -1.0, seed = 12345, OUT_VEL = False, OUT_FLAG = False, verbose = False):
    """
    posh: Positions of the halos| 2D array (Nh, 3)
    velh: Velocities of the halos| 2D array (Nh, 3)
    Mh: Mass of the halos | 1D array (Nh)
    Ch: Concentration of the halos | 1D array (Nh)
    nd or ndx, ndy, ndz: Number of cells in each direction | ints
    Lc: Size of each cell in Mpc/h | float
    Om0: Value of the matter overdensity today | float
    z: Redshift of the density grid and final halo catalogue | float
    logMmin, siglogM, logM0, logM1, alpha: Parameters of the HOD models (Zheng 2005) | float
    sigma: Parameter of the exclusion term of the halo density profile (Voivodic 2020) | float
    Deltah: Overdensity of the halos | float
    seed: Seed used to generate the density field | int
    OUT_VEL: Output the velocities of halos and particles | boolean
    OUT_FLAG: Output the flag of the host halo of each galaxy | boolean
    verbose: Output or do not output information in the c code | boolean

    return: Numpy arrays with the positions, velocities and flags of the halos | 2D arrays (Ng, 3) for positons and velocies and 1D array (Ng) for the flags
    """

    precision = exshalos.hod.hod.check_precision()

    In_C = False
    if(precision == 4):
        posh = posh.astype("float32")
        if(velh is not None):
            velh = velh.astype("float32")
        Mh = Mh.astype("float32")
        if(Ch is not None):
            In_C = True
            Ch = Ch.astype("float32")
        Lc = np.float32(Lc)
        Om0 = np.float32(Om0)
        z = np.float32(z)    
        logMmin = np.float32(logMmin)
        siglogM = np.float32(siglogM)
        logM0 = np.float32(logM0)
        logM1 = np.float32(logM1)
        alpha = np.float32(alpha)
        sigma = np.float32(sigma)
        Deltah = np.float32(Deltah)

    else:
        posh = posh.astype("float64")
        if(velh is not None):
            velh = velh.astype("float64")
        Mh = Mh.astype("float64")
        if(Ch is not None):
            In_C = True
            Ch = Ch.astype("float64")
        Lc = np.float64(Lc)
        Om0 = np.float64(Om0)
        z = np.float64(z)    
        logMmin = np.float64(logMmin)
        siglogM = np.float64(siglogM)
        logM0 = np.float64(logM0)
        logM1 = np.float64(logM1)
        alpha = np.float64(alpha)
        sigma = np.float64(sigma)
        Deltah = np.float64(Deltah)

    if(ndx is None):
        ndx = nd
    if(ndy is None):
        ndy = nd
    if(ndz is None):
        ndz = nd    

    x = exshalos.hod.hod.populate_halos(posh, velh, Mh, Ch, Lc, Om0, z, np.int32(ndx), np.int32(ndy), np.int32(ndz), logMmin, siglogM, logM0, logM1, alpha, sigma, Deltah, np.int32(seed), np.int32(OUT_VEL), np.int32(OUT_FLAG), np.int32(In_C), np.int32(verbose))

    return x

#Split the galaxies in two colors
def Split_Galaxies(Mh, Flag, params_cen = np.array([37.10265321, -5.07596644, 0.17497771]), params_sat = np.array([19.84341938, -2.8352781, 0.10443049]), seed = 12345, verbose = False):
    """
    Mh: Mass of the halos | 1D array (Nh)
    Flag: Flag with the label of the host halo of each galaxy | 1D array (Ng)
    C3, C2, C1, C0: Parameters used to split the central galaxies | float
    S3, S2, S1, S0: Parameters used to split the satellite galaxies | float
    seed: Seed used to generate the random numbers | int
    verbose: Output or do not output information in the c code | boolean

    return: Type of each galaxy | 1D array (Ng)
    """

    precision = exshalos.hod.hod.check_precision()

    if(precision == 4):
        Mh = Mh.astype("float32")
        params_cen = params_cen.astype("float32")
        params_sat = params_sat.astype("float32")
    else:
        Mh = Mh.astype("float64")
        params_cen = params_cen.astype("float64")
        params_sat = params_sat.astype("float64")

    if(len(params_cen.shape) == 1):
        params_cen = params_cen.reshape([1, len(params_cen)])
    if(len(params_sat.shape) == 1):
        params_sat = params_sat.reshape([1, len(params_sat)])        

    if(params_cen.shape[0] != params_sat.shape[0]):
        raise ValueError("Different number of types of galaxies for the centrals and satellites! %d != %d!" %(params_cen.shape[0],  params_sat.shape[0]))

    x = exshalos.hod.hod.split_galaxies(Mh, Flag, params_cen, params_sat, np.int32(seed), np.int32(verbose))

    return x