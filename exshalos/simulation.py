from dbm import ndbm
import exshalos
import numpy as np

#Compute the density grid
def Compute_Density_Grid(pos, vel = None, mass = None, type = None, nd = 256, L = 1000.0, Om0 = 0.31, z = 0.0, direction = None, window = "CIC", R = 4.0, R_times = 5.0, interlacing = False, verbose = False, nthreads = 1):
    """
    pos: Position of the tracers | 2D numpy array [number of tracers, 3]
    vel: Velocities of the particles in the given direction | 1D numpy array [number of tracers]
    mass: Weight of each tracer | 1D numpy array [number of tracers]
    type: Type of each tracer | 1D numpy array [number of tracers]
    nd: Number of cells per dimension | int
    L: Size of the box in Mpc/h | float
    direction: Direction to use to put the particles in redshift space | int
    window: Density assigment method used to construct the density grid | string or int ["NGP" = 0, "CIC" = 1, "SPHERICAL" = 2, "EXPONENTIAL" = 3]
    R: Smoothing lenght used in the spherical and exponential windows | float
    R_times: Scale considered to account for particles used in the exponential window in units of R | float
    interlacing: Use or do not use the interlacing technic | boolean
    verbose: Output or do not output information in the c code | boolean
    nthreads: Number of threads used by openmp | int

    return: The density grid for each type of tracer and for the grids with and without interlacing (if interlacing == True) | 5D numpy array [1 if interlacing == False and 2 instead, number of types of tracer, nd, nd, nd]
    """

    nmass = 0
    precision = exshalos.spectrum.spectrum.check_precision()

    if(precision == 4):
        pos = pos.astype("float32")
        vel = vel.astype("float32")
        if(mass is not None):
            mass = mass.astype("float32")
            nmass = 1
        if(vel is not None):
            vel = vel.astype("float32")
        L = np.float32(L)
        R = np.float32(R)
        R_times = np.float32(R_times)
        Om0 = np.float32(Om0)
        z = np.float32(z)
    else:
        pos = pos.astype("float64")
        vel = vel.astype("float64")
        if(mass is not None):
            mass = mass.astype("float64")
            nmass = 1
        if(vel is not None):
            vel = vel.astype("float64")
        L = np.float64(L)
        R = np.float64(R)
        R_times = np.float64(R_times)   
        Om0 = np.float64(Om0)
        z = np.float64(z)   

    if(type is not None):
        type = type.astype("int32") 
        ntype = len(np.unique(type))
    else:
        ntype = 1

    if(direction == None):
        direction = -1
        v = None
    elif(direction == "x" or direction == "X" or direction == 0):
        direction = 0
        v = vel[:,0]/100.0*(1.0 + z)/exshalos.theory.Get_Hz(Om0 = Om0, z = z)
    elif(direction == "y" or direction == "Y" or direction == 1):
        direction = 1
        v = vel[:,1]/100.0*(1.0 + z)/exshalos.theory.Get_Hz(Om0 = Om0, z = z)
    elif(direction == "z" or direction == "Z" or direction == 2):
        direction = 2
        v = vel[:,2]/100.0*(1.0 + z)/exshalos.theory.Get_Hz(Om0 = Om0, z = z)

    if(window == "NO" or window == "no" or window == 0):
        print("You need to choose some density assigment method to construct the density grid!")
        return None
    elif(window == "NGP" or window == "ngp" or window == 1):
        window = 1
    elif(window == "CIC" or window == "cic" or window == 2):
        window = 2
    elif(window == "SPHERICAL" or window == "spherical" or window == 3):
        window = 3
    elif(window == "EXPONENTIAL" or window == "exponential" or window == 4):
        window = 4

    grid = exshalos.spectrum.spectrum.grid_compute(pos, v, mass, np.int32(nmass), type, np.int32(ntype), np.int32(nd), L, Om0, z, np.int32(direction), np.int32(window), R, R_times, np.int32(interlacing), np.int32(verbose), np.int32(nthreads))

    del v

    if(interlacing == False and ntype == 1):
        grid = grid.reshape([nd, nd, nd])
    elif(interlacing == False):
        grid = grid.reshape([ntype, nd, nd, nd])
    elif(ntype == 1):
        grid = grid.reshape([2, nd, nd, nd])

    return grid

#Compute the power spectrum given the density grid
def Compute_Power_Spectrum(grid, L = 1000.0, window = 0, R = 4.0, Nk = 25, k_min = None, k_max = None, l_max = 0, verbose = False, nthreads = 1, ntype = 1):
    """
    grid: Density grid for all tracers | 5D numpy array [1 if interlacing == False and 2 instead, number of types of tracer, nd, nd, nd]
    L: Size of the box in Mpc/h | float
    window: Density assigment method used to construct the density grid | string or int ["NGP" = 0, "CIC" = 1, "SPHERICAL" = 2, "EXPONENTIAL" = 3]
    R: Smoothing lenght used in the spherical and exponential windows | float
    R_times: Scale considered to account for particles used in the exponential window in units of R | float
    Nk: Number of bins in k to compute the power spectra | int
    k_min: Minimum value of k to compute the power spectra | float
    k_max: Maximum value of k to compute the power spectra | float
    l_max: Maximum multipole computed | int
    verbose: Output or do not output information in the c code | boolean
    nthreads: Number of threads used by openmp | int
    ntype: Number of different types of tracers | int

    return: All possible power spectra, the wavenumbers where the power spectra were mesured and the number of independent modes | Dictionary with 3 arrays. "k": 1D array [Nk], "Pk": 2D array [Number of spectra x Nk], "Nk": 1D array [Nk]
    """

    precision = exshalos.spectrum.spectrum.check_precision()

    #Compute some parameters
    if(len(grid.shape) == 3):
        interlacing = 0
        ntype = 1
    elif(len(grid.shape) == 5):
        interlacing = 1
        ntype = grid.shape[0]
    elif(len(grid.shape) == 4 and ntype == 1):
        interlacing = 1
    elif(len(grid.shape) == 4 and ntype > 1):
        interlacing = 0
    nd = grid.shape[-1]

    if(k_min is None):
        k_min = 2.0*np.pi/L

    if(k_max is None or k_max > np.pi/L*nd):
        k_max = np.pi/L*nd

    if(precision == 4):
        grid = grid.astype("float32")
        L = np.float32(L)
        R = np.float32(R)
        k_min = np.float32(k_min)
        k_max = np.float32(k_max)
    else:
        grid = grid.astype("float64")
        L = np.float64(L)
        R = np.float64(R)
        k_min = np.float64(k_min)
        k_max = np.float64(k_max)  

    #Set the window function to be de-convolved
    if(window == "NO" or window == "no" or window == 0):
        window = 0
    elif(window == "NGP" or window == "ngp" or window == 1):
        window = 1
    elif(window == "CIC" or window == "cic" or window == 2):
        window = 2
    elif(window == "SPHERICAL" or window == "spherical" or window == 3):
        window = 3
    elif(window == "EXPONENTIAL" or window == "exponential" or window == 4):
        window = 4

    x = exshalos.spectrum.spectrum.power_compute(grid, np.int32(ntype), np.int32(nd), L, np.int32(window), R, np.int32(interlacing), np.int32(Nk), k_min, k_max, np.int32(l_max), np.int32(verbose), np.int32(nthreads))

    if(ntype == 1 and l_max == 0):
        x["Pk"] = x["Pk"].reshape([Nk])
    elif(ntype == 1):
        x["Pk"] = x["Pk"].reshape([l_max+1, Nk])
    elif(l_max == 0):
        x["Pk"] = x["Pk"].reshape([int(ntype*(ntype + 1)/2), Nk])

    return x

#Compute the bispectrum given the density grid
def Compute_BiSpectrum(grid, L = 1000.0, window = "CIC", R = 4.0, Nk = 25, k_min = None, k_max = None, verbose = False, nthreads = 1, ntype = 1):
    """
    grid: Density grid for all tracers | 5D numpy array [1 if interlacing == False and 2 instead, number of types of tracer, nd, nd, nd]
    L: Size of the box in Mpc/h | float
    window: Density assigment method used to construct the density grid | string or int ["NGP" = 0, "CIC" = 1, "SPHERICAL" = 2, "EXPONENTIAL" = 3]
    R: Smoothing lenght used in the spherical and exponential windows | float
    R_times: Scale considered to account for particles used in the exponential window in units of R | float
    Nk: Number of bins in k to compute the power spectra | int
    k_min: Minimum value of k to compute the power spectra | float
    k_max: Maximum value of k to compute the power spectra | float
    verbose: Output or do not output information in the c code | boolean
    nthreads: Number of threads used by openmp | int
    ntype: Number of different types of tracers | int

    return: All possible power spectra, the wavenumbers where the power spectra were mesured, the number of independent modes, all triplet of ks for the bispectra, all possible bispectra and the number of triangular configurations | Dictionaty with 6 arrays. "kP": 1D array [Nk], "Pk": 2D array [Number of spectra x Nk], "Nk": 1D array [Nk], "kB": 2D array [Number of bins of the bispectrum, 3], "Bk": 2D array [Number of bispectra, Number of bins of the bispectrum], "Ntri": 1D array [Number of independent triangles in this bin]
    """

    precision = exshalos.spectrum.spectrum.check_precision()

    #Compute some parameters
    if(len(grid.shape) == 3):
        interlacing = 0
        ntype = 1
    elif(len(grid.shape) == 5):
        interlacing = 1
        ntype = grid.shape[0]
    elif(len(grid.shape) == 4 and ntype == 1):
        interlacing = 1
    elif(len(grid.shape) == 4 and ntype > 1):
        interlacing = 0
    nd = grid.shape[-1]

    if(k_min is None):
        k_min = 2.0*np.pi/L

    if(k_max is None):
        k_max = np.pi/L*nd

    if(precision == 4):
        grid = grid.astype("float32")
        L = np.float32(L)
        R = np.float32(R)
        k_min = np.float32(k_min)
        k_max = np.float32(k_max)
    else:
        grid = grid.astype("float64")
        L = np.float64(L)
        R = np.float64(R)
        k_min = np.float64(k_min)
        k_max = np.float64(k_max)  

    if(window == "NO" or window == "no" or window == 0):
        window = 0
    elif(window == "NGP" or window == "ngp" or window == 1):
        window = 1
    elif(window == "CIC" or window == "cic" or window == 2):
        window = 2
    elif(window == "SPHERICAL" or window == "spherical" or window == 3):
        window = 3
    elif(window == "EXPONENTIAL" or window == "exponential" or window == 4):
        window = 4

    x = exshalos.spectrum.spectrum.bi_compute(grid, np.int32(ntype), np.int32(nd), L, np.int32(window), R, np.int32(interlacing), np.int32(Nk), k_min, k_max, np.int32(verbose), np.int32(nthreads))

    if(ntype == 1):
        x["Pk"] = x["Pk"].reshape([Nk])
        x["Bk"] = x["Bk"].reshape([len(x["Ntri"])])

    return x

#Compute the trispectrum given the density grid
def Compute_TriSpectrum(grid, L = 1000.0, window = "CIC", R = 4.0, Nk = 25, k_min = None, k_max = None, verbose = False, nthreads = 1, ntype = 1):
    """
    grid: Density grid for all tracers | 5D numpy array [1 if interlacing == False and 2 instead, number of types of tracer, nd, nd, nd]
    L: Size of the box in Mpc/h | float
    window: Density assigment method used to construct the density grid | string or int ["NGP" = 0, "CIC" = 1, "SPHERICAL" = 2, "EXPONENTIAL" = 3]
    R: Smoothing lenght used in the spherical and exponential windows | float
    R_times: Scale considered to account for particles used in the exponential window in units of R | float
    Nk: Number of bins in k to compute the power spectra | int
    k_min: Minimum value of k to compute the power spectra | float
    k_max: Maximum value of k to compute the power spectra | float
    verbose: Output or do not output information in the c code | boolean
    nthreads: Number of threads used by openmp | int
    ntype: Number of different types of tracers | int

    return: All possible power spectra, the wavenumbers where the power spectra were mesured and the number of independent modes | Tuple with 3 arrays: 1D array [Nk], 2D array [Number of spectra x Nk], 1D array [Nk]
    """

    precision = exshalos.spectrum.spectrum.check_precision()

    #Compute some parameters
    if(len(grid.shape) == 3):
        interlacing = 0
        ntype = 1
    elif(len(grid.shape) == 5):
        interlacing = 1
        ntype = grid.shape[0]
    elif(len(grid.shape) == 4 and ntype == 1):
        interlacing = 1
    elif(len(grid.shape) == 4 and ntype > 1):
        interlacing = 0
    nd = grid.shape[-1]

    if(k_min is None):
        k_min = 2.0*np.pi/L

    if(k_max is None):
        k_max = np.pi/L*nd

    if(precision == 4):
        grid = grid.astype("float32")
        L = np.float32(L)
        R = np.float32(R)
        k_min = np.float32(k_min)
        k_max = np.float32(k_max)
    else:
        grid = grid.astype("float64")
        L = np.float64(L)
        R = np.float64(R)
        k_min = np.float64(k_min)
        k_max = np.float64(k_max)  

    if(window == "NO" or window == "no" or window == 0):
        window = 0
    elif(window == "NGP" or window == "ngp" or window == 1):
        window = 1
    elif(window == "CIC" or window == "cic" or window == 2):
        window = 2
    elif(window == "SPHERICAL" or window == "spherical" or window == 3):
        window = 3
    elif(window == "EXPONENTIAL" or window == "exponential" or window == 4):
        window = 4

    x = exshalos.spectrum.spectrum.tri_compute(grid, np.int32(ntype), np.int32(nd), L, np.int32(window), R, np.int32(interlacing), np.int32(Nk), k_min, k_max, np.int32(verbose), np.int32(nthreads))

    if(ntype == 1):
        x["Pk"] = x["Pk"].reshape([Nk])
        x["Tk"] = x["Tk"].reshape([len(x["Nsq"])])
        x["Tuk"] = x["Tuk"].reshape([len(x["Nsq"])])

    return x

#Compute the correlation function given the power spectrum or the power spectrum given the correlation function
def Compute_Correlation(k, P, direction = 1, verbose = False):
    """
    k: Wavebumber of the power spectrum or the distance of the correlation function | 1D numpy array
    P: Power spectrum or the correlation function | 1D numpy array
    direction: Direction to compute the fftlog: 1 to compute the correlation and -1 to compute the power spectrum | int
    verbose: Output or do not output information in the c code | boolean

    return: The correlation function (direction == 1) or the power spectrum (direction == -1) | Dictonaty with 2 arrays. "R" 1D array (Number of radial bins), "Xi" 1D array (Number of radial bins) 
    """

    precision = exshalos.exshalos.exshalos.check_precision()

    if(precision == 4):
        k = k.astype("float32")
        P = P.astype("float32")
    else:
        k = k.astype("float64")
        P = P.astype("float64")     

    if(direction != 1 and direction != -1):
        print("Wrong direction gave! It must be 1 or -1 NOT %d!" %(direction))  
        return None

    x = exshalos.exshalos.exshalos.correlation_compute(k, P, np.int32(direction), np.int32(verbose))

    return x

#Measure the abundance of a list of halo masses
def Compute_Abundance(Mh, Mmin = -1.0, Mmax = -1.0, Nm = 25, Lc = 2.0, nd = 256, ndx = None, ndy = None, ndz = None, verbose = False):
    """
    Mh: Mass of each halo | 1D numpy array (Nh)
    Mmin: Minimum mass used to construct the mass bins | float
    Mmax: Maximum mass used to construct the mass bins | float
    Nm: Number of mass bins | int
    Lc: Size of each cell
    nd, ndx, ndy, ndz: Number of cells in each direction | int
    verbose: Output or do not output information in the c code | boolean
    
    return: The mean mass, differential abundance, and error in the differential abundance for each mass bins | Dictonary with 3 numpy arrays with size Nh
    """

    precision = exshalos.spectrum.spectrum.check_precision()

    if(precision == 4):
        Mh = Mh.astype("float32")
        Mmin = np.float32(Mmin)
        Mmax = np.float32(Mmax)
        Lc = np.float32(Lc)
    else:
        Mh = Mh.astype("float64")
        Mmin = np.float64(Mmin)
        Mmax = np.float64(Mmax)
        Lc = np.float64(Lc)

    if(ndx is None):
        ndx = nd
    if(ndy is None):
        ndy = nd
    if(ndz is None):
        ndz = nd   

    x = exshalos.spectrum.spectrum.abundance_compute(Mh, Mmin, Mmax, np.int32(Nm), Lc, np.int32(ndx), np.int32(ndy), np.int32(ndz), np.int32(verbose))

    return x