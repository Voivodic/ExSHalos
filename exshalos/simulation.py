import exshalos
import numpy as np

#Compute the density grid
def Compute_Density_Grid(pos, vel = None, mass = None, types = None, nd = 256, L = 1000.0, Om0 = 0.31, z = 0.0, direction = None, window = "CIC", R = 4.0, R_times = 5.0, interlacing = False, folds = 1, verbose = False, nthreads = 1):
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
    folds: Number of times that the box will be folded in each direction | int
    verbose: Output or do not output information in the c code | boolean
    nthreads: Number of threads used by openmp | int

    return: The density grid for each type of tracer and for the grids with and without interlacing (if interlacing == True) | 5D numpy array [1 if interlacing == False and 2 instead, number of types of tracer, nd, nd, nd]
    """

    nmass = 0
    precision = exshalos.spectrum.spectrum.check_precision()

    if(precision == 4):
        pos = pos.astype("float32")
        if(vel is not None):
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
        if(vel is not None):
            vel = vel.astype("float32")
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

    if(types is not None):
        types = np.fabs(types)
        types = types - np.min(types)
        types = types.astype("int32") 
        ntypes = len(np.unique(types))
    else:
        ntypes = 1

    if(direction == None or direction == -1):
        direction = -1
    elif(direction == "x" or direction == "X" or direction == 0):
        direction = 0
    elif(direction == "y" or direction == "Y" or direction == 1):
        direction = 1
    elif(direction == "z" or direction == "Z" or direction == 2):
        direction = 2
    else:
        raise ValueError("Direction must be None, x, y or z!")

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

    grid = exshalos.spectrum.spectrum.grid_compute(pos, vel, mass, np.int32(nmass), types, np.int32(ntypes), np.int32(nd), L, Om0, z, np.int32(direction), np.int32(window), R, R_times, np.int32(interlacing), np.int32(folds), np.int32(verbose), np.int32(nthreads))

    if(interlacing == False and ntypes == 1):
        grid = grid.reshape([nd, nd, nd])
    elif(interlacing == False):
        grid = grid.reshape([ntypes, nd, nd, nd])
    elif(ntypes == 1):
        grid = grid.reshape([2, nd, nd, nd])

    return grid

#Compute the power spectrum given the density grid
def Compute_Power_Spectrum(grid, L = 1000.0, window = 0, R = 4.0, Nk = 25, k_min = None, k_max = None, l_max = 0, direction = None, folds = 1, verbose = False, nthreads = 1, ntypes = 1):
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
    ntypes: Number of different types of tracers | int

    return: All possible power spectra, the wavenumbers where the power spectra were mesured and the number of independent modes | Dictionary with 3 arrays. "k": 1D array [Nk], "Pk": 2D array [Number of spectra x Nk], "Nk": 1D array [Nk]
    """

    precision = exshalos.spectrum.spectrum.check_precision()

    #Compute some parameters
    if(len(grid.shape) == 3):
        interlacing = 0
        ntypes = 1
    elif(len(grid.shape) == 5):
        interlacing = 1
        ntypes = grid.shape[0]
    elif(len(grid.shape) == 4 and ntypes == 1):
        interlacing = 1
    elif(len(grid.shape) == 4 and ntypes > 1):
        interlacing = 0
    nd = grid.shape[-1]
    L = L/folds

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

    #Set the window function to be de-convolved
    if(window == "NO" or window == "no" or window == "No" or window == 0):
        window = 0
    elif(window == "NGP" or window == "ngp" or window == 1):
        window = 1
    elif(window == "CIC" or window == "cic" or window == 2):
        window = 2
    elif(window == "SPHERICAL" or window == "spherical" or window == 3):
        window = 3
    elif(window == "EXPONENTIAL" or window == "exponential" or window == 4):
        window = 4

    if(direction == None or direction == -1):
        direction = -1
    elif(direction == "x" or direction == "X" or direction == 0):
        direction = 0
    elif(direction == "y" or direction == "Y" or direction == 1):
        direction = 1
    elif(direction == "z" or direction == "Z" or direction == 2):
        direction = 2
    else:
        raise ValueError("Direction must be None, x, y or z!")

    x = exshalos.spectrum.spectrum.power_compute(grid, np.int32(ntypes), np.int32(nd), L, np.int32(window), R, np.int32(interlacing), np.int32(Nk), k_min, k_max, np.int32(l_max), np.int32(direction), np.int32(verbose), np.int32(nthreads))

    if(ntypes == 1 and l_max == 0):
        x["Pk"] = x["Pk"].reshape([Nk])
    elif(ntypes == 1):
        x["Pk"] = x["Pk"].reshape([int(l_max/2)+1, Nk])
    elif(l_max == 0):
        x["Pk"] = x["Pk"].reshape([int(ntypes*(ntypes + 1)/2), Nk])

    return x

#Compute the power spectrum given the density grid
def Compute_Power_Spectrum_individual(grid, pos, L = 1000.0, window = 0, R = 4.0, Nk = 25, k_min = None, k_max = None, l_max = 0, direction = None, folds = 1, verbose = False, nthreads = 1, ntypes = 1):
    """
    grid: Density grid for all tracers | 5D numpy array [1 if interlacing == False and 2 instead, number of types of tracer, nd, nd, nd]
    pos: Position of the tracers | 2D numpy array [number of tracers, 3]
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
    ntypes: Number of different types of tracers | int

    return: All possible power spectra, the wavenumbers where the power spectra were mesured and the number of independent modes | Dictionary with 3 arrays. "k": 1D array [Nk], "Pk": 2D array [Number of spectra x Nk], "Nk": 1D array [Nk]
    """

    precision = exshalos.spectrum.spectrum.check_precision()

    #Compute some parameters
    if(len(grid.shape) == 3):
        interlacing = 0
        ntypes = 1
    elif(len(grid.shape) == 5):
        interlacing = 1
        ntypes = grid.shape[0]
    elif(len(grid.shape) == 4 and ntypes == 1):
        interlacing = 1
    elif(len(grid.shape) == 4 and ntypes > 1):
        interlacing = 0
    nd = grid.shape[-1]
    Np = pos.shape[0]
    L = L/folds

    if(k_min is None):
        k_min = 2.0*np.pi/L

    if(k_max is None):
        k_max = np.pi/L*nd

    if(precision == 4):
        grid = grid.astype("float32")
        pos = pos.astype("float32")
        L = np.float32(L)
        R = np.float32(R)
        k_min = np.float32(k_min)
        k_max = np.float32(k_max)
    else:
        grid = grid.astype("float64")
        pos = pos.astype("float64")
        L = np.float64(L)
        R = np.float64(R)
        k_min = np.float64(k_min)
        k_max = np.float64(k_max)  

    #Set the window function to be de-convolved
    if(window == "NO" or window == "no" or window == "No" or window == 0):
        window = 0
    elif(window == "NGP" or window == "ngp" or window == 1):
        window = 1
    elif(window == "CIC" or window == "cic" or window == 2):
        window = 2
    elif(window == "SPHERICAL" or window == "spherical" or window == 3):
        window = 3
    elif(window == "EXPONENTIAL" or window == "exponential" or window == 4):
        window = 4

    if(direction == None or direction == -1):
        direction = -1
    elif(direction == "x" or direction == "X" or direction == 0):
        direction = 0
    elif(direction == "y" or direction == "Y" or direction == 1):
        direction = 1
    elif(direction == "z" or direction == "Z" or direction == 2):
        direction = 2
    else:
        raise ValueError("Direction must be None, x, y or z!")

    x = exshalos.spectrum.spectrum.power_compute_individual(grid, pos, np.int32(ntypes), np.int32(nd), L, np.int32(window), R, np.int32(interlacing), np.int32(Nk), k_min, k_max, np.int32(l_max), np.int32(direction), np.int32(verbose), np.int32(nthreads))

    if(ntypes == 1 and l_max == 0):
        x["Pk"] = x["Pk"].reshape([Np, Nk])
    elif(ntypes == 1):
        x["Pk"] = x["Pk"].reshape([Np, int(l_max/2)+1, Nk])
    elif(l_max == 0):
        x["Pk"] = x["Pk"].reshape([Np, ntypes, Nk])

    return x

#Compute the bispectrum given the density grid
def Compute_BiSpectrum(grid, L = 1000.0, window = "CIC", R = 4.0, Nk = 25, k_min = None, k_max = None, folds = 1, verbose = False, nthreads = 1, ntypes = 1):
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
    ntypes: Number of different types of tracers | int

    return: All possible power spectra, the wavenumbers where the power spectra were mesured, the number of independent modes, all triplet of ks for the bispectra, all possible bispectra and the number of triangular configurations | Dictionaty with 6 arrays. "kP": 1D array [Nk], "Pk": 2D array [Number of spectra x Nk], "Nk": 1D array [Nk], "kB": 2D array [Number of bins of the bispectrum, 3], "Bk": 2D array [Number of bispectra, Number of bins of the bispectrum], "Ntri": 1D array [Number of independent triangles in this bin]
    """

    precision = exshalos.spectrum.spectrum.check_precision()

    #Compute some parameters
    if(len(grid.shape) == 3):
        interlacing = 0
        ntypes = 1
    elif(len(grid.shape) == 5):
        interlacing = 1
        ntypes = grid.shape[0]
    elif(len(grid.shape) == 4 and ntypes == 1):
        interlacing = 1
    elif(len(grid.shape) == 4 and ntypes > 1):
        interlacing = 0
    nd = grid.shape[-1]
    L = L/folds

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

    if(window == "NO" or window == "no" or window == "No" or window == 0):
        window = 0
    elif(window == "NGP" or window == "ngp" or window == 1):
        window = 1
    elif(window == "CIC" or window == "cic" or window == 2):
        window = 2
    elif(window == "SPHERICAL" or window == "spherical" or window == 3):
        window = 3
    elif(window == "EXPONENTIAL" or window == "exponential" or window == 4):
        window = 4

    x = exshalos.spectrum.spectrum.bi_compute(grid, np.int32(ntypes), np.int32(nd), L, np.int32(window), R, np.int32(interlacing), np.int32(Nk), k_min, k_max, np.int32(verbose), np.int32(nthreads))

    if(ntypes == 1):
        x["Pk"] = x["Pk"].reshape([Nk])
        x["Bk"] = x["Bk"].reshape([len(x["Ntri"])])

    return x

#Compute the trispectrum given the density grid
def Compute_TriSpectrum(grid, L = 1000.0, window = "CIC", R = 4.0, Nk = 25, k_min = None, k_max = None, folds = 1, verbose = False, nthreads = 1, ntypes = 1):
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
    ntypes: Number of different types of tracers | int

    return: All possible power spectra, the wavenumbers where the power spectra were mesured and the number of independent modes | Tuple with 3 arrays: 1D array [Nk], 2D array [Number of spectra x Nk], 1D array [Nk]
    """

    precision = exshalos.spectrum.spectrum.check_precision()

    #Compute some parameters
    if(len(grid.shape) == 3):
        interlacing = 0
        ntypes = 1
    elif(len(grid.shape) == 5):
        interlacing = 1
        ntypes = grid.shape[0]
    elif(len(grid.shape) == 4 and ntypes == 1):
        interlacing = 1
    elif(len(grid.shape) == 4 and ntypes > 1):
        interlacing = 0
    nd = grid.shape[-1]
    L = L/folds

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

    if(window == "NO" or window == "no" or window == "No" or window == 0):
        window = 0
    elif(window == "NGP" or window == "ngp" or window == 1):
        window = 1
    elif(window == "CIC" or window == "cic" or window == 2):
        window = 2
    elif(window == "SPHERICAL" or window == "spherical" or window == 3):
        window = 3
    elif(window == "EXPONENTIAL" or window == "exponential" or window == 4):
        window = 4

    x = exshalos.spectrum.spectrum.tri_compute(grid, np.int32(ntypes), np.int32(nd), L, np.int32(window), R, np.int32(interlacing), np.int32(Nk), k_min, k_max, np.int32(verbose), np.int32(nthreads))

    if(ntypes == 1):
        x["Pk"] = x["Pk"].reshape([Nk])
        x["Tk"] = x["Tk"].reshape([len(x["Nsq"])])
        x["Tuk"] = x["Tuk"].reshape([len(x["Nsq"])])

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

#Measure the b_n bias parameters uising Jens' technic
def Compute_Bias_Jens(grid, Mh, flag, Mmin = -1.0, Mmax = -1.0, Nm = 20, dmin = -1.0, dmax = 1.0, Nbins = 200, Lambda = -1.0, Lc = 2.0, Central = True, Normalized = False):
    """
    grid: Liniar (initial) density grid | 3D numpy array (Nd, Nd, Nd)
    Mh: Mass of all halos | 1D numpy array (Nh)
    flag: Flag of all grid cell as being part of a halo or not | 3D numpy array (Nd, Nd, Nd)
    Mmin: Minimum halo mass to be considered | float
    Mmax: Maximum halo mass to be considered | float
    Nm: Number of bins in mass | int
    dmin: Minimum density to be used in the construction of the histograms int delta | float
    dmax: Maximum density to be used in the construction of the histograms int delta | float
    Nbins: Number of bins to be used in the construction of the histograms int delta | int
    Lambda: Scale used to smooth the density field | float
    Lc: Size of the cells of the grid | float
    Central: Use only the central particle of the halo or all particles | boolean
    Normalized: Output the normalized pdf or not | boolean

    return: The histogram of the delta field unmasked and the histograms of the delta field masked for all halo mass bins | Python dictionary with Nm + 1 keys: "delta" 1D numpy array, "Mbin%d" 1D array where %d is in [0, Nm-1]
    """

    #Smooth the density field
    if(Lambda > 0.0):
        delta = exshalos.utils.Smooth_Fields(np.copy(grid), Lc = Lc, k_smooth = Lambda, Input_k = False, Nfields = 1, verbose = False, nthreads = 1)
    else:
        delta = np.copy(grid)

    #Convert the type of the floats
    precision = exshalos.spectrum.spectrum.check_precision()

    if(precision == 4):
        delta = np.float32(delta)
        Mh = np.float32(Mh)
        Mmin = np.float32(Mmin)
        Mmax = np.float32(Mmax)
        dmin = np.float32(dmin)
        dmax = np.float32(dmax)
    else:
        delta = np.float64(delta)
        Mh = np.float64(Mh)
        Mmin = np.float64(Mmin)
        Mmax = np.float64(Mmax)
        dmin = np.float64(dmin)
        dmax = np.float64(dmax)

    #Run the C code that does the job
    x = exshalos.spectrum.spectrum.histogram_compute(delta, Mh, np.int64(flag), Mmin, Mmax, np.int32(Nm), dmin, dmax, np.int32(Nbins), np.int32(Central))
    x["sigma2"] = np.mean(delta**2)
    del(delta)

    #Normalize the histograms
    if(Normalized == True):
        x["Unmasked"] = x["Unmasked"]/np.sum(x["Unmasked"])
        x["Masked"] = x["Masked"]/np.sum(x["Masked"], axis = 1).reshape([Nm, 1])

    return x