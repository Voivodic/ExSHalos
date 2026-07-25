from typing import Dict
from typing import Dict, Optional, Tuple, Union

import numpy as np


# Compute the gaussian density grid given the power spectrum
def Generate_Density_Grid(
    k: np.ndarray,
    P: np.ndarray,
    R_max: float = 100000.0,
    nd: int = 256,
    ndx: Optional[int] = None,
    ndy: Optional[int] = None,
    ndz: Optional[int] = None,
    Lc: float = 2.0,
    outk: bool = False,
    seed: int = 12345,
    fixed: bool = False,
    phase: float = 0.0,
    k_smooth: float = 100000.0,
    verbose: bool = False,
    nthreads: int = 1,
) -> np.ndarray:
    """
    Compute the Gaussian density grid given the power spectrum.

    :param k: Wavenumbers of the power spectrum.
    :type k: numpy.ndarray
    :param P: Power spectrum.
    :type P: numpy.ndarray
    :param R_max: Maximum size used to compute the correlation function in Mpc/h. Fiducial value: 100000.0
    :type R_max: float
    :param nd: Number of cells per dimension. Fiducial value: 256
    :type nd: int
    :param ndx: Number of cells in the x direction. Fiducial value: None
    :type ndx: Optional[int]
    :param ndy: Number of cells in the y direction. Fiducial value: None
    :type ndy: Optional[int]
    :param ndz: Number of cells in the z direction. Fiducial value: None
    :type ndz: Optional[int]
    :param Lc: Size of each cell in Mpc/h. Fiducial value: 2.0
    :type Lc: float
    :param outk: Whether to return the density field in Fourier space. Fiducial value: False
    :type outk: bool
    :param seed: Seed used to generate the random numbers. Fiducial value: 12345
    :type seed: int
    :param fixed: Whether to use fixed amplitudes. Fiducial value: False
    :type fixed: bool
    :param phase: Phase of the density field. Fiducial value: 0.0
    :type phase: float
    :param k_smooth: Smoothing scale in k-space. Fiducial value: 100000.0
    :type k_smooth: float
    :param verbose: Whether to output information in the C code. Fiducial value: False
    :type verbose: bool
    :param nthreads: Number of threads used by OpenMP. Fiducial value: 1
    :type nthreads: int

    :return: The 3D density grid in real space (and in Fourier space if outk is True).
    :rtype: numpy.ndarray
    """
    # Check the precision and convert the arrays
    from .lib.exshalos import check_precision

    precision = check_precision()
    if precision == 4:
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

    # Set the number of divisions per dimension
    if ndx is None:
        ndx = nd
    if ndy is None:
        ndy = nd
    if ndz is None:
        ndz = nd

    # Call the C function to compute the density field
    from .lib.exshalos import density_grid_compute

    x = density_grid_compute(
        k,
        P,
        R_max,
        np.int32(ndx),
        np.int32(ndy),
        np.int32(ndz),
        Lc,
        np.int32(outk),
        np.int32(seed),
        np.int32(fixed),
        phase,
        k_smooth,
        np.int32(verbose),
        np.int32(nthreads),
    )

    return x


# Generate a halo catalogue (in Lagrangian space) given an initial density grid
def Find_Halos_from_Grid(
    grid: np.ndarray,
    k: np.ndarray,
    P: np.ndarray,
    Lc: float = 2.0,
    Om0: float = 0.31,
    z: float = 0.0,
    delta_c: Optional[float] = None,
    Nmin: int = 10,
    a: float = 1.0,
    beta: float = 0.0,
    alpha: float = 0.0,
    OUT_FLAG: bool = False,
    verbose: bool = False,
) -> Dict[str, np.ndarray]:
    """
    Generate a halo catalogue (in Lagrangian space) given an initial density grid.

    :param grid: Density grid where the halos will be found.
    :type grid: numpy.ndarray
    :param k: Wavenumbers of the power spectrum.
    :type k: numpy.ndarray
    :param P: Power spectrum.
    :type P: numpy.ndarray
    :param Lc: Size of each cell in Mpc/h. Fiducial value: 2.0
    :type Lc: float
    :param Om0: Value of the matter overdensity today. Fiducial value: 0.31
    :type Om0: float
    :param z: Redshift of the density grid and final halo catalogue. Fiducial value: 0.0
    :type z: float
    :param delta_c: Critical density of the halo formation linearly extrapolated to z. Fiducial value: None
    :type delta_c: Optional[float]
    :param Nmin: Minimum number of particles in each halo. Fiducial value: 10
    :type Nmin: int
    :param a: Parameter a of the ellipsoidal barrier. Fiducial value: 1.0
    :type a: float
    :param beta: Parameter beta of the ellipsoidal barrier. Fiducial value: 0.0
    :type beta: float
    :param alpha: Parameter alpha of the ellipsoidal barrier. Fiducial value: 0.0
    :type alpha: float
    :param OUT_FLAG: Whether to output flag with the information if a cell belongs to a halo. Fiducial value: False
    :type OUT_FLAG: bool
    :param verbose: Whether to output information in the C code. Fiducial value: False
    :type verbose: bool

    :return: Dictionay with keys:

             - "posh": Ndarray with halo positions
             - "Mh": Ndarray with halo masses
             - "flag": Ndarray with flags for each cell

    :rtype: dict
    """
    # Check the precision and convert the arrays
    from .lib.exshalos import check_precision

    precision = check_precision()
    if precision == 4:
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

    # Call the C function to compute the halo catalogue
    from .lib.exshalos import find_halos

    x = find_halos(
        grid,
        k,
        P,
        Lc,
        Om0,
        z,
        delta_c,
        np.int32(Nmin),
        a,
        beta,
        alpha,
        np.int32(OUT_FLAG),
        np.int32(False),
        verbose,
    )

    return x


# Compute the positions and velocities of particles given a grid using LPT
def Displace_LPT(
    grid: np.ndarray,
    Lc: float = 2.0,
    Om0: float = 0.31,
    z: float = 0.0,
    k_smooth: float = 10000.0,
    DO_2LPT: bool = False,
    OUT_VEL: bool = False,
    Input_k: bool = False,
    OUT_POS: bool = True,
    verbose: bool = False,
) -> Dict[str, np.ndarray]:
    """
    Compute the displacement of particles using Lagrangian Perturbation Theory (LPT).

    :param grid: Density grid where the halos will be found.
    :type grid: numpy.ndarray
    :param Lc: Size of each cell in Mpc/h. Fiducial value: 2.0
    :type Lc: float
    :param Om0: Value of the matter overdensity today. Fiducial value: 0.31
    :type Om0: float
    :param z: Redshift of the density grid and final halo catalogue. Fiducial value: 0.0
    :type z: float
    :param k_smooth: Scale used to smooth the displacements. Fiducial value: 10000.0
    :type k_smooth: float
    :param DO_2LPT: Whether to use the second-order LPT. Fiducial value: False
    :type DO_2LPT: bool
    :param OUT_VEL: Whether to output the velocities of the particles. Fiducial value: False
    :type OUT_VEL: bool
    :param Input_k: Whether the input density grid is in real space (or Fourier). Fiducial value: False
    :type Input_k: bool
    :param OUT_POS: Whether to output the positions or just the displacements. Fiducial value: True
    :type OUT_POS: bool
    :param verbose: Whether to output information in the C code. Fiducial value: False
    :type verbose: bool

    :return: Dictionary with keys:

             - "pos": Ndarray with particle positions (displacements)
             - "vel": Ndarray with particle velocities (if OUT_VEL is True)
             
    :rtype: dict
    """
    # Check the precision and convert the arrays
    from .lib.exshalos import check_precision

    precision = check_precision()
    if precision == 4:
        grid = grid.astype("float32")
        Lc = np.float32(Lc)
        z = np.float32(z)
        k_smooth = np.float32(k_smooth)
    else:
        grid = grid.astype("float64")
        Lc = np.float64(Lc)
        z = np.float64(z)
        k_smooth = np.float64(k_smooth)

    # Call the C function to compute the displacements using LPT
    from .lib.exshalos import lpt_compute

    x = lpt_compute(
        grid,
        Lc,
        Om0,
        z,
        k_smooth,
        np.int32(DO_2LPT),
        np.int32(OUT_VEL),
        np.int32(Input_k),
        np.int32(OUT_POS),
        np.int32(verbose),
    )

    return x


# Fit the parameters of the barrier given a mass function
def Fit_Barrier(
    k: np.ndarray,
    P: np.ndarray,
    M: np.ndarray,
    dndlnM: np.ndarray,
    dn_err: Optional[np.ndarray] = None,
    grid: Optional[np.ndarray] = None,
    R_max: float = 100000.0,
    Mmin: Optional[float] = None,
    Mmax: Optional[float] = None,
    Nm: int = 25,
    nd: int = 256,
    Lc: float = 2.0,
    Om0: float = 0.31,
    z: float = 0.0,
    delta_c: Optional[float] = None,
    Nmin: int = 10,
    seed: int = 12345,
    x0: Optional[np.ndarray] = None,
    verbose: bool = False,
    nthreads: int = 1,
    Max_iter: int = 100,
    tol: Optional[float] = None,
) -> np.ndarray:
    """
    Fit the parameters of the barrier given a mass function.

    :param k: Wavenumbers of the power spectrum
    :type k: numpy.ndarray
    :param P: Power spectrum
    :type P: numpy.ndarray
    :param M: Mass of the mass function to be approximated.
    :type M: numpy.ndarray
    :param dndlnM: Differential mass function to be approximated.
    :type dndlnM: numpy.ndarray
    :param dn_err: Errors on the mass function. Fiducial value: None
    :type dn_err: Optional[numpy.ndarray]
    :param grid: Pre-computed Gaussian density grid. Fiducial value: None
    :type grid: Optional[numpy.ndarray]
    :param R_max: Maximum size used to compute the correlation function in Mpc/h. Fiducial value: 100000.0
    :type R_max: float
    :param Mmin: Minimum mass used to construct the mass bins. Fiducial value: None
    :type Mmin: Optional[float]
    :param Mmax: Maximum mass used to construct the mass bins. Fiducial value: None
    :type Mmax: Optional[float]
    :param Nm: Number of mass bins. Fiducial value: 25
    :type Nm: int
    :param nd: Number of cells in each direction. Fiducial value: 256
    :type nd: int
    :param Lc: Size of each cell in Mpc/h. Fiducial value: 2.0
    :type Lc: float
    :param Om0: Value of the matter overdensity today. Fiducial value: 0.31
    :type Om0: float
    :param z: Redshift of the density grid and final halo catalogue. Fiducial value: 0.0
    :type z: float
    :param delta_c: Critical density, for the halo formation, linearly extrapolated to z. Fiducial value: None
    :type delta_c: Optional[float]
    :param Nmin: Minimum number of particles in each halo. Fiducial value: 10
    :type Nmin: int
    :param seed: Seed used to generate the random numbers. Fiducial value: 12345
    :type seed: int
    :param x0: Initial guess for the parameters of the barrier. Fiducial value: None
    :type x0: Optional[numpy.ndarray]
    :param verbose: Whether to output  information in the C code. Fiducial value: False
    :type verbose: bool
    :param nthreads: Number of threads used by OpenMP. Fiducial value: 1
    :type nthreads: int
    :param Max_iter: Maximum number of iterations used in the minimization. Fiducial value: 100
    :type Max_iter: int
    :param tol: Tolerance for the minimization. Fiducial value: None
    :type tol: Optional[float]

    :return: Best fit of the values of the parameters of the ellipsoidal barrier.
    :rtype: numpy.ndarray
    """
    # Construct the Gaussian density grid
    if grid is None:
        grid = Generate_Density_Grid(
            k, P, R_max, nd=nd, Lc=Lc, seed=seed, verbose=False, nthreads=nthreads
        )["grid"]

    # Check if the mass function has an error
    if dn_err is None:
        dn_err = np.zeros(len(M))

    # Set Mmin and Mmax
    if Mmin is None:
        Mmin = -1.0
    if Mmax is None:
        Mmax = -1.0

    # Interpolate the given mass function
    from scipy.interpolate import interp1d

    fdn = interp1d(
        np.log(M[M > 0.0]), dndlnM[M > 0.0], bounds_error=False, fill_value=0.0
    )
    fdn_err = interp1d(
        np.log(M[M > 0.0]), dn_err[M > 0.0], bounds_error=False, fill_value=0.0
    )

    # Set the value of delta_c
    if delta_c is None:
        from .theory import Get_deltac

        delta_c = Get_deltac(z, Om0)

    # Define the function to be minimized to find the best parameters of the barrier
    def Chi2(theta):
        a, beta, alpha = theta

        x = Find_Halos_from_Grid(
            grid,
            k,
            P,
            Lc=Lc,
            Om0=Om0,
            z=z,
            delta_c=delta_c,
            Nmin=Nmin,
            a=a,
            beta=beta,
            alpha=alpha,
            verbose=False,
        )

        dnh = Compute_Abundance(
            x["Mh"], Mmin=Mmin, Mmax=Mmax, Nm=Nm, Lc=Lc, nd=nd, verbose=False
        )

        mask = dnh["dn"] > 0.0
        chi2 = np.sum(
            np.power((dnh["dn"][mask] - fdn(np.log(dnh["Mh"][mask]))), 2.0)
            / (
                np.power(dnh["dn_err"][mask], 2.0)
                + np.power(fdn_err(dnh["Mh"][mask]), 2.0)
            )
        ) / (Nm - 4)

        if verbose:
            print("Current try: (%f, %f, %f) with chi2 = %f" %
                  (a, beta, alpha, chi2))

        return chi2

    # Define the initial position
    if x0 is None:
        x0 = [0.55, 0.4, 0.7]

    # Minimize the Chi2 to get the best fit parameters
    from scipy.optimize import minimize
    from .simulation import Compute_Abundance

    bounds = [(0.1, 2.0), (0.0, 1.0), (0.0, 1.0)]
    x = minimize(
        Chi2,
        x0=x0,
        bounds=bounds,
        method="Nelder-Mead",
        options={"maxiter": Max_iter},
        tol=tol,
    )

    return x.x


# Fit the parameters of the HOD
def Fit_HOD(
    k: np.ndarray,
    P: np.ndarray,
    nbar: Optional[float] = None,
    posh: Optional[np.ndarray] = None,
    Mh: Optional[np.ndarray] = None,
    velh: Optional[np.ndarray] = None,
    Ch: Optional[np.ndarray] = None,
    nd: int = 256,
    ndx: Optional[int] = None,
    ndy: Optional[int] = None,
    ndz: Optional[int] = None,
    Lc: float = 2.0,
    Om0: float = 0.31,
    z: float = 0.0,
    x0: Optional[np.ndarray] = None,
    sigma: float = 0.5,
    Deltah: float = -1.0,
    seed: int = 12345,
    USE_VEL: bool = False,
    l_max: int = 0,
    direction: str = "z",
    window: Union[str, int] = "cic",
    R: float = 4.0,
    R_times: float = 5.0,
    interlacing: bool = True,
    Nk: int = 25,
    k_min: Optional[float] = None,
    k_max: Optional[float] = None,
    verbose: bool = False,
    nthreads: int = 1,
    Max_inter: int = 100,
    tol: Optional[float] = None,
) -> np.ndarray:
    """
    Fit the parameters of the Halo Occupation Distribution (HOD).

    :param k: Wavenumbers of the galaxy power spectrum.
    :type k: numpy.ndarray
    :param P: Galaxy power spectrum.
    :type P: numpy.ndarray
    :param nbar: Mean number density of galaxies. Fiducial value: None
    :type nbar: Optional[float]
    :param posh: Positions of the halos. Fiducial value: None
    :type posh: Optional[numpy.ndarray]
    :param Mh: Mass of the halos. Fiducial value: None
    :type Mh: Optional[numpy.ndarray]
    :param velh: Velocities of the halos. Fiducial value: None
    :type velh: Optional[numpy.ndarray]
    :param Ch: Concentration of the halos. Fiducial value: None
    :type Ch: Optional[numpy.ndarray]
    :param nd: Number of cells in each direction. Fiducial value: 256
    :type nd: int
    :param ndx: Number of cells in the x direction. Fiducial value: None
    :type ndx: Optional[int]
    :param ndy: Number of cells in the y direction. Fiducial value: None
    :type ndy: Optional[int]
    :param ndz: Number of cells in the z direction. Fiducial value: None
    :type ndz: Optional[int]
    :param Lc: Size of each cell in Mpc/h. Fiducial value: 2.0
    :type Lc: float
    :param Om0: Value of the matter overdensity today. Fiducial value: 0.31
    :type Om0: float
    :param z: Redshift of the density grid and final halo catalogue. Fiducial value: 0.0
    :type z: float
    :param x0: Initial guess for the best fit parameters of the HOD. Fiducial value: None
    :type x0: Optional[numpy.ndarray]
    :param sigma: Parameter of the exclusion term of the halo density profile. Fiducial value: 0.5
    :type sigma: float
    :param Deltah: Overdensity of the halos. Fiducial value: None
    :type Deltah: Optional[float]
    :param seed: Seed used to generate the density field. Fiducial value: 12345
    :type seed: int
    :param USE_VEL: Whether to use the power spectrum in redshift space. Fiducial value: False
    :type USE_VEL: bool
    :param l_max: Maximum multipole to consider. Fiducial value: 0
    :type l_max: int
    :param direction: Direction for redshift space distortions. Fiducial value: "z"
    :type direction: str
    :param window: Type of window function to use. Fiducial value: "cic"
    :type window: Union[str, int]
    :param R: Smoothing radius. Fiducial value: 4.0
    :type R: float
    :param R_times: Smoothing factor for the radius. Fiducial value: 5.0
    :type R_times: float
    :param interlacing: Whether to use interlacing to reduce aliasing effects. Fiducial value: True
    :type interlacing: bool
    :param Nk: Number of bins in k for the power spectrum. Fiducial value: 25
    :type Nk: int
    :param k_min: Minimum wavenumber for the power spectrum. Fiducial value: None
    :type k_min: Optional[float]
    :param k_max: Maximum wavenumber for the power spectrum. Fiducial value: None
    :type k_max: Optional[float]
    :param verbose: Whether to output information in the C code. Fiducial value: False
    :type verbose: bool
    :param nthreads: Number of threads used by OpenMP. Fiducial value: 1
    :type nthreads: int
    :param Max_inter: Maximum number of iterations used in the minimization. Fiducial value: 100
    :type Max_inter: int
    :param tol: Tolerance for the minimization. Fiducial value: None
    :type tol: Optional[float]

    :return: The best fit parameters of the HOD.
    :rtype: numpy.ndarray
    """
    # Interpolate the given power spectrum
    from scipy.interpolate import interp1d

    fP = interp1d(k, P)

    # Define the function to be minimized
    def Chi2(theta):
        logMmin, siglogM, logM0, logM1, alpha = theta

        gals = Generate_Galaxies_from_Halos(
            posh,
            Mh,
            velh=velh,
            Ch=Ch,
            nd=nd,
            ndx=ndx,
            ndy=ndy,
            ndz=ndz,
            Lc=Lc,
            Om0=Om0,
            z=z,
            logMmin=logMmin,
            siglogM=siglogM,
            logM0=logM0,
            logM1=logM1,
            alpha=alpha,
            sigma=sigma,
            Deltah=Deltah,
            seed=seed,
            OUT_VEL=USE_VEL,
            OUT_FLAG=False,
            verbose=verbose,
        )

        if USE_VEL:
            grid = Compute_Density_Grid(
                gals["posg"],
                vel=gals["velg"],
                mass=None,
                nd=nd,
                L=nd * Lc,
                Om0=Om0,
                z=z,
                direction=direction,
                window="CIC",
                R=R,
                R_times=R_times,
                interlacing=interlacing,
                verbose=verbose,
                nthreads=nthreads,
            )
        else:
            grid = Compute_Density_Grid(
                gals["posg"],
                vel=None,
                mass=None,
                nd=nd,
                L=nd * Lc,
                Om0=Om0,
                z=z,
                direction=None,
                window="CIC",
                R=R,
                R_times=R_times,
                interlacing=interlacing,
                verbose=verbose,
                nthreads=nthreads,
            )

        Pk = Compute_Power_Spectrum(
            grid,
            L=nd * Lc,
            window=window,
            R=R,
            Nk=Nk,
            k_min=k_min,
            k_max=k_max,
            l_max=l_max,
            verbose=verbose,
            nthreads=nthreads,
            ntypes=1,
            direction=direction,
        )

        if nbar is None:
            chi2 = (
                np.sum(
                    np.power(
                        (Pk["Pk"] - fP(Pk["k"])) /
                        (Pk["Pk"] / np.sqrt(Pk["Nk"])), 2.0
                    )
                )
            ) / (Nk - 6)
        else:
            chi2 = (
                np.sum(
                    np.power(
                        (Pk["Pk"] - fP(Pk["k"])) /
                        (Pk["Pk"] / np.sqrt(Pk["Nk"])), 2.0
                    )
                )
                + np.power(
                    (len(gals["posg"]) - nbar * (Lc * nd) ** 3)
                    / np.sqrt(len(gals["posg"])),
                    2.0,
                )
            ) / (Nk - 5)

        return chi2

    # Define the inital position
    if x0 is None:
        x0 = [13.25424743, 0.26461332, 13.28383025, 14.32465146, 1.00811277]

    # Minimaze the Chi2 to get the best fit parameters
    from scipy.optimize import minimize
    from .simulation import Compute_Power_Spectrum, Compute_Density_Grid
    from .mock import Generate_Galaxies_from_Halos

    bounds = [[9.0, 15.0], [0.0, 1.0], [9.0, 15.0], [9.0, 15.0], [0.0, 2.0]]
    x = minimize(
        Chi2,
        x0=x0,
        bounds=bounds,
        method="Nelder-Mead",
        options={"maxiter": Max_inter},
        tol=tol,
    )
    return x.x


# Compute the higher order local operators up to a given order


def Compute_High_Order_Operators(
    grid: np.ndarray,
    order: int = 2,
    nl_order: int = 0,
    Galileons: bool = False,
    Renormalized: bool = False,
    Lc: float = 2.0,
    verbose: bool = False,
    nthreads: int = 1
) -> Dict[str, np.ndarray]:
    """
    Compute the higher order operators up to a given order.

    :param grid: Lagrangian density grid.
    :type grid: numpy.ndarray
    :param order: Order to be used to compute the local operators. Fiducial value: 2
    :type order: int
    :param nl_order: Order to be used to compute the non-local operators. Fiducial value: 0
    :type nl_order: int
    :param Galileons: Whether to use the Galileons operators. Fiducial value: False
    :type Galileons: bool
    :param Renormalized: Whether to use renormalized operators. Fiducial value: False
    :type Renormalized: bool
    :param Lc: Size of each cell in Mpc/h. Fiducial value: 2.0
    :type Lc: float
    :param verbose: Whether to output information in the C code. Fiducial value: False
    :type verbose: bool
    :param nthreads: Number of threads used by OpenMP. Fiducial value: 1
    :type nthreads: int

    :return: Dictionary with keys:

             - "delta2": Ndarray with delta^2
             - "K2": Ndarray with K^2 or G_2
             - "delta3": Ndarray with delta^3
             - "K3": Ndarray with K^3 or G_3
             - "deltaK2": Ndarray with delta*K^2 or delta*G_2
             - "laplacian": Ndarray with Laplacian(delta)

    :rtype: dict
    """
    # Define the parameters used in the case (or not) of Galileons.
    # The operators are: K2 = K^2 - params[0]*delta^2, K3 = K_ij*K_jk*K_ki - params[1]*K^2*delta + params[2]*delta^3
    if not Galileons:
        params = np.array([1.0 / 3.0, 1.0, 2.0 / 9.0])
    else:
        params = np.array([1.0, 3.0 / 2.0, 1.0 / 2.0])

    # Check the precision and convert the arrays
    from .lib.exshalos import check_precision

    precision = check_precision()
    if precision == 4:
        grid = grid.astype("float32")
        params = params.astype("float32")
        Lc = np.float32(Lc)
    else:
        grid = grid.astype("float64")
        params = params.astype("float64")
        Lc = np.float64(Lc)

    # Call the c function to compute the operators
    from .lib.exshalos import operators_compute

    x = operators_compute(
        grid,
        np.int32(order),
        np.int32(nl_order),
        params,
        np.int32(Renormalized),
        Lc,
        np.int32(nthreads),
        np.int32(verbose)
    )

    return x

# Smooth a given field (or fields) in a given scale


def Smooth_Fields(
    grid: np.ndarray,
    Lc: float = 2.0,
    k_smooth: float = 10000.0,
    Input_k: bool = False,
    Nfields: int = 1,
    verbose: bool = False,
    nthreads: int = 1
) -> np.ndarray:
    """
    Smooth a given field (or fields) on a given scale.

    :param grid: Lagrangian density grid.
    :type grid: numpy.ndarray
    :param Lc: Size of each cell in Mpc/h. Fiducial value: 2.0
    :type Lc: float
    :param k_smooth: Scale used to smooth the fields. Fiducial value: 10000.0
    :type k_smooth: float
    :param Input_k: Whether the density grid is in real or Fourier space. Fiducial value: False
    :type Input_k: bool
    :param Nfields: Number of fields. Fiducial value: 1
    :type Nfields: int
    :param verbose: Whether to output information in the C code. Fiducial value: False
    :type verbose: bool
    :param nthreads: Number of threads used by OpenMP. Fiducial value: 1
    :type nthreads: int

    :return: A dictionary with all fields (excluding the linear) up to the given order or a single smoothed field.
    :rtype: np.ndarray 
    """
    # Check the precision and convert the arrays
    from .lib.exshalos import check_precision

    precision = check_precision()
    if precision == 4:
        grid = grid.astype("float32")
        Lc = np.float32(Lc)
        k_smooth = np.float32(k_smooth)
    else:
        grid = grid.astype("float64")
        Lc = np.float64(Lc)
        k_smooth = np.float64(k_smooth)

    # Call the c function to smooth the fields
    from .lib.exshalos import smooth_field

    if Nfields > 1:
        x = []
        for i in range(Nfields):
            x.append(
                smooth_field(
                    grid[i, :], Lc, k_smooth, np.int32(
                        Input_k), np.int32(nthreads), np.int32(verbose)
                )
            )
        return {"fields": np.array(x)}
    else:
        x = smooth_field(grid, Lc, k_smooth, np.int32(Input_k),
                         np.int32(nthreads), np.int32(verbose))

    return x


# Smooth a given field (or fields) in a given scale


def Smooth_and_Reduce_Fields(
    grid: np.ndarray,
    Lc: float = 2.0,
    k_smooth: float = 10000.0,
    Input_k: bool = False,
    Nfields: int = 1,
    verbose: bool = False,
    nthreads: int = 1
) -> np.ndarray:
    """
    Smooth a given field (or fields) on a given scale and reduce it.

    :param grid: Lagrangian density grid. 
    :type grid: numpy.ndarray
    :param Lc: Size of each cell in Mpc/h. Fiducial value: 2.0
    :type Lc: float
    :param k_smooth: Scale used to smooth the fields. Fiducial value: 10000.0
    :type k_smooth: float
    :param Input_k: Whether the density grid is in real or Fourier space. Fiducial value: False
    :type Input_k: bool
    :param Nfields: Number of fields. Fiducial value: 1
    :type Nfields: int
    :param verbose: Whether to output information in the C code. Fiducial value: False
    :type verbose: bool
    :param nthreads: Number of threads used by OpenMP. Fiducial value: 1
    :type nthreads: int

    :return: Smoothed and reduced fields.
    :rtype: numpy.ndarray
    """
    # Check the precision and convert the arrays
    from .lib.exshalos import check_precision

    precision = check_precision()
    if precision == 4:
        grid = grid.astype("float32")
        Lc = np.float32(Lc)
        k_smooth = np.float32(k_smooth)
    else:
        grid = grid.astype("float64")
        Lc = np.float64(Lc)
        k_smooth = np.float64(k_smooth)

    # Call the c function to smooth and reduce the fields
    from .lib.exshalos import smooth_and_reduce_field

    if Nfields > 1:
        x = []
        for i in range(Nfields):
            x.append(
                smooth_and_reduce_field(
                    grid[i, :], Lc, k_smooth, np.int32(
                        Input_k), np.int32(nthreads), np.int32(verbose)
                )
            )
        return np.array(x)
    else:
        x = smooth_and_reduce_field(
            grid, Lc, k_smooth, np.int32(Input_k), np.int32(
                nthreads), np.int32(verbose)
        )
        return x


# Compute the correlation function given the power spectrum or the power spectrum given the correlation function
def Compute_Correlation(
    k: np.ndarray,
    P: np.ndarray,
    direction: str = "pk2xi",
    verbose: bool = False
) -> dict:
    """
    Compute the correlation function given the power spectrum or the power spectrum given the correlation function.

    :param k: Wavenumbers of the power spectrum or the distance of the correlation function.
    :type k: numpy.ndarray
    :param P: Power spectrum or the correlation function.
    :type P: numpy.ndarray
    :param direction: Direction to compute the fftlog ("pk2xi" or "xi2pk"). Fiducial value: "pk2xi"
    :type direction: str
    :param verbose: Whether to output information in the C code. Fiducial value: False
    :type verbose: bool

    :return: Dictionary with keys:

             - "R" or "k": Ndarray with k or r 
             - "Xi" or "Pk": Ndarray with Xi(r) or P(k)

    :rtype: dict
    """
    # Check the precision and convert the arrays
    from .lib.analytical import check_precision

    precision = check_precision()
    if precision == 4:
        k = k.astype("float32")
        P = P.astype("float32")
    else:
        k = k.astype("float64")
        P = P.astype("float64")

    # Call the c function to compute the correlation function
    from .lib.analytical import correlation_compute

    x = correlation_compute(
        k, P, np.int32(direction), np.int32(verbose)
    )

    return x
