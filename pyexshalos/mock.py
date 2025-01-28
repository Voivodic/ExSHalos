"""
This module create mocks of halos and galaxies.
"""

from typing import Dict, Optional

import numpy as np


# Generate a halo catalogue from a linear power spectrum
def Generate_Halos_Box_from_Pk(
    k: np.ndarray,
    P: np.ndarray,
    R_max: float = 100000.0,
    nd: int = 256,
    ndx: int = 0,
    ndy: int = 0,
    ndz: int = 0,
    Lc: float = 2.0,
    Om0: float = 0.31,
    z: float = 0.0,
    k_smooth: float = 10000.0,
    delta_c: float = -1.0,
    Nmin: int = 1,
    a: float = 1.0,
    beta: float = 0.0,
    alpha: float = 0.0,
    seed: int = 12345,
    fixed: bool = False,
    phase: float = 0.0,
    OUT_DEN: bool = False,
    OUT_LPT: bool = False,
    OUT_VEL: bool = False,
    OUT_PROF: bool = False,
    DO_2LPT: bool = False,
    OUT_FLAG: bool = False,
    verbose: bool = False,
    nthreads: int = 1,
) -> Dict[str, np.ndarray]:
    """
    Generates halos, in a box, from a power spectrum.

    :param k: Wavenumbers of the power spectrum.
    :type k: numpy.ndarray
    :param P: Power spectrum.
    :type P: numpy.ndarray
    :param R_max: Maximum size used to compute the correlation function in Mpc/h. Fiducial value: 100000.0
    :type R_max: float
    :param nd: Number of cells in each direction. Fiducial value: 256
    :type nd: int
    :param ndx: Number of cells in x direction. Fiducial value: nd
    :type ndx: int
    :param ndy: Number of cells in y direction. Fiducial value: nd
    :type ndy: int
    :param ndz: Number of cells in z direction. Fiducial value: nd
    :type ndz: int
    :param Lc: Size of each cell in Mpc/h. Fiducial value: 2.0
    :type Lc: float
    :param Om0: Value of the matter overdensity today. Fiducial value: 0.31
    :type Om0: float
    :param z: Redshift of the density grid and final halo catalogue. Fiducial value: 0.0
    :type z: float
    :param k_smooth: Scale used to smooth the LPT computations. Fiducial value: 10000.0
    :type k_smooth: float
    :param delta_c: Critical density of the halo formation linearly extrapolated to z. Fiducial value: -1
    :type delta_c: float
    :param Nmin: Minimum number of particles in each halo. Fiducial value: 1
    :type Nmin: int
    :param a: Parameter a of the ellipsoidal barrier. Fiducial value: 1.0
    :type a: float
    :param beta: Parameter beta of the ellipsoidal barrier. Fiducial value: 0.0
    :type beta: float
    :param alpha: Parameters alpha of the ellipsoidal barrier. Fiducial value: 0.0
    :type alpha: float
    :param seed: Seed used to generate the density field. Fiducial value: 12345
    :type seed: int
    :param fixed: Whether to use fixed amplitudes of the Gaussian field. Fiducial value: False
    :type fixed: bool
    :param phase: Phase of the Gaussian field. Fiducial value: 0.0
    :type phase: float
    :param OUT_DEN: Whether to output the density field. Fiducial value: False
    :type OUT_DEN: bool
    :param OUT_LPT: Whether to output the displaced particles. Fiducial value: False
    :type OUT_LPT: bool
    :param OUT_VEL: Whether to output the velocities of halos and particles. Fiducial value: False
    :type OUT_VEL: bool
    :param OUT_PROF: Whether to output the density profile of the Lagrangian halos. Fiducial value: False
    :type OUT_PROF: bool
    :param DO_2LPT: Whether to use the second order LPT to displace the halos and particles. Fiducial value: False
    :type DO_2LPT: bool
    :param OUT_FLAG: Whether to output the flag corresponding to the host halo of each particle. Fiducial value: False
    :type OUT_FLAG: bool
    :param verbose: Whether to output information in the C code. Fiducial value: False
    :type verbose: bool
    :param nthreads: Number of threads to be used in some computations. Fiducial value: 1
    :type nthreads: int

    :return: Dictionary with the following keys:

             - "posh": ndarray with halo positions
             - "velh": ndarray with halo velocities
             - "Mh": ndarray with halo masses
             - "pos": ndarray with particle positions
             - "vel": ndarray with particle velocities
             - "flag": ndarray with particle flags
             - "grid": ndarray with the Gaussian density grid
             - "Prof": ndarray with the density profile of the Lagrangian halos
             - "ProfM": ndarray with the mass in each shell of the profile

    :rtype: dict
    """
    # Check the precision and convert the arrays
    from .lib.exshalos import check_precision

    precision = check_precision()
    if precision == 4:
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
        phase = np.float32(phase)

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

    # Define the number of cells in each direction
    if ndx <= 0:
        ndx = nd
    if ndy <= 0:
        ndy = nd
    if ndz <= 0:
        ndz = nd

    # Run the .C program to generate the halo catalogue
    from .lib.exshalos import halos_box_from_pk

    x = halos_box_from_pk(
        k,
        P,
        R_max,
        np.int32(ndx),
        np.int32(ndy),
        np.int32(ndz),
        Lc,
        np.int32(seed),
        k_smooth,
        Om0,
        z,
        delta_c,
        np.int32(Nmin),
        a,
        beta,
        alpha,
        np.int32(fixed),
        phase,
        np.int32(OUT_DEN),
        np.int32(OUT_LPT),
        np.int32(OUT_VEL),
        np.int32(DO_2LPT),
        np.int32(OUT_FLAG),
        np.int32(OUT_PROF),
        np.int32(verbose),
        np.int32(nthreads),
    )

    return x


# Generate a halo catalogue from a density grid
def Generate_Halos_Box_from_Grid(
    grid: np.ndarray,
    k: np.ndarray,
    P: np.ndarray,
    S: Optional[np.ndarray] = None,
    V: Optional[np.ndarray] = None,
    Lc: float = 2.0,
    Om0: float = 0.31,
    z: float = 0.0,
    k_smooth: float = 10000.0,
    delta_c: float = -1.0,
    Nmin: int = 1,
    a: float = 1.0,
    beta: float = 0.0,
    alpha: float = 0.0,
    OUT_LPT: bool = False,
    OUT_VEL: bool = False,
    DO_2LPT: bool = False,
    OUT_FLAG: bool = False,
    OUT_PROF: bool = False,
    verbose: bool = False,
    nthreads: int = 1,
) -> Dict[str, np.ndarray]:
    """
    Generates a halo catalogue, in a box, from a density grid.

    :param grid: Density grid used to generate the halos.
    :type grid: numpy.ndarray
    :param k: Wavenumbers of the power spectrum. Fiducial value.
    :type k: numpy.ndarray
    :param P: Power spectrum.
    :type P: numpy.ndarray
    :param S: Displacements of the particles in the grid. Fiducial value: None
    :type S: Optional[numpy.ndarray]
    :param V: Velocity of the particles in the grid. Fiducial value: None
    :type V: Optional[numpy.ndarray]
    :param Lc: Size of each cell in Mpc/h. Fiducial value: 2.0
    :type Lc: float
    :param Om0: Value of the matter overdensity today. Fiducial value: 0.31
    :type Om0: float
    :param z: Redshift of the density grid and final halo catalogue. Fiducial value: 0.0
    :type z: float
    :param k_smooth: Scale used to smooth the LPT computations. Fiducial value: 10000.0
    :type k_smooth: float
    :param delta_c: Critical density of the halo formation linearly extrapolated to z. Fiducial value: -1.0
    :type delta_c: float
    :param Nmin: Minimum number of particles in each halo. Fiducial value: 1
    :type Nmin: int
    :param a: Parameter a of the ellipsoidal barrier. Fiducial value: 1.0
    :type a: float
    :param beta: Parameter beta of the ellipsoidal barrier. Fiducial value: 0.0
    :type beta: float
    :param alpha: Parameter alpha of the ellipsoidal barrier. Fiducial value: 0.0
    :type alpha: float
    :param OUT_LPT: Whether to output the displaced particles. Fiducial value: False
    :type OUT_LPT: bool
    :param OUT_VEL: Whether to output the velocities of halos and particles. Fiducial value: False
    :type OUT_VEL: bool
    :param DO_2LPT: Whether to use the second order LPT to displace the halos and particles. Fiducial value: False
    :type DO_2LPT: bool
    :param OUT_FLAG: Whether to output the flag corresponding to the host halo of each particle. Fiducial value: False
    :type OUT_FLAG: bool
    :param OUT_PROF: (Not working yet) Whether to output density profiles of the Lagrangian halos. Fiducial value: False
    :type OUT_PROF: bool
    :param verbose: Whether to output information in the C code. Fiducial value: False
    :type verbose: bool
    :param nthreads: Number of threads to be used in some computations. Fiducial value: 1
    :type nthreads: int
    :return: Dictionary with the following keys:
    
             - "posh": ndarray with halo positions
             - "velh": ndarray with halo velocities
             - "Mh": ndarray with halo masses
             - "pos": ndarray with particle positions
             - "vel": ndarray with particle velocities
             - "flag": ndarray with particle flags

    :rtype: dict
    """
    # Check some of the imputs
    if S is None:
        In_disp = False
    else:
        In_disp = True

    if In_disp == True:
        if S.shape[0] != grid.shape[0] * grid.shape[1] * grid.shape[2]:
            raise ValueError(
                "The number of particles in S is different of the number of grid cells!"
            )
        if S.shape[1] != 3:
            raise ValueError("The number of components of S is different from 3!")
        if V is not None:
            if V.shape[0] != grid.shape[0] * grid.shape[1] * grid.shape[2]:
                raise ValueError(
                    "The number of particles in V is different of the number of grid cells!"
                )
            if V.shape[1] != 3:
                raise ValueError("The number of components of V is different from 3!")

    # Check the precision and convert the arrays
    from .lib.exshalos import check_precision

    precision = check_precision()
    if precision == 4:
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
        if S is not None:
            S = S.astype("float32")
        if V is not None:
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
        if S is not None:
            S = S.astype("float64")
        if V is not None:
            V = V.astype("float64")
    # Run the c function to generate the halo catalogue
    from .lib.exshalos import halos_box_from_grid

    x = halos_box_from_grid(
        k,
        P,
        grid,
        S,
        V,
        Lc,
        k_smooth,
        Om0,
        z,
        delta_c,
        np.int32(Nmin),
        a,
        beta,
        alpha,
        np.int32(OUT_LPT),
        np.int32(OUT_VEL),
        np.int32(DO_2LPT),
        np.int32(OUT_FLAG),
        np.int32(In_disp),
        np.int32(OUT_PROF),
        np.int32(verbose),
        np.int32(nthreads),
    )

    return x


# Populate a halo catalogue with galaxies
def Generate_Galaxies_from_Halos(
    posh: np.ndarray,
    Mh: np.ndarray,
    velh: Optional[np.ndarray] = None,
    Ch: Optional[np.ndarray] = None,
    nd: int = 256,
    ndx: int = 0,
    ndy: int = 0,
    ndz: int = 0,
    Lc: float = 2.0,
    Om0: float = 0.31,
    z: float = 0.0,
    logMmin: float = 13.25424743,
    siglogM: float = 0.26461332,
    logM0: float = 13.28383025,
    logM1: float = 14.32465146,
    alpha: float = 1.00811277,
    sigma: float = 0.5,
    Deltah: float = -1.0,
    seed: int = 12345,
    OUT_VEL: bool = False,
    OUT_FLAG: bool = False,
    verbose: bool = False,
) -> Dict[str, np.ndarray]:
    """
    Generate a galaxy catalogue from halo cataloguw.

    :param posh: Positions of the halos.
    :type posh: numpy.ndarray
    :param Mh: Mass of the halos.
    :type Mh: numpy.ndarray
    :param velh: Velocities of the halos. Fiducial value: None
    :type velh: Optional[numpy.ndarray]
    :param Ch: Concentration of the halos. Fiducial value: None
    :type Ch: Optional[numpy.ndarray]
    :param nd: Number of cells in each direction. Fiducial value: 256
    :type nd: int
    :param ndx: Number of cells in the x direction. Fiducial value: nd
    :type ndx: int
    :param ndy: Number of cells in the y direction. Fiducial value: nd
    :type ndy: int
    :param ndz: Number of cells in the z direction. Fiducial value: nd
    :type ndz: int
    :param Lc: Size of each cell in Mpc/h. Fiducial value: 2.0
    :type Lc: float
    :param Om0: Value of the matter overdensity today. Fiducial value: 0.31
    :type Om0: float
    :param z: Redshift of the density grid and final galaxy catalogue. Fiducial value: 0.0
    :type z: float
    :param logMmin: Parameter of the HOD models (Zheng 2005). Fiducial value: 13.25424743
    :type logMmin: float
    :param siglogM: Parameter of the HOD models (Zheng 2005). Fiducial value: 0.26461332
    :type siglogM: float
    :param logM0: Parameter of the HOD models (Zheng 2005). Fiducial value: 13.28383025
    :type logM0: float
    :param logM1: Parameter of the HOD models (Zheng 2005). Fiducial value: 14.32465146
    :type logM1: float
    :param alpha: Parameter of the HOD models (Zheng 2005). Fiducial value: 1.00811277
    :type alpha: float
    :param sigma: Parameter of the exclusion term of the halo density profile (Voivodic 2020). Fiducial value: 0.5
    :type sigma: float
    :param Deltah: Overdensity of the halos. Fiducial value: -1.0
    :type Deltah: float
    :param seed: Seed used to generate the random numbers. Fiducial value: 12345
    :type seed: int
    :param OUT_VEL: Whether to output the velocities of galaxies. Fiducial value: False
    :type OUT_VEL: bool
    :param OUT_FLAG: Whether to output the flag of galaxies (central or satellite). Fiducial value: False
    :type OUT_FLAG: bool
    :param verbose: Whether to output information in the C code. Fiducial value: False
    :type verbose: bool

    :return: Dictionary with the following keys:

             - "posg": ndarray with galaxy positions
             - "velg": ndarray with galaxy velocities
             - "flag": ndarray with galaxy flags

    :rtype: dict
    """
    # Get the precision and convert the arrays
    from .lib.hod import check_precision

    precision = check_precision()
    In_C = False
    if precision == 4:
        posh = posh.astype("float32")
        if velh is not None:
            velh = velh.astype("float32")
        Mh = Mh.astype("float32")
        if Ch is not None:
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
        if velh is not None:
            velh = velh.astype("float64")
        Mh = Mh.astype("float64")
        if Ch is not None:
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

    if ndx is None:
        ndx = nd
    if ndy is None:
        ndy = nd
    if ndz is None:
        ndz = nd

    # Call the C function to populate the halos with galaxies
    from .lib.hod import populate_halos

    x = populate_halos(
        posh,
        velh,
        Mh,
        Ch,
        Lc,
        Om0,
        z,
        np.int32(ndx),
        np.int32(ndy),
        np.int32(ndz),
        logMmin,
        siglogM,
        logM0,
        logM1,
        alpha,
        sigma,
        Deltah,
        np.int32(seed),
        np.int32(OUT_VEL),
        np.int32(OUT_FLAG),
        np.int32(In_C),
        np.int32(verbose),
    )

    return x


# Split the galaxies in two colors
def Split_Galaxies(
    Mh: np.ndarray,
    Flag: np.ndarray,
    params_cen: np.ndarray = np.array([37.10265321, -5.07596644, 0.17497771]),
    params_sat: np.ndarray = np.array([19.84341938, -2.8352781, 0.10443049]),
    seed: int = 12345,
    verbose: bool = False,
) -> np.ndarray:
    """
    Split galaxies into central and satellite types based on their properties.

    :param Mh: Mass of the halos.
    :type Mh: numpy.ndarray
    :param Flag: Flag with the label splitting central and satellites.
    :type Flag: numpy.ndarray
    :param params_cen: Parameters used to split the central galaxies. Fiducial value: [37.10265321, -5.07596644, 0.17497771]
    :type params_cen: numpy.ndarray
    :param params_sat: Parameters used to split the satellite galaxies. Fiducial value: [19.84341938, -2.8352781, 0.10443049]
    :type params_sat: numpy.ndarray
    :param seed: Seed used to generate the random numbers. Fiducial value: 12345
    :type seed: int
    :param verbose: Whether to output information in the C code. Fiducial value: False
    :type verbose: bool

    :return: Type of each galaxy.
    :rtype: numpy.ndarray
    """
    # Check the precision and convert the arrays
    from .lib.hod import check_precision

    precision = check_precision()
    if precision == 4:
        Mh = Mh.astype("float32")
        params_cen = params_cen.astype("float32")
        params_sat = params_sat.astype("float32")
    else:
        Mh = Mh.astype("float64")
        params_cen = params_cen.astype("float64")
        params_sat = params_sat.astype("float64")

    if len(params_cen.shape) == 1:
        params_cen = params_cen.reshape([1, len(params_cen)])
    if len(params_sat.shape) == 1:
        params_sat = params_sat.reshape([1, len(params_sat)])

    if params_cen.shape[0] != params_sat.shape[0]:
        raise ValueError(
            "Different number of types of galaxies for the centrals and satellites! %d != %d!"
            % (params_cen.shape[0], params_sat.shape[0])
        )
    # Call the C function to split the galaxies in types
    from .lib.hod import split_galaxies

    x = split_galaxies(
        Mh, Flag, params_cen, params_sat, np.int32(seed), np.int32(verbose)
    )

    return x
