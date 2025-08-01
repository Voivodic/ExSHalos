"""
This module create mocks of halos and galaxies.
"""

# Import for annotations
from typing import Optional, cast

# Import numpy
import numpy as np
from numpy.typing import NDArray


# Generate a halo catalogue from a linear power spectrum
def generate_halos_box_from_pk(
    k: NDArray[np.floating],
    pk: NDArray[np.floating],
    r_max: float = 100000.0,
    nd: int = 256,
    nd_x: Optional[int] = None,
    nd_y: Optional[int] = None,
    nd_z: Optional[int] = None,
    cell_size: float = 2.0,
    omega_m0: float = 0.31,
    z: float = 0.0,
    k_smooth: float = 10000.0,
    delta_c: float = -1.0,
    n_min: int = 1,
    a: float = 1.0,
    beta: float = 0.0,
    alpha: float = 0.0,
    seed: int = 12345,
    fixed: bool = False,
    phase: float = 0.0,
    out_den: bool = False,
    out_lpt: bool = False,
    out_vel: bool = False,
    out_prof: bool = False,
    do_2lpt: bool = False,
    out_flag: bool = False,
    verbose: bool = False,
    n_threads: int = 1,
) -> dict[str, NDArray[np.floating]]:
    """
    Generates halos, in a box, from a power spectrum.

    :param k: Wavenumbers of the power spectrum.
    :type k: NDArray[np.floating]
    :param pk: Power spectrum.
    :type pk: NDArray[np.floating]
    :param r_max: Maximum size used to compute the correlation function.
                  Fiducial value: 100000.0 Mpc/h
    :type r_max: float
    :param nd: Number of cells in each direction. Fiducial value: 256
    :type nd: int
    :param nd_x: Number of cells in x direction. Fiducial value: nd
    :type nd_x: int
    :param nd_y: Number of cells in y direction. Fiducial value: nd
    :type nd_y: int
    :param nd_z: Number of cells in z direction. Fiducial value: nd
    :type nd_z: int
    :param cell_size: Size of each cell in Mpc/h. Fiducial value: 2.0
    :type cell_size: float
    :param omega_m0: Value of the matter overdensity today.
                     Fiducial value: 0.31
    :type omega_m0: float
    :param z: Redshift of the density grid and final halo catalogue.
              Fiducial value: 0.0
    :type z: float
    :param k_smooth: Scale used to smooth the LPT computations.
                     Fiducial value: 10000.0
    :type k_smooth: float
    :param delta_c: Critical density for the halo formation
                    linearly extrapolated to z. Fiducial value: -1.0
    :type delta_c: float
    :param n_min: Minimum number of particles in each halo. Fiducial value: 1
    :type n_min: int
    :param a: Parameter a of the ellipsoidal barrier. Fiducial value: 1.0
    :type a: float
    :param beta: Parameter beta of the ellipsoidal barrier. Fiducial value: 0.0
    :type beta: float
    :param alpha: Parameters alpha of the ellipsoidal barrier.
                  Fiducial value: 0.0
    :type alpha: float
    :param seed: Seed used to generate the density field. Fiducial value: 12345
    :type seed: int
    :param fixed: Whether to use fixed amplitudes of the Gaussian field.
                  Fiducial value: False
    :type fixed: bool
    :param phase: Phase of the Gaussian field. Fiducial value: 0.0
    :type phase: float
    :param out_den: Whether to output the density field. Fiducial value: False
    :type out_den: bool
    :param out_lpt: Whether to output the displaced particles.
                    Fiducial value: False
    :type out_lpt: bool
    :param out_vel: Whether to output the velocities of halos and particles.
                    Fiducial value: False
    :type out_vel: bool
    :param out_prof: Whether to output the density profile of the halos.
                     Fiducial value: False
    :type out_prof: bool
    :param do_2lpt: Whether to use the second order LPT
                    to displace the halos and particles. Fiducial value: False
    :type do_2lpt: bool
    :param out_flag: Whether to output the flag corresponding
                     to the host halo of each particle. Fiducial value: False
    :type out_flag: bool
    :param verbose: Whether to output information in the C code.
                    Fiducial value: False
    :type verbose: bool
    :param n_threads: Number of threads to be used in some computations.
                      Fiducial value: 1
    :type n_threads: int

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
    # Import the c functions
    from .lib.exshalos import (  # pyright: ignore[reportMissingImports]
        check_precision,  # pyright: ignore[reportUnknownVariableType]
        halos_box_from_pk,  # pyright: ignore[reportUnknownVariableType]
    )

    # Initialize c_* variables with their expected types
    c_k: NDArray[np.floating]
    c_pk: NDArray[np.floating]

    c_r_max: np.floating
    c_cell_size: np.floating
    c_omega_m0: np.floating
    c_z: np.floating
    c_k_smooth: np.floating
    c_delta_c: np.floating
    c_a: np.floating
    c_beta: np.floating
    c_alpha: np.floating
    c_phase: np.floating

    # Convert the inputs to the expected types
    precision = cast(int, check_precision())
    if precision == 4:
        c_k = k.astype("float32")
        c_pk = pk.astype("float32")
        c_r_max = np.float32(r_max)
        c_cell_size = np.float32(cell_size)
        c_omega_m0 = np.float32(omega_m0)
        c_z = np.float32(z)
        c_k_smooth = np.float32(k_smooth)
        c_delta_c = np.float32(delta_c)
        c_a = np.float32(a)
        c_beta = np.float32(beta)
        c_alpha = np.float32(alpha)
        c_phase = np.float32(phase)

    else:
        c_k = k.astype("float64")
        c_pk = pk.astype("float64")
        c_r_max = np.float64(r_max)
        c_cell_size = np.float64(cell_size)
        c_omega_m0 = np.float64(omega_m0)
        c_z = np.float64(z)
        c_k_smooth = np.float64(k_smooth)
        c_delta_c = np.float64(delta_c)
        c_a = np.float64(a)
        c_beta = np.float64(beta)
        c_alpha = np.float64(alpha)
        c_phase = np.float64(phase)

    # Define the number of cells in each direction
    if nd_x is None:
        nd_x = nd
    if nd_y is None:
        nd_y = nd
    if nd_z is None:
        nd_z = nd

    # Run the .C program to generate the halo catalogue from the power spectrum
    x = cast(
        dict[str, NDArray[np.floating]],
        halos_box_from_pk(
            c_k,
            c_pk,
            c_r_max,
            np.int32(nd_x),
            np.int32(nd_y),
            np.int32(nd_z),
            c_cell_size,
            np.int32(seed),
            c_k_smooth,
            c_omega_m0,
            c_z,
            c_delta_c,
            np.int32(n_min),
            c_a,
            c_beta,
            c_alpha,
            np.int32(fixed),
            c_phase,
            np.int32(out_den),
            np.int32(out_lpt),
            np.int32(out_vel),
            np.int32(do_2lpt),
            np.int32(out_flag),
            np.int32(out_prof),
            np.int32(verbose),
            np.int32(n_threads),
        ),
    )

    return x


# Generate a halo catalogue from a density grid
def generate_halos_box_from_grid(
    grid: NDArray[np.floating],
    k: NDArray[np.floating],
    pk: NDArray[np.floating],
    s: Optional[NDArray[np.floating]] = None,
    v: Optional[NDArray[np.floating]] = None,
    cell_size: float = 2.0,
    omega_m0: float = 0.31,
    z: float = 0.0,
    k_smooth: float = 10000.0,
    delta_c: float = -1.0,
    n_min: int = 1,
    a: float = 1.0,
    beta: float = 0.0,
    alpha: float = 0.0,
    out_lpt: bool = False,
    out_vel: bool = False,
    do_2lpt: bool = False,
    out_flag: bool = False,
    out_prof: bool = False,
    verbose: bool = False,
    n_threads: int = 1,
) -> dict[str, NDArray[np.floating]]:
    """
    Generates a halo catalogue, in a box, from a density grid.

    :param grid: Density grid used to generate the halos.
    :type grid: NDArray[np.floating]
    :param k: Wavenumbers of the power spectrum. Fiducial value.
    :type k: NDArray[np.floating]
    :param pk: Power spectrum.
    :type pk: NDArray[np.floating]
    :param s: Displacements of the particles in the grid.
              Fiducial value: None
    :type s: Optional[NDArray[np.floating]]
    :param v: Velocity of the particles in the grid.
              Fiducial value: None
    :type v: Optional[NDArray[np.floating]]
    :param cell_size: Size of each cell in Mpc/h. Fiducial value: 2.0
    :type cell_size: float
    :param omega_m0: Value of the matter overdensity today.
                     Fiducial value: 0.31
    :type omega_m0: float
    :param z: Redshift of the density grid and final halo
              catalogue. Fiducial value: 0.0
    :type z: float
    :param k_smooth: Scale used to smooth the LPT
                     computations. Fiducial value: 10000.0
    :type k_smooth: float
    :param delta_c: Critical density of the halo formation
                    linearly extrapolated to z. Fiducial value: -1.0
    :type delta_c: float
    :param n_min: Minimum number of particles in each halo.
                  Fiducial value: 1
    :type n_min: int
    :param a: Parameter a of the ellipsoidal barrier.
              Fiducial value: 1.0
    :type a: float
    :param beta: Parameter beta of the ellipsoidal barrier.
                 Fiducial value: 0.0
    :type beta: float
    :param alpha: Parameter alpha of the ellipsoidal barrier.
                  Fiducial value: 0.0
    :type alpha: float
    :param out_lpt: Whether to output the displaced particles.
                    Fiducial value: False
    :type out_lpt: bool
    :param out_vel: Whether to output the velocities of halos
                    and particles. Fiducial value: False
    :type out_vel: bool
    :param do_2lpt: Whether to use the second order LPT to
                    displace the halos and particles.
                    Fiducial value: False
    :type do_2lpt: bool
    :param out_flag: Whether to output the flag corresponding to
                     the host halo of each particle.
                     Fiducial value: False
    :type out_flag: bool
    :param out_prof: (Not working yet) Whether to output density
                     profiles of the Lagrangian halos.
                     Fiducial value: False
    :type out_prof: bool
    :param verbose: Whether to output information in the C code.
                    Fiducial value: False
    :type verbose: bool
    :param n_threads: Number of threads to be used in some
                      computations. Fiducial value: 1
    :type n_threads: int
    :return: Dictionary with the following keys:
             - "posh": ndarray with halo positions
             - "velh": ndarray with halo velocities
             - "Mh": ndarray with halo masses
             - "pos": ndarray with particle positions
             - "vel": ndarray with particle velocities
             - "flag": ndarray with particle flags

    :rtype: dict
    """
    # Check the dimensions of the displacements and velocities
    if s is not None:
        if s.shape[0] != grid.shape[0] * grid.shape[1] * grid.shape[2]:
            raise ValueError(
                "The number of particles in S is different of "
                "the number of grid cells!"
            )
        if s.shape[1] != 3:
            raise ValueError(
                "The number of components of S is different from 3!"
            )
        if v is not None:
            if v.shape[0] != grid.shape[0] * grid.shape[1] * grid.shape[2]:
                raise ValueError(
                    "The number of particles in V is different of "
                    "the number of grid cells!"
                )
            if v.shape[1] != 3:
                raise ValueError(
                    "The number of components of V is different from 3!"
                )

    # Check the precision and convert the arrays
    from .lib.exshalos import (  # pyright: ignore[reportMissingImports]
        check_precision,  # pyright: ignore[reportUnknownVariableType]
        halos_box_from_grid,  # pyright: ignore[reportUnknownVariableType]
    )

    # Initialize c_* variables with their expected types
    c_grid: NDArray[np.floating]
    c_k: NDArray[np.floating]
    c_pk: NDArray[np.floating]
    c_s: Optional[NDArray[np.floating]] = None
    c_v: Optional[NDArray[np.floating]] = None

    c_cell_size: np.floating
    c_omega_m0: np.floating
    c_z: np.floating
    c_k_smooth: np.floating
    c_delta_c: np.floating
    c_a: np.floating
    c_beta: np.floating
    c_alpha: np.floating

    # Convert the inputs to the expected types
    precision = cast(int, check_precision())
    if precision == 4:
        c_grid = grid.astype("float32")
        c_k = k.astype("float32")
        c_pk = pk.astype("float32")
        c_cell_size = np.float32(cell_size)
        c_omega_m0 = np.float32(omega_m0)
        c_z = np.float32(z)
        c_k_smooth = np.float32(k_smooth)
        c_delta_c = np.float32(delta_c)
        c_a = np.float32(a)
        c_beta = np.float32(beta)
        c_alpha = np.float32(alpha)
        if s is not None:
            c_s = s.astype("float32")
        if v is not None:
            c_v = v.astype("float32")
    else:
        c_grid = grid.astype("float64")
        c_k = k.astype("float64")
        c_pk = pk.astype("float64")
        c_cell_size = np.float64(cell_size)
        c_omega_m0 = np.float64(omega_m0)
        c_z = np.float64(z)
        c_k_smooth = np.float64(k_smooth)
        c_delta_c = np.float64(delta_c)
        c_a = np.float64(a)
        c_beta = np.float64(beta)
        c_alpha = np.float64(alpha)
        if s is not None:
            c_s = s.astype("float64")
        if v is not None:
            c_v = v.astype("float64")

    # Run the c function to generate the halo catalogue
    x = cast(
        dict[str, NDArray[np.floating]],
        halos_box_from_grid(
            c_k,
            c_pk,
            c_grid,
            c_s,
            c_v,
            c_cell_size,
            c_k_smooth,
            c_omega_m0,
            c_z,
            c_delta_c,
            np.int32(n_min),
            c_a,
            c_beta,
            c_alpha,
            np.int32(out_lpt),
            np.int32(out_vel),
            np.int32(do_2lpt),
            np.int32(out_flag),
            np.int32(s is not None),
            np.int32(out_prof),
            np.int32(verbose),
            np.int32(n_threads),
        ),
    )

    return x


# Populate a halo catalogue with galaxies
def generate_galaxies_from_halos(
    pos_h: NDArray[np.floating],
    m_h: NDArray[np.floating],
    vel_h: Optional[NDArray[np.floating]] = None,
    c_h: Optional[NDArray[np.floating]] = None,
    nd: int = 256,
    nd_x: int = 0,
    nd_y: int = 0,
    nd_z: int = 0,
    cell_size: float = 2.0,
    omega_m0: float = 0.31,
    z: float = 0.0,
    log_m_min: float = 13.25424743,
    sig_log_m: float = 0.26461332,
    log_m0: float = 13.28383025,
    log_m1: float = 14.32465146,
    alpha: float = 1.00811277,
    sigma: float = 0.5,
    delta_h: float = -1.0,
    seed: int = 12345,
    out_vel: bool = False,
    out_flag: bool = False,
    verbose: bool = False,
) -> dict[str, NDArray[np.floating]]:
    """
    Generate a galaxy catalogue from halo cataloguw.

    :param posh: Positions of the halos.
    :type posh: NDArray[np.floating]
    :param mh: Mass of the halos.
    :type mh: NDArray[np.floating]
    :param vel_h: Velocities of the halos. Fiducial value:
                  None
    :type vel_h: Optional[NDArray[np.floating]]
    :param ch: Concentration of the halos. Fiducial value:
               None
    :type c_h: Optional[NDArray[np.floating]]
    :param nd: Number of cells in each direction. Fiducial
               value: 256
    :type nd: int
    :param nd_x: Number of cells in the x direction. Fiducial
                 value: nd
    :type nd_x: int
    :param nd_y: Number of cells in the y direction. Fiducial
                 value: nd
    :type nd_y: int
    :param nd_z: Number of cells in the z direction. Fiducial
                 value: nd
    :type nd_z: int
    :param cell_size: Size of each cell in Mpc/h. Fiducial
                      value: 2.0
    :type cell_size: float
    :param omega_m0: Value of the matter overdensity today.
                     Fiducial value: 0.31
    :type omega_m0: float
    :param z: Redshift of the density grid and final galaxy
              catalogue. Fiducial value: 0.0
    :type z: float
    :param log_m_min: Parameter of the HOD models (Zheng
                      2005). Fiducial value: 13.25424743
    :type log_m_min: float
    :param sig_log_m: Parameter of the HOD models (Zheng
                      2005). Fiducial value: 0.26461332
    :type sig_log_m: float
    :param log_m0: Parameter of the HOD models (Zheng
                   2005). Fiducial value: 13.28383025
    :type log_m0: float
    :param log_m1: Parameter of the HOD models (Zheng
                   2005). Fiducial value: 14.32465146
    :type log_m1: float
    :param alpha: Parameter of the HOD models (Zheng
                  2005). Fiducial value: 1.00811277
    :type alpha: float
    :param sigma: Parameter of the exclusion term of the halo
                  density profile (Voivodic 2020). Fiducial
                  value: 0.5
    :type sigma: float
    :param delta_h: Overdensity of the halos. Fiducial value:
                    -1.0
    :type delta_h: float
    :param seed: Seed used to generate the random numbers.
                 Fiducial value: 12345
    :type seed: int
    :param out_vel: Whether to output the velocities of
                    galaxies. Fiducial value: False
    :type out_vel: bool
    :param out_flag: Whether to output the flag of galaxies
                     (central or satellite). Fiducial value:
                     False
    :type out_flag: bool
    :param verbose: Whether to output information in the C
                    code. Fiducial value: False
    :type verbose: bool

    :return: Dictionary with the following keys:
             - "posg": ndarray with galaxy positions
             - "velg": ndarray with galaxy velocities
             - "flag": ndarray with galaxy flags

    :rtype: dict
    """
    # Get the precision and convert the arrays
    from .lib.hod import (  # pyright: ignore[reportMissingImports]
        check_precision,  # pyright: ignore[reportUnknownVariableType]
        populate_halos,  # pyright: ignore[reportUnknownVariableType]
    )

    # Initialize c_* variables with their expected types
    c_pos_h: NDArray[np.floating]
    c_m_h: NDArray[np.floating]
    c_vel_h: Optional[NDArray[np.floating]] = None
    c_c_h: Optional[NDArray[np.floating]] = None

    c_cell_size: np.floating
    c_omega_m0: np.floating
    c_z: np.floating
    c_log_m_min: np.floating
    c_sig_log_m: np.floating
    c_log_m0: np.floating
    c_log_m1: np.floating
    c_alpha: np.floating
    c_sigma: np.floating
    c_delta_h: np.floating

    # Convert the inputs to the expected types
    precision = cast(int, check_precision())
    if precision == 4:
        c_pos_h = pos_h.astype("float32")
        if vel_h is not None:
            c_vel_h = vel_h.astype("float32")
        c_m_h = m_h.astype("float32")
        if c_h is not None:
            c_c_h = c_h.astype("float32")
        c_cell_size = np.float32(cell_size)
        c_omega_m0 = np.float32(omega_m0)
        c_z = np.float32(z)
        c_log_m_min = np.float32(log_m_min)
        c_sig_log_m = np.float32(sig_log_m)
        c_log_m0 = np.float32(log_m0)
        c_log_m1 = np.float32(log_m1)
        c_alpha = np.float32(alpha)
        c_sigma = np.float32(sigma)
        c_delta_h = np.float32(delta_h)

    else:
        c_pos_h = pos_h.astype("float64")
        if vel_h is not None:
            c_vel_h = vel_h.astype("float64")
        c_m_h = m_h.astype("float64")
        if c_h is not None:
            c_c_h = c_h.astype("float64")
        c_cell_size = np.float64(cell_size)
        c_omega_m0 = np.float64(omega_m0)
        c_z = np.float64(z)
        c_log_m_min = np.float64(log_m_min)
        c_sig_log_m = np.float64(sig_log_m)
        c_log_m0 = np.float64(log_m0)
        c_log_m1 = np.float64(log_m1)
        c_alpha = np.float64(alpha)
        c_sigma = np.float64(sigma)
        c_delta_h = np.float64(delta_h)

    # Define the number of cells in each direction
    if nd_x <= 0:
        nd_x = nd
    if nd_y <= 0:
        nd_y = nd
    if nd_z <= 0:
        nd_z = nd

    # Call the C function to populate the halos with galaxies
    x = cast(
        dict[str, NDArray[np.floating]],
        populate_halos(
            c_pos_h,
            c_vel_h,
            c_m_h,
            c_c_h,
            c_cell_size,
            c_omega_m0,
            c_z,
            np.int32(nd_x),
            np.int32(nd_y),
            np.int32(nd_z),
            c_log_m_min,
            c_sig_log_m,
            c_log_m0,
            c_log_m1,
            c_alpha,
            c_sigma,
            c_delta_h,
            np.int32(seed),
            np.int32(out_vel),
            np.int32(out_flag),
            np.int32(c_h is not None),
            np.int32(verbose),
        ),
    )

    return x


# Split the galaxies in two colors
def split_galaxies(
    m_h: NDArray[np.floating],
    flag: NDArray[np.integer],
    params_central: Optional[NDArray[np.floating]] = None,
    params_satellite: Optional[NDArray[np.floating]] = None,
    seed: int = 12345,
    verbose: bool = False,
) -> NDArray[np.integer]:
    """
    Split galaxies into central and satellite types based on their
    properties.

    :param mh: Mass of the halos.
    :type m_h: NDArray[np.floating]
    :param flag: Flag with the label splitting central and
                 satellites.
    :type flag: NDArray[np.integer]
    :param params_central: Parameters used to split the central
                           galaxies. Fiducial value:
                           [37.10265321, -5.07596644,
                           0.17497771]
    :type params_central: NDArray[np.floating]
    :param params_satellite: Parameters used to split the
                             satellite galaxies. Fiducial
                             value: [19.84341938, -2.8352781,
                             0.10443049]
    :type params_satellite: NDArray[np.floating]
    :param seed: Seed used to generate the random numbers.
                 Fiducial value: 12345
    :type seed: int
    :param verbose: Whether to output information in the C
                    code. Fiducial value: False
    :type verbose: bool

    :return: Type of each galaxy.
    :rtype: numpy.ndarray
    """
    # Check the precision and convert the arrays
    from .lib.hod import (  # pyright: ignore[reportMissingImports]
        check_precision,  # pyright: ignore[reportUnknownVariableType]
        split_galaxies,  # pyright: ignore[reportUnknownVariableType]
    )

    # Set the default parameters
    if params_central is None:
        params_central = np.array(
            [37.10265321, -5.07596644, 0.17497771], dtype=np.float64
        )
    if params_satellite is None:
        params_satellite = np.array(
            [19.84341938, -2.8352781, 0.10443049], dtype=np.float64
        )

    # Initialize c_* variables with their expected types
    c_m_h: NDArray[np.floating]
    c_flag: NDArray[np.integer]
    c_params_central: NDArray[np.floating]
    c_params_satellite: NDArray[np.floating]

    # Convert the inputs to the expected types
    precision = cast(int, check_precision())
    if precision == 4:
        c_m_h = m_h.astype("float32")
        c_flag = flag.astype("int32")
        c_params_central = params_central.astype("float32")
        c_params_satellite = params_satellite.astype("float32")
    else:
        c_m_h = m_h.astype("float64")
        c_flag = flag.astype("int32")
        c_params_central = params_central.astype("float64")
        c_params_satellite = params_satellite.astype("float64")

    # Add a dimension to the parameters if needed
    if len(c_params_central.shape) == 1:
        c_params_central = c_params_central.reshape([1, len(c_params_central)])
    if len(c_params_satellite.shape) == 1:
        c_params_satellite = c_params_satellite.reshape([
            1,
            len(c_params_satellite),
        ])

    # Check the number of tracers for the centrals and satellites
    if c_params_central.shape[0] != c_params_satellite.shape[0]:
        raise ValueError(
            "Different number of types for the centrals and satellites!"
            f"{c_params_central.shape[0]} != {c_params_satellite.shape[0]}!"
        )

    # Call the C function to split the galaxies in types
    x = cast(
        NDArray[np.integer],
        split_galaxies(
            c_m_h,
            c_flag,
            c_params_central,
            c_params_satellite,
            np.int32(seed),
            np.int32(verbose),
        ),
    )

    return x
