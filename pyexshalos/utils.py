"""
This module compute quantities from data of simulations.
"""

# Import for annotations
from typing import Optional, cast

# Import numpy
import numpy as np
from numpy.typing import NDArray


# Compute the gaussian density grid given the power spectrum
def generate_density_grid(
    k: NDArray[np.floating],
    pk: NDArray[np.floating],
    r_max: float = 100000.0,
    nd: int = 256,
    nd_x: Optional[int] = None,
    nd_y: Optional[int] = None,
    nd_z: Optional[int] = None,
    cell_size: float = 2.0,
    out_k: bool = False,
    seed: int = 12345,
    fixed: bool = False,
    phase: float = 0.0,
    k_smooth: float = 100000.0,
    verbose: bool = False,
    n_threads: int = 1,
) -> NDArray[np.floating]:
    """
    Compute the Gaussian density grid given the power spectrum.

    :param k: Wavenumbers of the power spectrum.
    :type k: NDArray[np.floating]
    :param pk: Power spectrum.
    :type pk: NDArray[np.floating]
    :param r_max: Maximum size used to compute the correlation
                  function in Mpc/h. Fiducial value: 100000.0
    :type r_max: float
    :param nd: Number of cells per dimension. Fiducial value: 256
    :type nd: int
    :param nd_x: Number of cells in the x direction. Fiducial value:
                 None
    :type nd_x: Optional[int]
    :param nd_y: Number of cells in the y direction. Fiducial value:
                 None
    :type nd_y: Optional[int]
    :param nd_z: Number of cells in the z direction. Fiducial value:
                 None
    :type nd_z: Optional[int]
    :param cell_size: Size of each cell in Mpc/h. Fiducial value: 2.0
    :type cell_size: float
    :param out_k: Whether to return the density field in Fourier
                  space. Fiducial value: False
    :type out_k: bool
    :param seed: Seed used to generate the random numbers.
                 Fiducial value: 12345
    :type seed: int
    :param fixed: Whether to use fixed amplitudes. Fiducial value:
                  False
    :type fixed: bool
    :param phase: Phase of the density field. Fiducial value: 0.0
    :type phase: float
    :param k_smooth: Smoothing scale in k-space. Fiducial value:
                     100000.0
    :type k_smooth: float
    :param verbose: Whether to output information in the C code.
                    Fiducial value: False
    :type verbose: bool
    :param n_threads: Number of threads used by OpenMP. Fiducial
                      value: 1
    :type n_threads: int

    :return: The 3D density grid in real space (and in Fourier
             space if outk is True).
    :rtype: NDArray[np.floating]
    """
    # Import the C functions
    from .lib.exshalos import (  # pyright: ignore[reportMissingImports]
        check_precision,  # pyright: ignore[reportUnknownVariableType]
        density_grid_compute,  # pyright: ignore[reportUnknownVariableType]
    )

    # Initialize c_* variables with their expected types
    c_k: NDArray[np.floating]
    c_pk: NDArray[np.floating]

    c_r_max: np.floating
    c_cell_size: np.floating
    c_phase: np.floating
    c_k_smooth: np.floating

    # Check the precision and convert the arrays
    precision = cast(int, check_precision())
    if precision == 4:
        c_k = k.astype("float32")
        c_pk = pk.astype("float32")
        c_r_max = np.float32(r_max)
        c_cell_size = np.float32(cell_size)
        c_phase = np.float32(phase)
        c_k_smooth = np.float32(k_smooth)
    else:
        c_k = k.astype("float64")
        c_pk = pk.astype("float64")
        c_r_max = np.float64(r_max)
        c_cell_size = np.float64(cell_size)
        c_phase = np.float64(phase)
        c_k_smooth = np.float64(k_smooth)

    # Set the number of divisions per dimension
    if nd_x is None:
        nd_x = nd
    if nd_y is None:
        nd_y = nd
    if nd_z is None:
        nd_z = nd

    # Run the C function to compute the density field
    x = cast(
        NDArray[np.floating],
        density_grid_compute(
            c_k,
            c_pk,
            c_r_max,
            np.int32(nd_x),
            np.int32(nd_y),
            np.int32(nd_z),
            c_cell_size,
            np.int32(out_k),
            np.int32(seed),
            np.int32(fixed),
            c_phase,
            c_k_smooth,
            np.int32(verbose),
            np.int32(n_threads),
        ),
    )

    return x


# Generate a halo catalogue (in Lagrangian space) given an initial density grid
def find_halos_from_grid(
    grid: NDArray[np.floating],
    k: NDArray[np.floating],
    pk: NDArray[np.floating],
    cell_size: float = 2.0,
    omega_m0: float = 0.31,
    redshift: float = 0.0,
    delta_crit: Optional[float] = None,
    min_particles: int = 10,
    barrier_a: float = 1.0,
    barrier_beta: float = 0.0,
    barrier_alpha: float = 0.0,
    output_flag: bool = False,
    verbose: bool = False,
) -> dict[str, NDArray[np.floating]]:
    """
    Generate a halo catalogue (in Lagrangian space) given an initial
    density grid.

    :param grid: Density grid where the halos will be found.
    :type grid: NDArray[np.floating]
    :param k: Wavenumbers of the power spectrum.
    :type k: NDArray[np.floating]
    :param pk: Power spectrum.
    :type pk: NDArray[np.floating]
    :param cell_size: Size of each cell in Mpc/h. Fiducial value:
                      2.0
    :type cell_size: float
    :param omega_m0: Value of the matter overdensity today. Fiducial
                     value: 0.31
    :type omega_m0: float
    :param redshift: Redshift of the density grid and final halo
                     catalogue. Fiducial value: 0.0
    :type redshift: float
    :param delta_crit: Critical density of the halo formation
                       linearly extrapolated to z. Fiducial
                       value: None
    :type delta_crit: Optional[float]
    :param min_particles: Minimum number of particles in each halo.
                          Fiducial value: 10
    :type min_particles: int
    :param barrier_a: Parameter a of the ellipsoidal barrier.
                      Fiducial value: 1.0
    :type barrier_a: float
    :param barrier_beta: Parameter beta of the ellipsoidal barrier.
                         Fiducial value: 0.0
    :type barrier_beta: float
    :param barrier_alpha: Parameter alpha of the ellipsoidal barrier.
                          Fiducial value: 0.0
    :type barrier_alpha: float
    :param output_flag: Whether to output flag with the information
                        if a cell belongs to a halo. Fiducial
                        value: False
    :type output_flag: bool
    :param verbose: Whether to output information in the C code.
                    Fiducial value: False
    :type verbose: bool

    :return: Dictionary with keys:
             - "posh": Ndarray with halo positions
             - "Mh": Ndarray with halo masses
             - "flag": Ndarray with flags for each cell

    :rtype: Dict[str, NDArray[np.floating]]
    """
    # Import the C functions
    from .lib.exshalos import (  # pyright: ignore[reportMissingImports]
        check_precision,  # pyright: ignore[reportUnknownVariableType]
        find_halos,  # pyright: ignore[reportUnknownVariableType]
    )

    # Import Get_deltac if delta_crit needs to be computed
    if delta_crit is None:
        from .theory import get_deltac  # pyright: ignore[reportMissingImports]

        delta_crit = get_deltac(np.array(redshift), omega_m0).item()

    # Initialize c_* variables with their expected types
    c_grid: NDArray[np.floating]
    c_k: NDArray[np.floating]
    c_pk: NDArray[np.floating]

    c_cell_size: np.floating
    c_omega_m0: np.floating
    c_redshift: np.floating
    c_delta_crit: np.floating
    c_barrier_a: np.floating
    c_barrier_beta: np.floating
    c_barrier_alpha: np.floating

    # Check the precision and convert the arrays
    precision = cast(int, check_precision())
    if precision == 4:
        c_grid = grid.astype("float32")
        c_k = k.astype("float32")
        c_pk = pk.astype("float32")
        c_cell_size = np.float32(cell_size)
        c_omega_m0 = np.float32(omega_m0)
        c_redshift = np.float32(redshift)
        c_delta_crit = np.float32(delta_crit)
        c_barrier_a = np.float32(barrier_a)
        c_barrier_beta = np.float32(barrier_beta)
        c_barrier_alpha = np.float32(barrier_alpha)
    else:
        c_grid = grid.astype("float64")
        c_k = k.astype("float64")
        c_pk = pk.astype("float64")
        c_cell_size = np.float64(cell_size)
        c_omega_m0 = np.float64(omega_m0)
        c_redshift = np.float64(redshift)
        c_delta_crit = np.float32(delta_crit)
        c_barrier_a = np.float64(barrier_a)
        c_barrier_beta = np.float64(barrier_beta)
        c_barrier_alpha = np.float64(barrier_alpha)

    # Call the C function to compute the halo catalogue
    x = cast(
        dict[str, NDArray[np.floating]],
        find_halos(
            c_grid,
            c_k,
            c_pk,
            c_cell_size,
            c_omega_m0,
            c_redshift,
            c_delta_crit,
            np.int32(min_particles),
            c_barrier_a,
            c_barrier_beta,
            c_barrier_alpha,
            np.int32(output_flag),
            np.int32(False),
            np.int32(verbose),
        ),
    )

    return x


# Compute the positions and velocities of particles given a grid using LPT
def displace_lpt(
    grid: NDArray[np.floating],
    cell_size: float = 2.0,
    omega_m0: float = 0.31,
    redshift: float = 0.0,
    k_smooth: float = 10000.0,
    do_2lpt: bool = False,
    out_vel: bool = False,
    input_k: bool = False,
    out_pos: bool = True,
    verbose: bool = False,
) -> dict[str, NDArray[np.floating]]:
    """
    Compute the displacement of particles using Lagrangian Perturbation Theory.

    :param grid: Density grid where the halos will be found.
    :type grid: NDArray[np.floating]
    :param cell_size: Size of each cell in Mpc/h. Fiducial value: 2.0
    :type cell_size: float
    :param omega_m0: Value of the matter overdensity today.
                     Fiducial value: 0.31
    :type omega_m0: float
    :param redshift: Redshift of the density grid and final halo
                     catalogue. Fiducial value: 0.0
    :type redshift: float
    :param k_smooth: Scale used to smooth the displacements.
                     Fiducial value: 10000.0
    :type k_smooth: float
    :param do_2lpt: Whether to use the second-order LPT.
                    Fiducial value: False
    :type do_2lpt: bool
    :param out_vel: Whether to output the velocities of the particles.
                    Fiducial value: False
    :type out_vel: bool
    :param input_k: Whether the input density grid is in real space
                    (or Fourier). Fiducial value: False
    :type input_k: bool
    :param out_pos: Whether to output the positions or just the
                    displacements. Fiducial value: True
    :type out_pos: bool
    :param verbose: Whether to output information in the C code.
                    Fiducial value: False
    :type verbose: bool

    :return: Dictionary with keys:
             - "pos": Ndarray with particle positions (displacements)
             - "vel": Ndarray with particle velocities (if out_vel
                      is True)

    :rtype: Dict[str, NDArray[np.floating]]
    """
    # Import the C functions
    from .lib.exshalos import (  # pyright: ignore[reportMissingImports]
        check_precision,  # pyright: ignore[reportUnknownVariableType]
        lpt_compute,  # pyright: ignore[reportUnknownVariableType]
    )

    # Initialize c_* variables with their expected types
    c_grid: NDArray[np.floating]

    c_cell_size: np.floating
    c_omega_m0: np.floating
    c_redshift: np.floating
    c_k_smooth: np.floating

    # Check the precision and convert the arrays
    precision = cast(int, check_precision())
    if precision == 4:
        c_grid = grid.astype("float32")
        c_cell_size = np.float32(cell_size)
        c_omega_m0 = np.float32(omega_m0)
        c_redshift = np.float32(redshift)
        c_k_smooth = np.float32(k_smooth)
    else:
        c_grid = grid.astype("float64")
        c_cell_size = np.float64(cell_size)
        c_omega_m0 = np.float64(omega_m0)
        c_redshift = np.float64(redshift)
        c_k_smooth = np.float64(k_smooth)

    # Call the C function to compute the displacements using LPT
    x = cast(
        dict[str, NDArray[np.floating]],
        lpt_compute(
            c_grid,
            c_cell_size,
            c_omega_m0,
            c_redshift,
            c_k_smooth,
            np.int32(do_2lpt),
            np.int32(out_vel),
            np.int32(input_k),
            np.int32(out_pos),
            np.int32(verbose),
        ),
    )

    return x


# Fit the parameters of the barrier given a mass function
def fit_barrier(
    k: NDArray[np.floating],
    pk: NDArray[np.floating],
    masses: NDArray[np.floating],
    dndlnm: NDArray[np.floating],
    dn_err: Optional[NDArray[np.floating]] = None,
    grid: Optional[NDArray[np.floating]] = None,
    r_max: float = 100000.0,
    m_min: Optional[float] = None,
    m_max: Optional[float] = None,
    n_m_bins: int = 25,
    nd: int = 256,
    cell_size: float = 2.0,
    omega_m0: float = 0.31,
    redshift: float = 0.0,
    delta_crit: Optional[float] = None,
    min_particles: int = 10,
    seed: int = 12345,
    x0: Optional[NDArray[np.floating]] = None,
    verbose: bool = False,
    n_threads: int = 1,
    max_iter: int = 100,
    tol: Optional[float] = None,
) -> NDArray[np.floating]:
    """
    Fit the parameters of the barrier given a mass function.

    :param k: Wavenumbers of the power spectrum.
    :type k: NDArray[np.floating]
    :param pk: Power spectrum.
    :type pk: NDArray[np.floating]
    :param masses: Mass of the mass function to be approximated.
    :type masses: NDArray[np.floating]
    :param dndlnm: Differential mass function to be approximated.
    :type dndlnm: NDArray[np.floating]
    :param dn_err: Errors on the mass function. Fiducial
                   value: None
    :type dn_err: Optional[NDArray[np.floating]]
    :param grid: Pre-computed Gaussian density grid. Fiducial
                 value: None
    :type grid: Optional[NDArray[np.floating]]
    :param r_max: Maximum size used to compute the correlation
                  function in Mpc/h. Fiducial value: 100000.0
    :type r_max: float
    :param m_min: Minimum mass used to construct the mass bins.
                  Fiducial value: None
    :type m_min: Optional[float]
    :param m_max: Maximum mass used to construct the mass bins.
                  Fiducial value: None
    :type m_max: Optional[float]
    :param n_m_bins: Number of mass bins. Fiducial value: 25
    :type n_m_bins: int
    :param nd: Number of cells in each direction. Fiducial
               value: 256
    :type nd: int
    :param cell_size: Size of each cell in Mpc/h. Fiducial
                      value: 2.0
    :type cell_size: float
    :param omega_m0: Value of the matter overdensity today.
                     Fiducial value: 0.31
    :type omega_m0: float
    :param redshift: Redshift of the density grid and final
                     halo catalogue. Fiducial value: 0.0
    :type redshift: float
    :param delta_crit: Critical density, for the halo formation,
                       linearly extrapolated to z. Fiducial
                       value: None
    :type delta_crit: Optional[float]
    :param min_particles: Minimum number of particles in each halo.
                          Fiducial value: 10
    :type min_particles: int
    :param seed: Seed used to generate the random numbers.
                 Fiducial value: 12345
    :type seed: int
    :param x0: Initial guess for the parameters of the barrier.
               Fiducial value: None
    :type x0: Optional[NDArray[np.floating]]
    :param verbose: Whether to output information in the C code.
                    Fiducial value: False
    :type verbose: bool
    :param n_threads: Number of threads used by OpenMP. Fiducial
                      value: 1
    :type n_threads: int
    :param max_iter: Maximum number of iterations used in the
                     minimization. Fiducial value: 100
    :type max_iter: int
    :param tol: Tolerance for the minimization. Fiducial value:
                None
    :type tol: Optional[float]

    :return: Best fit of the values of the parameters of the
             ellipsoidal barrier.
    :rtype: NDArray[np.floating]
    """
    # Import modules needed for the function.
    from scipy.interpolate import interp1d
    from scipy.optimize import (
        minimize,  # pyright: ignore[reportUnknownVariableType]
    )

    from .lib.exshalos import (  # pyright: ignore[reportMissingImports]
        check_precision,  # pyright: ignore[reportUnknownVariableType]
    )
    from .simulation import (
        compute_abundance,  # pyright: ignore[reportMissingImports]
    )
    from .theory import get_deltac  # pyright: ignore[reportMissingImports]

    # Check the precision to determine numpy dtype for consistency
    precision = cast(int, check_precision())
    dtype = np.float32 if precision == 4 else np.float64

    # Construct the Gaussian density grid if not provided.
    # The `generate_density_grid` function (defined earlier in this file)
    # handles its own internal c_* conversions.
    if grid is None:
        grid = generate_density_grid(
            k,
            pk,
            r_max,
            nd=nd,
            cell_size=cell_size,
            seed=seed,
            verbose=False,
            n_threads=n_threads,
        )

    # Check if the mass function has an error, otherwise initialize to zeros
    if dn_err is None:
        dn_err = np.zeros(len(masses), dtype=masses.dtype)

    # Set m_min and m_max if they are None
    if m_min is None:
        m_min = -1.0
    if m_max is None:
        m_max = -1.0

    # Interpolate the given mass function and its errors
    f_dn = interp1d(
        np.log(masses[masses > 0.0]),
        dndlnm[masses > 0.0],
        bounds_error=False,
        fill_value=0.0,
    )
    f_dn_err = interp1d(
        np.log(masses[masses > 0.0]),
        dn_err[masses > 0.0],
        bounds_error=False,
        fill_value=0.0,
    )

    # Set the value of delta_crit if not provided
    if delta_crit is None:
        delta_crit = get_deltac(np.array(redshift), omega_m0).item()

    # Define the function to be minimized
    # to find the best parameters of the barrier
    def chi2_func(barrier_params: NDArray[np.floating]) -> float:
        a, beta, alpha = barrier_params

        # Call the find_halos_from_grid wrapper function.
        # This wrapper (defined earlier in this file)
        # handles its own internal c_* conversions.
        halo_catalogue = find_halos_from_grid(
            grid,
            k,
            pk,
            cell_size=cell_size,
            omega_m0=omega_m0,
            redshift=redshift,
            delta_crit=delta_crit,
            min_particles=min_particles,
            barrier_a=float(a),
            barrier_beta=float(beta),
            barrier_alpha=float(alpha),
            verbose=False,
        )

        # Compute the abundance from the found halos.
        dnh = compute_abundance(
            halo_mass=halo_catalogue["Mh"],
            min_mass=m_min,
            max_mass=m_max,
            num_mass_bins=n_m_bins,
            cell_size=cell_size,
            nd=nd,
            verbose=False,
        )

        # Calculate chi-squared
        mask = dnh["dn"] > 0.0

        # Calculate terms for chi-squared
        predicted_dndlnm_at_halo_masses = cast(
            NDArray[np.floating], f_dn(np.log(dnh["Mh"][mask]))
        )
        errors_on_predicted_dndlnm = cast(
            NDArray[np.floating], f_dn_err(dnh["Mh"][mask])
        )

        numerator_term = np.power(
            dnh["dn"][mask] - predicted_dndlnm_at_halo_masses,
            2.0,
        )
        denominator_term = np.power(dnh["dn_err"][mask], 2.0) + np.power(
            errors_on_predicted_dndlnm, 2.0
        )

        # Sum the chi-squared components and normalize by degrees of freedom
        chi2 = np.sum(numerator_term / denominator_term) / (n_m_bins - 4)

        if verbose:
            print(
                f"Current try: ({a:.6f}, {beta:.6f}, {alpha:.6f}) "
                f"with chi2 = {chi2:.6f}"
            )

        return float(chi2)

    # Define the initial guess for the parameters of the barrier
    if x0 is None:
        x0 = np.array(
            [0.55, 0.4, 0.7], dtype=dtype
        )  # Use determined dtype for consistency

    # Minimize the chi2_func to get the best fit parameters
    bounds = [(0.1, 2.0), (0.0, 1.0), (0.0, 1.0)]  # Bounds for a, beta, alpha
    best_fit = cast(
        NDArray[np.floating],
        minimize(  # pyright: ignore[reportUnknownMemberType]
            fun=chi2_func,
            x0=x0,
            bounds=bounds,
            method="Nelder-Mead",
            options={"maxiter": max_iter},
            tol=tol,
        ).x,
    )

    # Return the best-fit parameters, cast to NDArray[np.floating]
    return best_fit


# Fit the parameters of the HOD
def pk_fit_hod(
    k: NDArray[np.floating],
    pk: NDArray[np.floating],
    halo_positions: NDArray[np.floating],
    halo_masses: NDArray[np.floating],
    n_bar: Optional[float] = None,
    halo_velocities: Optional[NDArray[np.floating]] = None,
    halo_concentrations: Optional[NDArray[np.floating]] = None,
    nd: int = 256,
    nd_x: Optional[int] = None,
    nd_y: Optional[int] = None,
    nd_z: Optional[int] = None,
    cell_size: float = 2.0,
    omega_m0: float = 0.31,
    redshift: float = 0.0,
    x0: Optional[NDArray[np.floating]] = None,
    sigma: float = 0.5,
    delta_h: float = -1.0,
    seed: int = 12345,
    use_velocities: bool = False,
    l_max: int = 0,
    direction: str = "z",
    window: str | int = "cic",
    smoothing_radius: float = 4.0,
    smoothing_factor: float = 5.0,
    interlacing: bool = True,
    n_k_bins: int = 25,
    k_min: Optional[float] = None,
    k_max: Optional[float] = None,
    verbose: bool = False,
    n_threads: int = 1,
    max_iterations: int = 100,
    tol: Optional[float] = None,
) -> NDArray[np.floating]:
    """
    Fit the parameters of the Halo Occupation Distribution (HOD).

    :param k: Wavenumbers of the galaxy power spectrum.
    :type k: NDArray[np.floating]
    :param pk: Galaxy power spectrum.
    :type pk: NDArray[np.floating]
    :param n_bar: Mean number density of galaxies. Fiducial
                  value: None
    :type n_bar: Optional[float]
    :param halo_positions: Positions of the halos. Fiducial
                           value: None
    :type halo_positions: Optional[NDArray[np.floating]]
    :param halo_masses: Mass of the halos. Fiducial value:
                        None
    :type halo_masses: Optional[NDArray[np.floating]]
    :param halo_velocities: Velocities of the halos. Fiducial
                            value: None
    :type halo_velocities: Optional[NDArray[np.floating]]
    :param halo_concentrations: Concentration of the halos.
                                Fiducial value: None
    :type halo_concentrations: Optional[NDArray[np.floating]]
    :param nd: Number of cells in each direction. Fiducial
               value: 256
    :type nd: int
    :param nd_x: Number of cells in the x direction. Fiducial
                 value: None
    :type nd_x: Optional[int]
    :param nd_y: Number of cells in the y direction. Fiducial
                 value: None
    :type nd_y: Optional[int]
    :param nd_z: Number of cells in the z direction. Fiducial
                 value: None
    :type nd_z: Optional[int]
    :param cell_size: Size of each cell in Mpc/h. Fiducial
                      value: 2.0
    :type cell_size: float
    :param omega_m0: Value of the matter overdensity today.
                     Fiducial value: 0.31
    :type omega_m0: float
    :param redshift: Redshift of the density grid and final
                     halo catalogue. Fiducial value: 0.0
    :type redshift: float
    :param x0: Initial guess for the best fit parameters of
               the HOD. Fiducial value: None
    :type x0: Optional[NDArray[np.floating]]
    :param sigma: Parameter of the exclusion term of the halo
                  density profile. Fiducial value: 0.5
    :type sigma: float
    :param delta_h: Overdensity of the halos. Fiducial value:
                    -1.0
    :type delta_h: float
    :param seed: Seed used to generate the density field.
                 Fiducial value: 12345
    :type seed: int
    :param use_velocities: Whether to use the power spectrum
                           in redshift space. Fiducial
                           value: False
    :type use_velocities: bool
    :param l_max: Maximum multipole to consider. Fiducial
                  value: 0
    :type l_max: int
    :param direction: Direction for redshift space distortions.
                      Fiducial value: "z"
    :type direction: str
    :param window: Type of window function to use. Fiducial
                   value: "cic"
    :type window: str | int
    :param smoothing_radius: Smoothing radius. Fiducial value:
                             4.0
    :type smoothing_radius: float
    :param smoothing_factor: Smoothing factor for the radius.
                             Fiducial value: 5.0
    :type smoothing_factor: float
    :param interlacing: Whether to use interlacing to reduce
                        aliasing effects. Fiducial value:
                        True
    :type interlacing: bool
    :param n_k_bins: Number of bins in k for the power
                     spectrum. Fiducial value: 25
    :type n_k_bins: int
    :param k_min: Minimum wavenumber for the power spectrum.
                  Fiducial value: None
    :type k_min: Optional[float]
    :param k_max: Maximum wavenumber for the power spectrum.
                  Fiducial value: None
    :type k_max: Optional[float]
    :param verbose: Whether to output information in the C
                    code. Fiducial value: False
    :type verbose: bool
    :param n_threads: Number of threads used by OpenMP.
                      Fiducial value: 1
    :type n_threads: int
    :param max_iterations: Maximum number of iterations used
                           in the minimization. Fiducial
                           value: 100
    :type max_iterations: int
    :param tol: Tolerance for the minimization. Fiducial
                value: None
    :type tol: Optional[float]

    :return: The best fit parameters of the HOD.
    :rtype: NDArray[np.floating]
    """
    # Import modules needed for the function.
    from scipy.interpolate import interp1d
    from scipy.optimize import (
        minimize,  # pyright: ignore[reportUnknownVariableType]
    )

    from .mock import generate_galaxies_from_halos
    from .simulation import (
        compute_density_grid,
        compute_power_spectrum,
    )

    # Set the number of divisions per dimension
    if nd_x is None:
        nd_x = nd
    if nd_y is None:
        nd_y = nd
    if nd_z is None:
        nd_z = nd

    # Interpolate the given power spectrum
    power_spectrum_interpolator = interp1d(
        k, pk, bounds_error=False, fill_value=0.0
    )

    # Define the function to be minimized
    def chi2_func(theta: NDArray[np.floating]) -> float:
        log_m_min, sig_log_m, log_m0, log_m1, alpha = theta

        galaxy_catalogue = (
            generate_galaxies_from_halos(
                pos_h=halo_positions,
                m_h=halo_masses,
                vel_h=halo_velocities,
                c_h=halo_concentrations,
                nd=nd,
                nd_x=nd_x,
                nd_y=nd_y,
                nd_z=nd_z,
                cell_size=cell_size,
                omega_m0=omega_m0,
                z=redshift,
                log_m_min=log_m_min,
                sig_log_m=sig_log_m,
                log_m0=log_m0,
                log_m1=log_m1,
                alpha=alpha,
                sigma=sigma,
                delta_h=delta_h,
                seed=seed,
                out_vel=use_velocities,
                out_flag=False,
                verbose=verbose,
            ),
        )[0]
        box_size = nd * cell_size

        density_grid: NDArray[np.floating]
        if use_velocities:
            density_grid = (
                compute_density_grid(
                    pos=galaxy_catalogue["posg"],
                    vel=galaxy_catalogue["velg"],
                    mass=cast(Optional[NDArray[np.floating]], None),
                    nd=nd,
                    box_size=box_size,
                    omega_m0=omega_m0,
                    z=redshift,
                    direction=direction,
                    window=window,
                    r=smoothing_radius,
                    r_times=smoothing_factor,
                    interlacing=interlacing,
                    verbose=verbose,
                    n_threads=n_threads,
                ),
            )[0]

        else:
            density_grid = (
                compute_density_grid(
                    galaxy_catalogue["posg"],
                    vel=cast(Optional[NDArray[np.floating]], None),
                    mass=cast(Optional[NDArray[np.floating]], None),
                    nd=nd,
                    box_size=box_size,
                    omega_m0=omega_m0,
                    z=redshift,
                    direction=direction,
                    window=window,
                    r=smoothing_radius,
                    r_times=smoothing_factor,
                    interlacing=interlacing,
                    verbose=verbose,
                    n_threads=n_threads,
                ),
            )[0]

        power_spectrum_result = (
            compute_power_spectrum(
                grid=density_grid,
                box_size=box_size,
                window=window,
                r=smoothing_radius,
                n_k=n_k_bins,
                k_min=k_min,
                k_max=k_max,
                l_max=l_max,
                verbose=verbose,
                n_threads=n_threads,
                n_types=1,
                direction=direction,
            ),
        )[0]

        calculated_power_spectrum = power_spectrum_result["Pk"]
        calculated_k = power_spectrum_result["k"]
        calculated_num_k_modes = power_spectrum_result["Nk"]

        # Predict P(k) at the calculated k values using the interpolator
        predicted_power_spectrum = cast(
            NDArray[np.floating], power_spectrum_interpolator(calculated_k)
        )

        if n_bar is None:
            chi2 = np.sum(
                np.power(
                    (calculated_power_spectrum - predicted_power_spectrum)
                    / (
                        calculated_power_spectrum
                        / np.sqrt(calculated_num_k_modes)
                    ),
                    2.0,
                )
            ) / (n_k_bins - 6)
        else:
            galaxy_count = float(len(galaxy_catalogue["posg"]))
            expected_galaxy_count = n_bar * float((cell_size * nd) ** 3)

            chi2 = (
                np.sum(
                    np.power(
                        (calculated_power_spectrum - predicted_power_spectrum)
                        / (
                            calculated_power_spectrum
                            / np.sqrt(calculated_num_k_modes)
                        ),
                        2.0,
                    )
                )
                + np.power(
                    (galaxy_count - expected_galaxy_count)
                    / np.sqrt(galaxy_count),
                    2.0,
                )
            ) / (n_k_bins - 5)

        return float(chi2)

    # Define the initial guess for the parameters of the HOD
    if x0 is None:
        x0 = np.array([
            13.25424743,
            0.26461332,
            13.28383025,
            14.32465146,
            1.00811277,
        ])

    # Minimize the chi2_func to get the best fit parameters
    bounds = [(9.0, 15.0), (0.0, 1.0), (9.0, 15.0), (9.0, 15.0), (0.0, 2.0)]
    best_fit_params = cast(
        NDArray[np.floating],
        minimize(  # pyright: ignore[reportUnknownMemberType]
            fun=chi2_func,
            x0=x0,
            bounds=bounds,
            method="Nelder-Mead",
            options={"maxiter": max_iterations},
            tol=tol,
        ).x,
    )
    return best_fit_params


# Compute the higher order local operators up to a given order
def compute_higher_order_operators(
    grid: NDArray[np.floating],
    order: int = 2,
    nl_order: int = 0,
    galileons: bool = False,
    renormalized: bool = False,
    cell_size: float = 2.0,
    verbose: bool = False,
    n_threads: int = 1,
) -> dict[str, NDArray[np.floating]]:
    """
    Compute the higher order operators up to a given order.

    :param grid: Lagrangian density grid.
    :type grid: NDArray[np.floating]
    :param order: Order to be used to compute the local operators.
                  Fiducial value: 2
    :type order: int
    :param nl_order: Order to be used to compute the non-local
                     operators. Fiducial value: 0
    :type nl_order: int
    :param galileons: Whether to use the Galileons operators.
                      Fiducial value: False
    :type galileons: bool
    :param renormalized: Whether to use renormalized operators.
                         Fiducial value: False
    :type renormalized: bool
    :param cell_size: Size of each cell in Mpc/h. Fiducial
                      value: 2.0
    :type cell_size: float
    :param verbose: Whether to output information in the C code.
                    Fiducial value: False
    :type verbose: bool
    :param n_threads: Number of threads used by OpenMP. Fiducial
                      value: 1
    :type n_threads: int

    :return: Dictionary with keys:
             - "delta2": Ndarray with delta^2
             - "K2": Ndarray with K^2 or G_2
             - "delta3": Ndarray with delta^3
             - "K3": Ndarray with K^3 or G_3
             - "deltaK2": Ndarray with delta*K^2 or delta*G_2
             - "laplacian": Ndarray with Laplacian(delta)

    :rtype: Dict[str, NDArray[np.floating]]
    """
    # Import the C functions
    from .lib.exshalos import (  # pyright: ignore[reportMissingImports]
        check_precision,  # pyright: ignore[reportUnknownVariableType]
        operators_compute,  # pyright: ignore[reportUnknownVariableType]
    )

    # Define the parameters used in the case (or not) of Galileons.
    # The operators are: K2 = K^2 - params[0]*delta^2,
    # K3 = K_ij*K_jk*K_ki - params[1]*K^2*delta + params[2]*delta^3
    params_array: NDArray[np.floating]
    if not galileons:
        params_array = np.array([1.0 / 3.0, 1.0, 2.0 / 9.0])
    else:
        params_array = np.array([1.0, 3.0 / 2.0, 1.0 / 2.0])

    # Initialize c_* variables with their expected types
    c_grid: NDArray[np.floating]
    c_params_array: NDArray[np.floating]
    c_cell_size: np.floating

    # Check the precision and convert the arrays
    precision = cast(int, check_precision())
    if precision == 4:
        c_grid = grid.astype("float32")
        c_params_array = params_array.astype("float32")
        c_cell_size = np.float32(cell_size)
    else:
        c_grid = grid.astype("float64")
        c_params_array = params_array.astype("float64")
        c_cell_size = np.float64(cell_size)

    # Call the c function to compute the operators
    x = cast(
        dict[str, NDArray[np.floating]],
        operators_compute(
            c_grid,
            np.int32(order),
            np.int32(nl_order),
            c_params_array,
            np.int32(renormalized),
            c_cell_size,
            np.int32(n_threads),
            np.int32(verbose),
        ),
    )

    return x


# Smooth a given field (or fields) in a given scale
def smooth_fields(
    grid: NDArray[np.floating],
    cell_size: float = 2.0,
    k_smooth: float = 10000.0,
    input_k: bool = False,
    n_fields: int = 1,
    verbose: bool = False,
    n_threads: int = 1,
) -> NDArray[np.floating] | dict[str, NDArray[np.floating]]:
    """
    Smooth a given field (or fields) on a given scale.

    :param grid: Lagrangian density grid.
    :type grid: NDArray[np.floating]
    :param cell_size: Size of each cell in Mpc/h. Fiducial
                      value: 2.0
    :type cell_size: float
    :param k_smooth: Scale used to smooth the fields. Fiducial
                     value: 10000.0
    :type k_smooth: float
    :param input_k: Whether the density grid is in real or Fourier
                    space. Fiducial value: False
    :type input_k: bool
    :param n_fields: Number of fields. Fiducial value: 1
    :type n_fields: int
    :param verbose: Whether to output information in the C code.
                    Fiducial value: False
    :type verbose: bool
    :param n_threads: Number of threads used by OpenMP. Fiducial
                      value: 1
    :type n_threads: int

    :return: A dictionary with all fields (excluding the linear)
             up to the given order or a single smoothed field.
    :rtype: Union[NDArray[np.floating], Dict[str, NDArray[np.floating]]]
    """
    # Import the C functions
    from .lib.exshalos import (  # pyright: ignore[reportMissingImports]
        check_precision,  # pyright: ignore[reportUnknownVariableType]
        smooth_field,  # pyright: ignore[reportUnknownVariableType]
    )

    # Initialize c_* variables with their expected types
    c_grid: NDArray[np.floating]
    c_cell_size: np.floating
    c_k_smooth: np.floating

    # Check the precision and convert the arrays
    precision = cast(int, check_precision())
    if precision == 4:
        c_grid = grid.astype("float32")
        c_cell_size = np.float32(cell_size)
        c_k_smooth = np.float32(k_smooth)
    else:
        c_grid = grid.astype("float64")
        c_cell_size = np.float64(cell_size)
        c_k_smooth = np.float64(k_smooth)

    # Call the C function to smooth the fields
    if n_fields > 1:
        x: list[NDArray[np.floating]] = []
        for i in range(n_fields):
            smoothed_field = cast(
                NDArray[np.floating],
                smooth_field(
                    c_grid[i, :],
                    c_cell_size,
                    c_k_smooth,
                    np.int32(input_k),
                    np.int32(n_threads),
                    np.int32(verbose),
                ),
            )
            x.append(smoothed_field)
        return {"fields": np.array(x, dtype=c_grid.dtype)}
    else:
        x_single = cast(
            NDArray[np.floating],
            smooth_field(
                c_grid,
                c_cell_size,
                c_k_smooth,
                np.int32(input_k),
                np.int32(n_threads),
                np.int32(verbose),
            ),
        )
        return x_single


# Smooth a given field (or fields) in a given scale
def smooth_and_reduce_fields(
    grid: NDArray[np.floating],
    cell_size: float = 2.0,
    k_smooth: float = 10000.0,
    input_k: bool = False,
    n_fields: int = 1,
    verbose: bool = False,
    n_threads: int = 1,
) -> NDArray[np.floating]:
    """
    Smooth a given field (or fields) on a given scale and reduce it.

    :param grid: Lagrangian density grid.
    :type grid: NDArray[np.floating]
    :param cell_size: Size of each cell in Mpc/h. Fiducial
                      value: 2.0
    :type cell_size: float
    :param k_smooth: Scale used to smooth the fields. Fiducial
                     value: 10000.0
    :type k_smooth: float
    :param input_k: Whether the density grid is in real or Fourier
                    space. Fiducial value: False
    :type input_k: bool
    :param n_fields: Number of fields. Fiducial value: 1
    :type n_fields: int
    :param verbose: Whether to output information in the C code.
                    Fiducial value: False
    :type verbose: bool
    :param n_threads: Number of threads used by OpenMP.
                       Fiducial value: 1
    :type n_threads: int

    :return: Smoothed and reduced fields.
    :rtype: NDArray[np.floating]
    """
    # Import the C functions
    from .lib.exshalos import (  # pyright: ignore[reportMissingImports]
        check_precision,  # pyright: ignore[reportUnknownVariableType]
        smooth_and_reduce_field,  # pyright: ignore[reportUnknownVariableType]
    )

    # Initialize c_* variables with their expected types
    c_grid: NDArray[np.floating]
    c_cell_size: np.floating
    c_k_smooth: np.floating

    # Check the precision and convert the arrays
    precision = cast(int, check_precision())
    if precision == 4:
        c_grid = grid.astype("float32")
        c_cell_size = np.float32(cell_size)
        c_k_smooth = np.float32(k_smooth)
    else:
        c_grid = grid.astype("float64")
        c_cell_size = np.float64(cell_size)
        c_k_smooth = np.float64(k_smooth)

    # Call the C function to smooth and reduce the fields
    if n_fields > 1:
        x: list[NDArray[np.floating]] = []
        for i in range(n_fields):
            smoothed_field = cast(
                NDArray[np.floating],
                smooth_and_reduce_field(
                    c_grid[i, :],
                    c_cell_size,
                    c_k_smooth,
                    np.int32(input_k),
                    np.int32(n_threads),
                    np.int32(verbose),
                ),
            )
            x.append(smoothed_field)
        return np.array(x, dtype=c_grid.dtype)
    else:
        x_single = cast(
            NDArray[np.floating],
            smooth_and_reduce_field(
                c_grid,
                c_cell_size,
                c_k_smooth,
                np.int32(input_k),
                np.int32(n_threads),
                np.int32(verbose),
            ),
        )
        return x_single


# Compute the correlation function given the power spectrum
# or the power spectrum given the correlation function
def compute_correlation(
    k: NDArray[np.floating],
    p: NDArray[np.floating],
    direction: str = "pk2xi",
    verbose: bool = False,
) -> dict[str, NDArray[np.floating]]:
    """
    Compute the correlation function given the power spectrum or
    the power spectrum given the correlation function.

    :param k: Wavenumbers of the power spectrum or the distance
              of the correlation function.
    :type k: NDArray[np.floating]
    :param p: Power spectrum or the correlation function.
    :type p: NDArray[np.floating]
    :param direction: Direction to compute the fftlog
                      ("pk2xi" or "xi2pk"). Fiducial value:
                      "pk2xi"
    :type direction: str
    :param verbose: Whether to output information in the C code.
                    Fiducial value: False
    :type verbose: bool

    :return: Dictionary with keys:
             - "R" or "k": Ndarray with k or r
             - "Xi" or "Pk": Ndarray with Xi(r) or P(k)

    :rtype: Dict[str, NDArray[np.floating]]
    """
    # Import the C functions
    from .lib.analytical import (  # pyright: ignore[reportMissingImports]
        check_precision,  # pyright: ignore[reportUnknownVariableType]
        correlation_compute,  # pyright: ignore[reportUnknownVariableType]
    )

    # Initialize c_* variables with their expected types
    c_k: NDArray[np.floating]
    c_p: NDArray[np.floating]

    # Check the precision and convert the arrays
    precision = cast(int, check_precision())
    if precision == 4:
        c_k = k.astype("float32")
        c_p = p.astype("float32")
    else:
        c_k = k.astype("float64")
        c_p = p.astype("float64")

    # Call the C function to compute the correlation function
    x = cast(
        dict[str, NDArray[np.floating]],
        correlation_compute(c_k, c_p, np.int32(direction), np.int32(verbose)),
    )

    return x
