"""
This module compute quantities from data of simulations.
"""

# Import for annotations
from typing import Optional, cast

# Import numpy
import numpy as np
from numpy.typing import NDArray


# Compute the density grid
def compute_density_grid(
    pos: NDArray[np.floating],
    vel: Optional[NDArray[np.floating]] = None,
    mass: Optional[NDArray[np.floating]] = None,
    types: Optional[NDArray[np.floating]] = None,
    nd: int = 256,
    box_size: float = 1000.0,
    omega_m0: float = 0.31,
    z: float = 0.0,
    direction: Optional[str | int] = None,
    window: str | int = "CIC",
    r: float = 4.0,
    r_times: float = 5.0,
    interlacing: bool = False,
    folds: int = 1,
    verbose: bool = False,
    n_threads: int = 1,
) -> NDArray[np.floating]:
    """
    Compute the density grid of a set of tracers.

    :param pos: Position of the tracers.
    :type pos: NDArray[np.floating]
    :param vel: Velocities of the particles in the given direction.
                Fiducial value: None
    :type vel: Optional[NDArray[np.floating]]
    :param mass: Weight of each tracer. Fiducial value: None
    :type mass: Optional[NDArray[np.floating]]
    :param types: Type of each tracer. Fiducial value: None
    :type types: Optional[NDArray[np.floating]]
    :param nd: Number of cells per dimension. Fiducial value: 256
    :type nd: int
    :param box_size: Size of the box in Mpc/h. Fiducial value: 1000.0
    :type box_size: float
    :param omega_m0: Value of the matter overdensity today.
                     Fiducial value: 0.31
    :type omega_m0: float
    :param z: Redshift of the density grid. Fiducial value: 0.0
    :type z: float
    :param direction: Direction to use to put the particles in redshift space.
                      Fiducial value: None
    :type direction: Optional[Union[str, int]]
    :param window: Density assignment method used.
                   Fiducial value: "CIC"
                   (Options: "NGP" = 0, "CIC" = 1,
                   "SPHERICAL" = 2, "EXPONENTIAL" = 3)
    :type window: Union[str, int]
    :param r: Smoothing length used in the spherical and exponential windows.
              Fiducial value: 4.0
    :type r: float
    :param r_times: Scale considered to account for particles
                    (used in the exponential window) in units of R.
                    Fiducial value: 5.0
    :type r_times: float
    :param interlacing: Whether to use or do not use the interlacing technique.
                        Fiducial value: False
    :type interlacing: bool
    :param folds: Number of times that the box will be folded in per direction.
                  Fiducial value: 1
    :type folds: int
    :param verbose: Whether to output information in the C code.
                    Fiducial value: False
    :type verbose: bool
    :param n_threads: Number of threads used by OpenMP. Fiducial value: 1
    :type n_threads: int

    :return: The density grid for each type of tracer and for the grids
             with and without interlacing (if interlacing is True).
    :rtype: NDArray[np.floating]
    """
    # Import the c functions
    from .lib.spectrum import (  # pyright: ignore[reportMissingImports]
        check_precision,  # pyright: ignore[reportUnknownVariableType]
        grid_compute,  # pyright: ignore[reportUnknownVariableType]
    )

    # Initialize c_* variables with their expected types
    c_pos: NDArray[np.floating]
    c_vel: Optional[NDArray[np.floating]] = None
    c_mass: Optional[NDArray[np.floating]] = None
    c_types: Optional[NDArray[np.floating]] = None

    c_box_size: np.floating
    c_omega_m0: np.floating
    c_z: np.floating
    c_direction: int

    # Check the precision and convert the arrays
    n_mass = 0
    precision = cast(int, check_precision())
    if precision == 4:
        c_pos = pos.astype("float32")
        if vel is not None:
            c_vel = vel.astype("float32")
        if mass is not None:
            mass = mass.astype("float32")
            n_mass = 1
        if vel is not None:
            vel = vel.astype("float32")
        c_box_size = np.float32(box_size)
        c_r = np.float32(r)
        c_r_times = np.float32(r_times)
        c_omega_m0 = np.float32(omega_m0)
        c_z = np.float32(z)
    else:
        c_pos = pos.astype("float64")
        if vel is not None:
            c_vel = vel.astype("float32")
        if mass is not None:
            c_mass = mass.astype("float64")
            n_mass = 1
        if vel is not None:
            c_vel = vel.astype("float64")
        c_box_size = np.float64(box_size)
        c_r = np.float64(r)
        c_r_times = np.float64(r_times)
        c_omega_m0 = np.float64(omega_m0)
        c_z = np.float64(z)

    # Get the number of types
    if types is not None:
        c_types = np.fabs(types)
        c_types = c_types - np.min(c_types)
        c_types = c_types.astype("int32")
        n_types = len(np.unique(c_types))
    else:
        n_types = 1

    # Get the direction for RSD
    if direction is None or direction == -1:
        c_direction = -1
    elif direction == "x" or direction == "X" or direction == 0:
        c_direction = 0
    elif direction == "y" or direction == "Y" or direction == 1:
        c_direction = 1
    elif direction == "z" or direction == "Z" or direction == 2:
        c_direction = 2
    else:
        raise ValueError("Direction must be None, x, y or z!")

    # Check the window function
    if window == "NO" or window == "no" or window == 0:
        raise ValueError(
            "You need to choose some density assigment "
            "method to construct the density grid!"
        )
    elif window == "NGP" or window == "ngp" or window == 1:
        window = 1
    elif window == "CIC" or window == "cic" or window == 2:
        window = 2
    elif window == "SPHERICAL" or window == "spherical" or window == 3:
        window = 3
    elif window == "EXPONENTIAL" or window == "exponential" or window == 4:
        window = 4

    # Call the c function to compute the density grid and return it
    grid = cast(
        NDArray[np.floating],
        grid_compute(
            c_pos,
            c_vel,
            c_mass,
            np.int32(n_mass),
            c_types,
            np.int32(n_types),
            np.int32(nd),
            c_box_size,
            c_omega_m0,
            c_z,
            np.int32(c_direction),
            np.int32(window),
            c_r,
            c_r_times,
            np.int32(interlacing),
            np.int32(folds),
            np.int32(verbose),
            np.int32(n_threads),
        ),
    )

    # Reshape the grid to the desired shape
    if not interlacing and n_types == 1:
        grid = grid.reshape([nd, nd, nd])
    elif not interlacing:
        grid = grid.reshape([n_types, nd, nd, nd])
    elif n_types == 1:
        grid = grid.reshape([2, nd, nd, nd])

    return grid


# Compute the power spectrum given the density grid
def compute_power_spectrum(
    grid: NDArray[np.floating],
    box_size: float = 1000.0,
    window: str | int = "CIC",
    r: float = 4.0,
    n_k: int = 25,
    k_min: Optional[float] = None,
    k_max: Optional[float] = None,
    l_max: int = 0,
    direction: Optional[str | int] = None,
    folds: int = 1,
    verbose: bool = False,
    n_threads: int = 1,
    n_types: int = 1,
) -> dict[str, NDArray[np.floating]]:
    """
    Compute the power spectrum from a density grid.

    :param grid: Density grid for all tracers.
    :type grid: NDArray[np.floating]
    :param box_size: Size of the box in Mpc/h. Fiducial value: 1000.0
    :type box_size: float
    :param window: Density assignment method used to construct the
                   density grid. Fiducial value: "CIC" (Options:
                   "NGP" = 0, "CIC" = 1, "SPHERICAL" = 2,
                   "EXPONENTIAL" = 3)
    :type window: Union[str, int]
    :param r: Smoothing length used in the spherical and exponential
              windows. Fiducial value: 4.0
    :type r: float
    :param n_k: Number of bins in k to compute the power spectra.
                Fiducial value: 25
    :type n_k: int
    :param k_min: Minimum value of k to compute the power spectra.
                  Fiducial value: None
    :type k_min: Optional[float]
    :param k_max: Maximum value of k to compute the power spectra.
                  Fiducial value: None
    :type k_max: Optional[float]
    :param l_max: Maximum multipole computed. Fiducial value: 0
    :type l_max: int
    :param direction: Direction to use to put the particles in redshift
                      space. Fiducial value: None
    :type direction: Optional[Union[str, int]]
    :param folds: Number of times that the box was folded in each
                  direction. Fiducial value: 1
    :type folds: int
    :param verbose: Whether to output information in the C code.
                    Fiducial value: False
    :type verbose: bool
    :param n_threads: Number of threads used by OpenMP. Fiducial value: 1
    :type n_threads: int
    :param n_types: Number of different types of tracers. Fiducial value: 1
    :type n_types: int

    :return: Dictionary with the following keys:
             - "k": Ndarray with the mean wavenumbers
             - "Pk": Ndarray with all possible power spectra
             - "Nk": Ndarray with the number of independent modes per bin

    :rtype: Dict[str, NDArray[np.floating]]
    """
    # Import the c functions
    from .lib.spectrum import (  # pyright: ignore[reportMissingImports]
        check_precision,  # pyright: ignore[reportUnknownVariableType]
        power_compute,  # pyright: ignore[reportUnknownVariableType]
    )

    # Initialize c_* variables with their expected types
    c_grid: NDArray[np.floating]
    c_box_size: np.floating
    c_r: np.floating
    c_k_min: np.floating
    c_k_max: np.floating
    c_window: int
    c_direction: int
    actual_interlacing: int
    actual_n_types: int = n_types  # might be overridden by grid shape.

    # Compute some parameters
    if len(grid.shape) == 3:
        actual_interlacing = 0
        actual_n_types = 1
    elif len(grid.shape) == 5:
        actual_interlacing = 1
        actual_n_types = grid.shape[0]
    elif len(grid.shape) == 4 and n_types == 1:
        actual_interlacing = 1
        actual_n_types = 1
    elif len(grid.shape) == 4 and n_types > 1:
        actual_interlacing = 0
        actual_n_types = n_types
    else:
        raise ValueError(
            f"Unexpected grid shape {grid.shape}. "
            "Cannot determine interlacing and number of types."
        )

    nd = grid.shape[-1]
    current_box_size: float = box_size / folds

    if k_min is None:
        k_min = 2.0 * np.pi / current_box_size
    if k_max is None:
        k_max = np.pi / current_box_size * nd

    # Check the precision and convert the arrays
    precision = cast(int, check_precision())
    if precision == 4:
        c_grid = grid.astype("float32")
        c_box_size = np.float32(current_box_size)
        c_r = np.float32(r)
        c_k_min = np.float32(k_min)
        c_k_max = np.float32(k_max)
    else:
        c_grid = grid.astype("float64")
        c_box_size = np.float64(current_box_size)
        c_r = np.float64(r)
        c_k_min = np.float64(k_min)
        c_k_max = np.float64(k_max)

    # Set the window function to be de-convolved
    if window == "NO" or window == "no" or window == "No" or window == 0:
        c_window = 0
    elif window == "NGP" or window == "ngp" or window == 1:
        c_window = 1
    elif window == "CIC" or window == "cic" or window == 2:
        c_window = 2
    elif window == "SPHERICAL" or window == "spherical" or window == 3:
        c_window = 3
    elif window == "EXPONENTIAL" or window == "exponential" or window == 4:
        c_window = 4
    else:
        raise ValueError(
            "Invalid window type. Must be one of 'NO', 'NGP', 'CIC', "
            "'SPHERICAL', 'EXPONENTIAL' or their integer equivalents."
        )

    if direction is None or direction == -1:
        c_direction = -1
    elif direction == "x" or direction == "X" or direction == 0:
        c_direction = 0
    elif direction == "y" or direction == "Y" or direction == 1:
        c_direction = 1
    elif direction == "z" or direction == "Z" or direction == 2:
        c_direction = 2
    else:
        raise ValueError("Direction must be None, x, y or z!")

    # Call the c function to compute the power spectrum
    x = cast(
        dict[str, NDArray[np.floating]],
        power_compute(
            c_grid,
            np.int32(actual_n_types),
            np.int32(nd),
            c_box_size,
            np.int32(c_window),
            c_r,
            np.int32(actual_interlacing),
            np.int32(n_k),
            c_k_min,
            c_k_max,
            np.int32(l_max),
            np.int32(c_direction),
            np.int32(verbose),
            np.int32(n_threads),
        ),
    )

    # Reshape the output to the desired shape
    if actual_n_types == 1 and l_max == 0:
        x["Pk"] = x["Pk"].reshape([n_k])
    elif actual_n_types == 1:
        x["Pk"] = x["Pk"].reshape([int(l_max / 2) + 1, n_k])
    elif l_max == 0:
        x["Pk"] = x["Pk"].reshape([
            int(actual_n_types * (actual_n_types + 1) / 2),
            n_k,
        ])
    # If actual_n_types > 1 and l_max > 0,
    # the output remains as-is from the C function.

    return x


# Compute the power spectrum given the density grid
def compute_power_spectrum_individual(
    grid: NDArray[np.floating],
    pos: NDArray[np.floating],
    box_size: float = 1000.0,
    window: str | int = "CIC",
    r: float = 4.0,
    n_k: int = 25,
    k_min: Optional[float] = None,
    k_max: Optional[float] = None,
    l_max: int = 0,
    direction: Optional[str | int] = None,
    folds: int = 1,
    verbose: bool = False,
    n_threads: int = 1,
    n_types: int = 1,
) -> dict[str, NDArray[np.floating]]:
    """
    Compute the power spectrum for individual tracers from a density grid.

    :param grid: Density grid for all tracers.
    :type grid: NDArray[np.floating]
    :param pos: Position of the tracers.
    :type pos: NDArray[np.floating]
    :param box_size: Size of the box in Mpc/h. Fiducial value: 1000.0
    :type box_size: float
    :param window: Density assignment method used to construct the
                   density grid. Fiducial value: "CIC" (Options:
                   "NGP" = 0, "CIC" = 1, "SPHERICAL" = 2,
                   "EXPONENTIAL" = 3)
    :type window: str | int
    :param r: Smoothing length used in the spherical and exponential
              windows. Fiducial value: 4.0
    :type r: float
    :param n_k: Number of bins in k to compute the power spectra.
                Fiducial value: 25
    :type n_k: int
    :param k_min: Minimum value of k to compute the power spectra.
                  Fiducial value: None
    :type k_min: Optional[float]
    :param k_max: Maximum value of k to compute the power spectra.
                  Fiducial value: None
    :type k_max: Optional[float]
    :param l_max: Maximum multipole computed. Fiducial value: 0
    :type l_max: int
    :param direction: Direction to use to put the particles in redshift
                      space. Fiducial value: None
    :type direction: Optional[str | int]
    :param folds: Number of times that the box will be folded in each
                  direction. Fiducial value: 1
    :type folds: int
    :param verbose: Whether to output information in the C code.
                    Fiducial value: False
    :type verbose: bool
    :param n_threads: Number of threads used by OpenMP. Fiducial value: 1
    :type n_threads: int
    :param n_types: Number of different types of tracers. Fiducial value: 1
    :type n_types: int

    :return: Dictionary with the following keys:
             - "k": Ndarray with the mean wavenumbers
             - "Pk": Ndarray with all possible power spectra
             - "Nk": Ndarray with the number of independent modes per bin

    :rtype: Dict[str, NDArray[np.floating]]
    """
    # Import the c functions
    from .lib.spectrum import (  # pyright: ignore[reportMissingImports]
        check_precision,  # pyright: ignore[reportUnknownVariableType]
        power_compute_individual,  # pyright: ignore[reportUnknownVariableType]
    )

    # Initialize c_* variables with their expected types
    c_grid: NDArray[np.floating]
    c_pos: NDArray[np.floating]
    c_box_size: np.floating
    c_r: np.floating
    c_k_min: np.floating
    c_k_max: np.floating
    c_window: int
    c_direction: int
    actual_interlacing: int = 0
    actual_n_types: int = n_types

    # Compute some parameters
    if len(grid.shape) == 3:
        actual_interlacing = 0
        actual_n_types = 1
    elif len(grid.shape) == 5:
        actual_interlacing = 1
        actual_n_types = grid.shape[0]
    elif len(grid.shape) == 4 and n_types == 1:
        actual_interlacing = 1
        actual_n_types = 1
    elif len(grid.shape) == 4 and n_types > 1:
        actual_interlacing = 0
        actual_n_types = n_types
    else:
        raise ValueError(
            f"Unexpected grid shape {grid.shape}. "
            "Cannot determine interlacing and number of types."
        )

    nd = grid.shape[-1]
    n_p = pos.shape[0]
    current_box_size: float = box_size / folds

    # Ensure k_min and k_max are float and not None
    if k_min is None:
        k_min = 2.0 * np.pi / current_box_size

    if k_max is None:
        k_max = np.pi / current_box_size * nd

    # Check the precision and convert the arrays
    precision = cast(int, check_precision())
    if precision == 4:
        c_grid = grid.astype("float32")
        c_pos = pos.astype("float32")
        c_box_size = np.float32(current_box_size)
        c_r = np.float32(r)
        c_k_min = np.float32(k_min)
        c_k_max = np.float32(k_max)
    else:
        c_grid = grid.astype("float64")
        c_pos = pos.astype("float64")
        c_box_size = np.float64(current_box_size)
        c_r = np.float64(r)
        c_k_min = np.float64(k_min)
        c_k_max = np.float64(k_max)

    # Set the window function to be de-convolved
    if window == "NO" or window == "no" or window == "No" or window == 0:
        c_window = 0
    elif window == "NGP" or window == "ngp" or window == 1:
        c_window = 1
    elif window == "CIC" or window == "cic" or window == 2:
        c_window = 2
    elif window == "SPHERICAL" or window == "spherical" or window == 3:
        c_window = 3
    elif window == "EXPONENTIAL" or window == "exponential" or window == 4:
        c_window = 4
    else:
        raise ValueError(
            "Invalid window type. Must be one of 'NO', 'NGP', 'CIC', "
            "'SPHERICAL', 'EXPONENTIAL' or their integer equivalents."
        )

    # Get the direction for RSD
    if direction is None or direction == -1:
        c_direction = -1
    elif direction == "x" or direction == "X" or direction == 0:
        c_direction = 0
    elif direction == "y" or direction == "Y" or direction == 1:
        c_direction = 1
    elif direction == "z" or direction == "Z" or direction == 2:
        c_direction = 2
    else:
        raise ValueError("Direction must be None, x, y or z!")

    # Call the c function to compute the power spectrum
    x = cast(
        dict[str, NDArray[np.floating]],
        power_compute_individual(
            c_grid,
            c_pos,
            np.int32(actual_n_types),
            np.int32(nd),
            c_box_size,
            np.int32(c_window),
            c_r,
            np.int32(actual_interlacing),
            np.int32(n_k),
            c_k_min,
            c_k_max,
            np.int32(l_max),
            np.int32(c_direction),
            np.int32(verbose),
            np.int32(n_threads),
        ),
    )

    # Reshape the power spectrum
    if actual_n_types == 1 and l_max == 0:
        x["Pk"] = x["Pk"].reshape([n_p, n_k])
    elif actual_n_types == 1:
        x["Pk"] = x["Pk"].reshape([n_p, int(l_max / 2) + 1, n_k])
    elif l_max == 0:
        x["Pk"] = x["Pk"].reshape([n_p, actual_n_types, n_k])

    return x


# Compute the bispectrum given the density grid
def compute_bispectrum(
    grid: NDArray[np.floating],
    box_size: float = 1000.0,
    window: str | int = "CIC",
    r: float = 4.0,
    n_k: int = 25,
    k_min: Optional[float] = None,
    k_max: Optional[float] = None,
    folds: int = 1,
    verbose: bool = False,
    n_threads: int = 1,
    n_types: int = 1,
) -> dict[str, NDArray[np.floating]]:
    """
    Compute the bispectrum from a density grid.

    :param grid: Density grid for all tracers.
    :type grid: NDArray[np.floating]
    :param box_size: Size of the box in Mpc/h. Fiducial value: 1000.0
    :type box_size: float
    :param window: Density assignment method used to construct the
                   density grid. Fiducial value: "CIC" (Options:
                   "NGP" = 0, "CIC" = 1, "SPHERICAL" = 2,
                   "EXPONENTIAL" = 3)
    :type window: Union[str, int]
    :param r: Smoothing length used in the spherical and exponential
              windows. Fiducial value: 4.0
    :type r: float
    :param n_k: Number of bins in k to compute the tri-spectra.
                Fiducial value: 25
    :type n_k: int
    :param k_min: Minimum value of k to compute the tri-spectra.
                  Fiducial value: None
    :type k_min: Optional[float]
    :param k_max: Maximum value of k to compute the tri-spectra.
                  Fiducial value: None
    :type k_max: Optional[float]
    :param folds: Number of times that the box was folded in each
                  direction. Fiducial value: 1
    :type folds: int
    :param verbose: Whether to output information in the C code.
                    Fiducial value: False
    :type verbose: bool
    :param n_threads: Number of threads used by OpenMP. Fiducial value: 1
    :type n_threads: int
    :param n_types: Number of different types of tracers. Fiducial value: 1
    :type n_types: int

    :return: Dictionary with the following keys:
             - "kP": Ndarray with the mean wavenumbers
             - "Pk": Ndarray with all possible power spectra
             - "Nk": Ndarray with the number of independent modes
               per bin
             - "kB": Ndarray with the mean tripets of k for the
               bispectra
             - "Bk": Ndarray with all possible bispectra
             - "Ntri": Number of independent triangles in each bin

    :rtype: Dict[str, NDArray[np.floating]]
    """
    # Import the c functions
    from .lib.spectrum import (  # pyright: ignore[reportMissingImports]
        bi_compute,  # pyright: ignore[reportUnknownVariableType]
        check_precision,  # pyright: ignore[reportUnknownVariableType]
    )

    # Initialize c_* variables with their expected types
    c_grid: NDArray[np.floating]
    c_box_size: np.floating
    c_r: np.floating
    c_k_min: np.floating
    c_k_max: np.floating
    c_window: int
    actual_interlacing: int = 0
    actual_n_types: int = n_types

    # Compute some parameters
    if len(grid.shape) == 3:
        actual_interlacing = 0
        actual_n_types = 1
    elif len(grid.shape) == 5:
        actual_interlacing = 1
        actual_n_types = grid.shape[0]
    elif len(grid.shape) == 4 and n_types == 1:
        actual_interlacing = 1
        actual_n_types = 1
    elif len(grid.shape) == 4 and n_types > 1:
        actual_interlacing = 0
        actual_n_types = n_types
    else:
        raise ValueError(
            f"Unexpected grid shape {grid.shape}. "
            "Cannot determine interlacing and number of types."
        )

    nd = grid.shape[-1]
    current_box_size: float = box_size / folds

    # Ensure k_min and k_max are float and not None
    if k_min is None:
        k_min = 2.0 * np.pi / current_box_size

    if k_max is None:
        k_max = np.pi / current_box_size * nd

    # Check the precision and convert the arrays
    precision = cast(int, check_precision())
    if precision == 4:
        c_grid = grid.astype("float32")
        c_box_size = np.float32(current_box_size)
        c_r = np.float32(r)
        c_k_min = np.float32(k_min)
        c_k_max = np.float32(k_max)
    else:
        c_grid = grid.astype("float64")
        c_box_size = np.float64(current_box_size)
        c_r = np.float64(r)
        c_k_min = np.float64(k_min)
        c_k_max = np.float64(k_max)

    # Set the window function to be de-convolved
    if window == "NO" or window == "no" or window == "No" or window == 0:
        c_window = 0
    elif window == "NGP" or window == "ngp" or window == 1:
        c_window = 1
    elif window == "CIC" or window == "cic" or window == 2:
        c_window = 2
    elif window == "SPHERICAL" or window == "spherical" or window == 3:
        c_window = 3
    elif window == "EXPONENTIAL" or window == "exponential" or window == 4:
        c_window = 4
    else:
        raise ValueError(
            "Invalid window type. Must be one of 'NO', 'NGP', 'CIC', "
            "'SPHERICAL', 'EXPONENTIAL' or their integer equivalents."
        )

    # Call the c function to compute the bispectrum
    x = cast(
        dict[str, NDArray[np.floating]],
        bi_compute(
            c_grid,
            np.int32(actual_n_types),
            np.int32(nd),
            c_box_size,
            np.int32(c_window),
            c_r,
            np.int32(actual_interlacing),
            np.int32(n_k),
            c_k_min,
            c_k_max,
            np.int32(verbose),
            np.int32(n_threads),
        ),
    )

    # Reshape the bispectrum
    if actual_n_types == 1:
        x["Pk"] = x["Pk"].reshape([n_k])
        x["Bk"] = x["Bk"].reshape([len(x["Ntri"])])

    return x


# Compute the trispectrum given the density grid
def compute_trispectrum(
    grid: NDArray[np.floating],
    box_size: float = 1000.0,
    window: str | int = "CIC",
    r: float = 4.0,
    n_k: int = 25,
    k_min: Optional[float] = None,
    k_max: Optional[float] = None,
    folds: int = 1,
    verbose: bool = False,
    n_threads: int = 1,
    n_types: int = 1,
) -> dict[str, NDArray[np.floating]]:
    """
    Compute the trispectrum from a density grid.

    :param grid: Density grid for all tracers.
    :type grid: NDArray[np.floating]
    :param box_size: Size of the box in Mpc/h. Fiducial value: 1000.0
    :type box_size: float
    :param window: Density assignment method used to construct the
                   density grid. Fiducial value: "CIC" (Options:
                   "NGP" = 0, "CIC" = 1, "SPHERICAL" = 2,
                   "EXPONENTIAL" = 3)
    :type window: str | int
    :param r: Smoothing length used in the spherical and exponential
              windows. Fiducial value: 4.0
    :type r: float
    :param n_k: Number of bins in k to compute the tri-spectra.
                Fiducial value: 25
    :type n_k: int
    :param k_min: Minimum value of k to compute the tri-spectra.
                  Fiducial value: None
    :type k_min: Optional[float]
    :param k_max: Maximum value of k to compute the tri-spectra.
                  Fiducial value: None
    :type k_max: Optional[float]
    :param folds: Number of times that the box was folded in each
                  direction. Fiducial value: 1
    :type folds: int
    :param verbose: Whether to output information in the C code.
                    Fiducial value: False
    :type verbose: bool
    :param n_threads: Number of threads used by OpenMP. Fiducial value: 1
    :type n_threads: int
    :param n_types: Number of different types of tracers. Fiducial value: 1
    :type n_types: int

    :return: Dictionary with the following keys:
             - "kP": Ndarray with the mean wavenumbers
             - "Pk": Ndarray with all possible power spectra
             - "Nk": Ndarray with the number of independent modes
               per bin
             - "kT": Ndarray with the mean duplets of k for the
               trispectra
             - "Tk": Ndarray with all possible trispectra
             - "Tuk": Ndarray with all possible unconnected
               trispectr
             - "Nsq": Number of independent tetrahedra in each bin

    :rtype: Dict[str, NDArray[np.floating]]
    """
    # Import the c functions
    from .lib.spectrum import (  # pyright: ignore[reportMissingImports]
        check_precision,  # pyright: ignore[reportUnknownVariableType]
        tri_compute,  # pyright: ignore[reportUnknownVariableType]
    )

    # Initialize c_* variables with their expected types
    c_grid: NDArray[np.floating]
    c_box_size: np.floating
    c_r: np.floating
    c_k_min: np.floating
    c_k_max: np.floating
    c_window: int
    actual_interlacing: int
    actual_n_types: int = n_types

    # Compute some parameters
    if len(grid.shape) == 3:
        actual_interlacing = 0
        actual_n_types = 1
    elif len(grid.shape) == 5:
        actual_interlacing = 1
        actual_n_types = grid.shape[0]
    elif len(grid.shape) == 4 and n_types == 1:
        actual_interlacing = 1
        actual_n_types = 1
    elif len(grid.shape) == 4 and n_types > 1:
        actual_interlacing = 0
        actual_n_types = n_types
    else:
        raise ValueError(
            f"Unexpected grid shape {grid.shape}. "
            "Cannot determine interlacing and number of types."
        )

    nd = grid.shape[-1]
    current_box_size: float = box_size / folds

    # Ensure k_min and k_max are float and not None
    if k_min is None:
        k_min = 2.0 * np.pi / current_box_size

    if k_max is None:
        k_max = np.pi / current_box_size * nd

    # Check the precision and convert the arrays
    precision = cast(int, check_precision())
    if precision == 4:
        c_grid = grid.astype("float32")
        c_box_size = np.float32(current_box_size)
        c_r = np.float32(r)
        c_k_min = np.float32(k_min)
        c_k_max = np.float32(k_max)
    else:
        c_grid = grid.astype("float64")
        c_box_size = np.float64(current_box_size)
        c_r = np.float64(r)
        c_k_min = np.float64(k_min)
        c_k_max = np.float64(k_max)

    # Set the window function to be de-convolved
    if window == "NO" or window == "no" or window == "No" or window == 0:
        c_window = 0
    elif window == "NGP" or window == "ngp" or window == 1:
        c_window = 1
    elif window == "CIC" or window == "cic" or window == 2:
        c_window = 2
    elif window == "SPHERICAL" or window == "spherical" or window == 3:
        c_window = 3
    elif window == "EXPONENTIAL" or window == "exponential" or window == 4:
        c_window = 4
    else:
        raise ValueError(
            "Invalid window type. Must be one of 'NO', 'NGP', 'CIC', "
            "'SPHERICAL', 'EXPONENTIAL' or their integer equivalents."
        )

    # Call the c function to compute the trispectrum
    x = cast(
        dict[str, NDArray[np.floating]],
        tri_compute(
            c_grid,
            np.int32(actual_n_types),
            np.int32(nd),
            c_box_size,
            np.int32(c_window),
            c_r,
            np.int32(actual_interlacing),
            np.int32(n_k),
            c_k_min,
            c_k_max,
            np.int32(verbose),
            np.int32(n_threads),
        ),
    )

    # Reshape the trispectrum
    if actual_n_types == 1:
        x["Pk"] = x["Pk"].reshape([n_k])
        x["Tk"] = x["Tk"].reshape([len(x["Nsq"])])
        x["Tuk"] = x["Tuk"].reshape([len(x["Nsq"])])

    return x


# Measure the abundance of a list of halo masses
def compute_abundance(
    halo_mass: NDArray[np.floating],
    min_mass: Optional[float] = None,
    max_mass: Optional[float] = None,
    num_mass_bins: int = 25,
    cell_size: float = 2.0,
    nd: int = 256,
    nd_x: Optional[int] = None,
    nd_y: Optional[int] = None,
    nd_z: Optional[int] = None,
    verbose: bool = False,
) -> dict[str, NDArray[np.floating]]:
    """
    Compute the abundance from an array of masses.

    :param halo_mass: Mass of each halo.
    :type halo_mass: NDArray[np.floating]
    :param min_mass: Minimum mass used to construct the mass bins.
                     Fiducial value: None
    :type min_mass: Optional[float]
    :param max_mass: Maximum mass used to construct the mass bins.
                     Fiducial value: None
    :type max_mass: Optional[float]
    :param num_mass_bins: Number of mass bins. Fiducial value: 25
    :type num_mass_bins: int
    :param cell_size: Size of each cell in Mpc/h. Fiducial value: 2.0
    :type cell_size: float
    :param nd: Number of cells in each direction. Fiducial value: 256
    :type nd: int
    :param nd_x: Number of cells in the x direction. Fiducial value: None
    :type nd_x: Optional[int]
    :param nd_y: Number of cells in the y direction. Fiducial value: None
    :type nd_y: Optional[int]
    :param nd_z: Number of cells in the z direction. Fiducial value: None
    :type nd_z: Optional[int]
    :param verbose: Whether to output information in the C code.
                    Fiducial value: False
    :type verbose: bool

    :return: Dictionary with the following keys:
             - "Mh": Ndarray with the mean mass in the bin
             - "dn": Ndarray with differential abundance in
               the bin
             - "dn_err": Ndarray with the error in the
               differential abundance in the bin

    :rtype: Dict[str, NDArray[np.floating]]
    """
    # Import the c functions
    from .lib.spectrum import (  # pyright: ignore[reportMissingImports]
        abundance_compute,  # pyright: ignore[reportUnknownVariableType]
        check_precision,  # pyright: ignore[reportUnknownVariableType]
    )

    # Initialize c_* variables with their expected types
    c_halo_mass: NDArray[np.floating]
    c_min_mass: np.floating
    c_max_mass: np.floating
    c_cell_size: np.floating
    c_nd_x: int
    c_nd_y: int
    c_nd_z: int

    # Set the maximum and minimum masses to -1 if no value was given
    min_mass_val: float = -1.0 if min_mass is None else min_mass
    max_mass_val: float = -1.0 if max_mass is None else max_mass

    # Check the precision and convert the arrays
    precision = cast(int, check_precision())
    if precision == 4:
        c_halo_mass = halo_mass.astype("float32")
        c_min_mass = np.float32(min_mass_val)
        c_max_mass = np.float32(max_mass_val)
        c_cell_size = np.float32(cell_size)
    else:
        c_halo_mass = halo_mass.astype("float64")
        c_min_mass = np.float64(min_mass_val)
        c_max_mass = np.float64(max_mass_val)
        c_cell_size = np.float64(cell_size)

    # Set the number of cells in each dimension
    c_nd_x = nd if nd_x is None else nd_x
    c_nd_y = nd if nd_y is None else nd_y
    c_nd_z = nd if nd_z is None else nd_z

    # Call the c function to compute the abundance
    x = cast(
        dict[str, NDArray[np.floating]],
        abundance_compute(
            c_halo_mass,
            c_min_mass,
            c_max_mass,
            np.int32(num_mass_bins),
            c_cell_size,
            np.int32(c_nd_x),
            np.int32(c_nd_y),
            np.int32(c_nd_z),
            np.int32(verbose),
        ),
    )

    return x


# Measure the b_n bias parameters using Jens' technic
def compute_bias_jens(
    grid: NDArray[np.floating],
    halo_mass: NDArray[np.floating],
    flag_grid: NDArray[np.floating],
    min_mass: Optional[float] = None,
    max_mass: Optional[float] = None,
    num_mass_bins: int = 20,
    min_density: float = -1.0,
    max_density: float = 1.0,
    num_density_bins: int = 200,
    smoothing_scale: Optional[float] = None,
    cell_size: float = 2.0,
    use_central: bool = True,
    normalize_pdf: bool = False,
) -> dict[str, NDArray[np.floating]]:
    """
    Compute the bias of halos using Jens' method.

    :param grid: Linear (initial) density grid.
    :type grid: NDArray[np.floating]
    :param halo_mass: Mass of all halos.
    :type halo_mass: NDArray[np.floating]
    :param flag_grid: Flag of all grid cells as being part of a
                      tracer or not.
    :type flag_grid: NDArray[np.floating]
    :param min_mass: Minimum halo mass to be considered. Fiducial
                     value: None
    :type min_mass: Optional[float]
    :param max_mass: Maximum halo mass to be considered. Fiducial
                     value: None
    :type max_mass: Optional[float]
    :param num_mass_bins: Number of bins in mass. Fiducial value: 20
    :type num_mass_bins: int
    :param min_density: Minimum density to be used in the
                        construction of the histograms in delta.
                        Fiducial value: -1.0
    :type min_density: float
    :param max_density: Maximum density to be used in the
                        construction of the histograms in delta.
                        Fiducial value: 1.0
    :type max_density: float
    :param num_density_bins: Number of bins to be used in the
                             construction of the histograms in
                             delta. Fiducial value: 200
    :type num_density_bins: int
    :param smoothing_scale: Scale used to smooth the density field.
                            Fiducial value: None
    :type smoothing_scale: Optional[float]
    :param cell_size: Size of the cells of the grid. Fiducial
                      value: 2.0
    :type cell_size: float
    :param use_central: Whether to use only the central particle of
                        the halo. Fiducial value: True
    :type use_central: bool
    :param normalize_pdf: Whether to output the normalized PDF.
                          Fiducial value: False
    :type normalize_pdf: bool

    :return: Dictionary with the following keys:
                - "delta": Ndarray
                - "Mh": Ndarray
                - "Unmasked": Ndarray
                - "Masked": Ndarray

    :rtype: Dict[str, NDArray[np.floating]]
    """
    # Import the c functions
    from .lib.spectrum import (  # pyright: ignore[reportMissingImports]
        check_precision,  # pyright: ignore[reportUnknownVariableType]
        histogram_compute,  # pyright: ignore[reportUnknownVariableType]
    )

    # Import from utils
    from .utils import (
        smooth_fields,  # pyright: ignore[reportMissingImports, reportUnknownVariableType]
    )

    # Initialize c_* variables with their expected types
    c_grid: NDArray[np.floating]
    c_halo_mass: NDArray[np.floating]
    c_flag_grid: NDArray[np.integer]
    c_min_mass: np.floating
    c_max_mass: np.floating
    c_min_density: np.floating
    c_max_density: np.floating

    # Smooth the density field
    if smoothing_scale is not None:
        smoothed_grid: NDArray[np.floating] = cast(
            NDArray[np.floating],
            smooth_fields(
                grid=np.copy(grid),
                cell_size=cell_size,
                k_smooth=smoothing_scale,
                input_k=False,
                n_fields=1,
                verbose=False,
                n_threads=1,
            ),
        )
    else:
        smoothed_grid: NDArray[np.floating] = np.copy(grid)

    # Set the minimum and maximum masses to -1.0 if not provided
    min_mass_val: float = -1.0 if min_mass is None else min_mass
    max_mass_val: float = -1.0 if max_mass is None else max_mass

    # Check the precision and convert the arrays
    precision = cast(int, check_precision())
    if precision == 4:
        c_grid = smoothed_grid.astype("float32")
        c_halo_mass = halo_mass.astype("float32")
        c_flag_grid = flag_grid.astype("int64")
        c_min_mass = np.float32(min_mass_val)
        c_max_mass = np.float32(max_mass_val)
        c_min_density = np.float32(min_density)
        c_max_density = np.float32(max_density)
    else:
        c_grid = smoothed_grid.astype("float64")
        c_halo_mass = halo_mass.astype("float64")
        c_flag_grid = flag_grid.astype("int64")
        c_min_mass = np.float64(min_mass_val)
        c_max_mass = np.float64(max_mass_val)
        c_min_density = np.float64(min_density)
        c_max_density = np.float64(max_density)

    # Call the c function that computes the histogram
    x = cast(
        dict[str, NDArray[np.floating]],
        histogram_compute(
            c_grid,
            c_halo_mass,
            c_flag_grid,
            c_min_mass,
            c_max_mass,
            np.int32(num_mass_bins),
            c_min_density,
            c_max_density,
            np.int32(num_density_bins),
            np.int32(use_central),
        ),
    )
    x["sigma2"] = cast(NDArray[np.floating], np.mean(c_grid**2))
    del smoothed_grid  # Clean up the temporary smoothed grid

    # Normalize the histograms
    if normalize_pdf:
        x["Unmasked"] = x["Unmasked"] / np.sum(x["Unmasked"])
        x["Masked"] = x["Masked"] / np.sum(x["Masked"], axis=1).reshape([
            num_mass_bins,
            1,
        ])

    return x
