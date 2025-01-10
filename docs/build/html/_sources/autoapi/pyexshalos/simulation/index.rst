pyexshalos.simulation
=====================

.. py:module:: pyexshalos.simulation

.. autoapi-nested-parse::

   This module compute quantities from data of simulations.



Functions
---------

.. autoapisummary::

   pyexshalos.simulation.Compute_Density_Grid
   pyexshalos.simulation.Compute_Power_Spectrum
   pyexshalos.simulation.Compute_Power_Spectrum_individual
   pyexshalos.simulation.Compute_BiSpectrum
   pyexshalos.simulation.Compute_TriSpectrum
   pyexshalos.simulation.Compute_Abundance
   pyexshalos.simulation.Compute_Bias_Jens


Module Contents
---------------

.. py:function:: Compute_Density_Grid(pos: numpy.ndarray, vel: Optional[numpy.ndarray] = None, mass: Optional[numpy.ndarray] = None, types: Optional[numpy.ndarray] = None, nd: int = 256, L: float = 1000.0, Om0: float = 0.31, z: float = 0.0, direction: Optional[int] = None, window: Union[str, int] = 'CIC', R: float = 4.0, R_times: float = 5.0, interlacing: bool = False, folds: int = 1, verbose: bool = False, nthreads: int = 1) -> numpy.ndarray

   Compute the density grid of a set of tracers.

   :param pos: Position of the tracers.
   :type pos: numpy.ndarray
   :param vel: Velocities of the particles in the given direction. Fiducial value: None
   :type vel: Optional[numpy.ndarray]
   :param mass: Weight of each tracer. Fiducial value: None
   :type mass: Optional[numpy.ndarray]
   :param types: Type of each tracer. Fiducial value: None
   :type types: Optional[numpy.ndarray]
   :param nd: Number of cells per dimension. Fiducial value: 256
   :type nd: int
   :param L: Size of the box in Mpc/h. Fiducial value: 1000.0
   :type L: float
   :param Om0: Value of the matter overdensity today. Fiducial value: 0.31
   :type Om0: float
   :param z: Redshift of the density grid. Fiducial value: 0.0
   :type z: float
   :param direction: Direction to use to put the particles in redshift space. Fiducial value: None
   :type direction: Optional[int]
   :param window: Density assignment method used to construct the density grid. Fiducial value: "CIC" (Options: "NGP" = 0, "CIC" = 1, "SPHERICAL" = 2, "EXPONENTIAL" = 3)
   :type window: Union[str, int]
   :param R: Smoothing length used in the spherical and exponential windows. Fiducial value: 4.0
   :type R: float
   :param R_times: Scale considered to account for particles (used in the exponential window) in units of R. Fiducial value: 5.0
   :type R_times: float
   :param interlacing: Whether to use or do not use the interlacing technique. Fiducial value: False
   :type interlacing: bool
   :param folds: Number of times that the box will be folded in each direction. Fiducial value: 1
   :type folds: int
   :param verbose: Whether to output information in the C code. Fiducial value: False
   :type verbose: bool
   :param nthreads: Number of threads used by OpenMP. Fiducial value: 1
   :type nthreads: int

   :return: The density grid for each type of tracer and for the grids with and without interlacing (if interlacing is True).
   :rtype: numpy.ndarray


.. py:function:: Compute_Power_Spectrum(grid: numpy.ndarray, L: float = 1000.0, window: Union[str, int] = 'CIC', R: float = 4.0, Nk: int = 25, k_min: Optional[float] = None, k_max: Optional[float] = None, l_max: int = 0, direction: Optional[Union[str, int]] = None, folds: int = 1, verbose: bool = False, nthreads: int = 1, ntypes: int = 1) -> Dict[str, numpy.ndarray]

   Compute the power spectrum from a density grid.

   :param grid: Density grid for all tracers.
   :type grid: numpy.ndarray
   :param L: Size of the box in Mpc/h. Fiducial value: 1000.0
   :type L: float
   :param window: Density assignment method used to construct the density grid. Fiducial value: "CIC" (Options: "NGP" = 0, "CIC" = 1, "SPHERICAL" = 2, "EXPONENTIAL" = 3)
   :type window: Union[str, int]
   :param R: Smoothing length used in the spherical and exponential windows. Fiducial value: 4.0
   :type R: float
   :param Nk: Number of bins in k to compute the power spectra. Fiducial value: 25
   :type Nk: int
   :param k_min: Minimum value of k to compute the power spectra. Fiducial value: None
   :type k_min: Optional[float]
   :param k_max: Maximum value of k to compute the power spectra. Fiducial value: None
   :type k_max: Optional[float]
   :param l_max: Maximum multipole computed. Fiducial value: 0
   :type l_max: int
   :param direction: Direction to use to put the particles in redshift space. Fiducial value: None
   :type direction: Optional[Union[str, int]]
   :param folds: Number of times that the box was folded in each direction. Fiducial value: 1
   :type folds: int
   :param verbose: Whether to output information in the C code. Fiducial value: False
   :type verbose: bool
   :param nthreads: Number of threads used by OpenMP. Fiducial value: 1
   :type nthreads: int
   :param ntypes: Number of different types of tracers. Fiducial value: 1
   :type ntypes: int

   :return: Dictionary with the following keys:
            - "k": Ndarray with the mean wavenumbers
            - "Pk": Ndarray with all possible power spectra
            - "Nk": Ndarray with the number of independent modes per bin
   :rtype: dict


.. py:function:: Compute_Power_Spectrum_individual(grid: numpy.ndarray, pos: numpy.ndarray, L: float = 1000.0, window: Union[str, int] = 'CIC', R: float = 4.0, Nk: int = 25, k_min: Optional[float] = None, k_max: Optional[float] = None, l_max: int = 0, direction: Optional[Union[str, int]] = None, folds: int = 1, verbose: bool = False, nthreads: int = 1, ntypes: int = 1) -> Dict[str, numpy.ndarray]

   Compute the power spectrum for individual tracers from a density grid.

   :param grid: Density grid for all tracers.
   :type grid: numpy.ndarray
   :param pos: Position of the tracers.
   :type pos: numpy.ndarray
   :param L: Size of the box in Mpc/h. Fiducial value: 1000.0
   :type L: float
   :param window: Density assignment method used to construct the density grid. Fiducial value: "CIC" (Options: "NGP" = 0, "CIC" = 1, "SPHERICAL" = 2, "EXPONENTIAL" = 3)
   :type window: Union[str, int]
   :param R: Smoothing length used in the spherical and exponential windows. Fiducial value: 4.0
   :type R: float
   :param Nk: Number of bins in k to compute the power spectra. Fiducial value: 25
   :type Nk: int
   :param k_min: Minimum value of k to compute the power spectra. Fiducial value: None
   :type k_min: Optional[float]
   :param k_max: Maximum value of k to compute the power spectra. Fiducial value: None
   :type k_max: Optional[float]
   :param l_max: Maximum multipole computed. Fiducial value: 0
   :type l_max: int
   :param direction: Direction to use to put the particles in redshift space. Fiducial value: None
   :type direction: Optional[Union[str, int]]
   :param folds: Number of times that the box will be folded in each direction. Fiducial value: 1
   :type folds: int
   :param verbose: Whether to output information in the C code. Fiducial value: False
   :type verbose: bool
   :param nthreads: Number of threads used by OpenMP. Fiducial value: 1
   :type nthreads: int
   :param ntypes: Number of different types of tracers. Fiducial value: 1
   :type ntypes: int

   :return: Dictionary with the following keys:
            - "k": Ndarray with the mean wavenumbers
            - "Pk": Ndarray with all possible power spectra
            - "Nk": Ndarray with the number of independent modes per bin
   :rtype: dict


.. py:function:: Compute_BiSpectrum(grid: numpy.ndarray, L: float = 1000.0, window: Union[str, int] = 'CIC', R: float = 4.0, Nk: int = 25, k_min: Optional[float] = None, k_max: Optional[float] = None, folds: int = 1, verbose: bool = False, nthreads: int = 1, ntypes: int = 1) -> Dict[str, numpy.ndarray]

   Compute the bispectrum from a density grid.

   :param grid: Density grid for all tracers.
   :type grid: numpy.ndarray
   :param L: Size of the box in Mpc/h. Fiducial value: 1000.0
   :type L: float
   :param window: Density assignment method used to construct the density grid. Fiducial value: "CIC" (Options: "NGP" = 0, "CIC" = 1, "SPHERICAL" = 2, "EXPONENTIAL" = 3)
   :type window: Union[str, int]
   :param R: Smoothing length used in the spherical and exponential windows. Fiducial value: 4.0
   :type R: float
   :param Nk: Number of bins in k to compute the tri-spectra. Fiducial value: 25
   :type Nk: int
   :param k_min: Minimum value of k to compute the tri-spectra. Fiducial value: None
   :type k_min: Optional[float]
   :param k_max: Maximum value of k to compute the tri-spectra. Fiducial value: None
   :type k_max: Optional[float]
   :param folds: Number of times that the box was folded in each direction. Fiducial value: 1
   :type folds: int
   :param verbose: Whether to output information in the C code. Fiducial value: False
   :type verbose: bool
   :param nthreads: Number of threads used by OpenMP. Fiducial value: 1
   :type nthreads: int
   :param ntypes: Number of different types of tracers. Fiducial value: 1
   :type ntypes: int

   :return: Dictionary with the following keys:
            - "kP": Ndarray with the mean wavenumbers
            - "Pk": Ndarray with all possible power spectra
            - "Nk": Ndarray with the number of independent modes per bin
            - "kB": Ndarray with the mean tripets of k for the bispectra
            - "Bk": Ndarray with all possible bispectra
            - "Ntri": Number of independent triangles in each bin
   :rtype: dict


.. py:function:: Compute_TriSpectrum(grid: numpy.ndarray, L: float = 1000.0, window: Union[str, int] = 'CIC', R: float = 4.0, Nk: int = 25, k_min: Optional[float] = None, k_max: Optional[float] = None, folds: int = 1, verbose: bool = False, nthreads: int = 1, ntypes: int = 1) -> Dict[str, numpy.ndarray]

   Compute the trispectrum from a density grid.

   :param grid: Density grid for all tracers.
   :type grid: numpy.ndarray
   :param L: Size of the box in Mpc/h. Fiducial value: 1000.0
   :type L: float
   :param window: Density assignment method used to construct the density grid. Fiducial value: "CIC" (Options: "NGP" = 0, "CIC" = 1, "SPHERICAL" = 2, "EXPONENTIAL" = 3)
   :type window: Union[str, int]
   :param R: Smoothing length used in the spherical and exponential windows. Fiducial value: 4.0
   :type R: float
   :param Nk: Number of bins in k to compute the tri-spectra. Fiducial value: 25
   :type Nk: int
   :param k_min: Minimum value of k to compute the tri-spectra. Fiducial value: None
   :type k_min: Optional[float]
   :param k_max: Maximum value of k to compute the tri-spectra. Fiducial value: None
   :type k_max: Optional[float]
   :param folds: Number of times that the box was folded in each direction. Fiducial value: 1
   :type folds: int
   :param verbose: Whether to utput information in the C code. Fiducial value: False
   :type verbose: bool
   :param nthreads: Number of threads used by OpenMP. Fiducial value: 1
   :type nthreads: int
   :param ntypes: Number of different types of tracers. Fiducial value: 1
   :type ntypes: int

   :return: Dictionary with the following keys:
            - "kP": Ndarray with the mean wavenumbers
            - "Pk": Ndarray with all possible power spectra
            - "Nk": Ndarray with the number of independent modes per bin
            - "kT": Ndarray with the mean duplets of k for the trispectra
            - "Tk": Ndarray with all possible trispectra
            - "Tuk": Ndarray with all possible unconnected trispectr
            - "Nsq": Number of independent tetrahedra in each bin
   :rtype: dict


.. py:function:: Compute_Abundance(Mh: numpy.ndarray, Mmin: Optional[float] = None, Mmax: Optional[float] = None, Nm: int = 25, Lc: float = 2.0, nd: int = 256, ndx: Optional[int] = None, ndy: Optional[int] = None, ndz: Optional[int] = None, verbose: bool = False) -> Dict[str, numpy.ndarray]

   Compute the abundance from an array of masses.

   :param Mh: Mass of each halo.
   :type Mh: numpy.ndarray
   :param Mmin: Minimum mass used to construct the mass bins. Fiducial value: None
   :type Mmin: Optional[float]
   :param Mmax: Maximum mass used to construct the mass bins. Fiducial value: None
   :type Mmax: Optional[float]
   :param Nm: Number of mass bins. Fiducial value: 25
   :type Nm: int
   :param Lc: Size of each cell in Mpc/h. Fiducial value: 2.0
   :type Lc: float
   :param nd: Number of cells in each direction. Fiducial value: 256
   :type nd: int
   :param ndx: Number of cells in the x direction. Fiducial value: None
   :type ndx: Optional[int]
   :param ndy: Number of cells in the y direction. Fiducial value: None
   :type ndy: Optional[int]
   :param ndz: Number of cells in the z direction. Fiducial value: None
   :type ndz: Optional[int]
   :param verbose: Whether to output information in the C code. Fiducial value: False
   :type verbose: bool

   :return: Dictionary with the following keys:
            - "Mh": Ndarray with the mean mass in the bin
            - "dn": Ndarray with differential abundance in the bin
            - "dn_err": Ndarray with the error in the differential abundance in the bin
   :rtype: dict


.. py:function:: Compute_Bias_Jens(grid: numpy.ndarray, Mh: numpy.ndarray, flag: numpy.ndarray, Mmin: Optional[float] = None, Mmax: Optional[float] = None, Nm: int = 20, dmin: float = -1.0, dmax: float = 1.0, Nbins: int = 200, Lambda: Optional[float] = None, Lc: float = 2.0, Central: bool = True, Normalized: bool = False) -> Dict[str, numpy.ndarray]

   Compute the bias of halos using Jens' method.

   :param grid: Linear (initial) density grid.
   :type grid: numpy.ndarray
   :param Mh: Mass of all halos.
   :type Mh: numpy.ndarray
   :param flag: Flag of all grid cells as being part of a trace or not.
   :type flag: numpy.ndarray
   :param Mmin: Minimum halo mass to be considered. Fiducial value: None
   :type Mmin: Optional[float]
   :param Mmax: Maximum halo mass to be considered. Fiducial value: None
   :type Mmax: Optional[float]
   :param Nm: Number of bins in mass. Fiducial value: 20
   :type Nm: int
   :param dmin: Minimum density to be used in the construction of the histograms in delta. Fiducial value: -1.0
   :type dmin: float
   :param dmax: Maximum density to be used in the construction of the histograms in delta. Fiducial value: 1.0
   :type dmax: float
   :param Nbins: Number of bins to be used in the construction of the histograms in delta. Fiducial value: 200
   :type Nbins: int
   :param Lambda: Scale used to smooth the density field. Fiducial value: None
   :type Lambda: Optional[float]
   :param Lc: Size of the cells of the grid. Fiducial value: 2.0
   :type Lc: float
   :param Central: Whether to use only the central particle of the halo. Fiducial value: True
   :type Central: bool
   :param Normalized: Whether to output the normalized PDF. Fiducial value: False
   :type Normalized: bool

   :return: Dictionay with the following keys:
            - "delta": Ndarray
            - "Mh": Ndarray
            - "Unmasked": Ndarray
            - "Masked": Ndarray
   :rtype: dict


