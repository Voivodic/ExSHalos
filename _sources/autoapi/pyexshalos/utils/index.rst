pyexshalos.utils
================

.. py:module:: pyexshalos.utils


Functions
---------

.. autoapisummary::

   pyexshalos.utils.Generate_Density_Grid
   pyexshalos.utils.Find_Halos_from_Grid
   pyexshalos.utils.Displace_LPT
   pyexshalos.utils.Fit_Barrier
   pyexshalos.utils.Fit_HOD
   pyexshalos.utils.Compute_High_Order_Operators
   pyexshalos.utils.Smooth_Fields
   pyexshalos.utils.Smooth_and_Reduce_Fields
   pyexshalos.utils.Compute_Correlation


Module Contents
---------------

.. py:function:: Generate_Density_Grid(k: numpy.ndarray, P: numpy.ndarray, R_max: float = 100000.0, nd: int = 256, ndx: Optional[int] = None, ndy: Optional[int] = None, ndz: Optional[int] = None, Lc: float = 2.0, outk: bool = False, seed: int = 12345, fixed: bool = False, phase: float = 0.0, k_smooth: float = 100000.0, verbose: bool = False, nthreads: int = 1) -> numpy.ndarray

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


.. py:function:: Find_Halos_from_Grid(grid: numpy.ndarray, k: numpy.ndarray, P: numpy.ndarray, Lc: float = 2.0, Om0: float = 0.31, z: float = 0.0, delta_c: Optional[float] = None, Nmin: int = 10, a: float = 1.0, beta: float = 0.0, alpha: float = 0.0, OUT_FLAG: bool = False, verbose: bool = False) -> Dict[str, numpy.ndarray]

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


.. py:function:: Displace_LPT(grid: numpy.ndarray, Lc: float = 2.0, Om0: float = 0.31, z: float = 0.0, k_smooth: float = 10000.0, DO_2LPT: bool = False, OUT_VEL: bool = False, Input_k: bool = False, OUT_POS: bool = True, verbose: bool = False) -> Dict[str, numpy.ndarray]

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


.. py:function:: Fit_Barrier(k: numpy.ndarray, P: numpy.ndarray, M: numpy.ndarray, dndlnM: numpy.ndarray, dn_err: Optional[numpy.ndarray] = None, grid: Optional[numpy.ndarray] = None, R_max: float = 100000.0, Mmin: Optional[float] = None, Mmax: Optional[float] = None, Nm: int = 25, nd: int = 256, Lc: float = 2.0, Om0: float = 0.31, z: float = 0.0, delta_c: Optional[float] = None, Nmin: int = 10, seed: int = 12345, x0: Optional[numpy.ndarray] = None, verbose: bool = False, nthreads: int = 1, Max_inter: int = 100, tol: Optional[float] = None) -> numpy.ndarray

   Fit the parameters of the barrier given a mass function.

   :param k: Wavenumbers of the power spectrum
   :type k: numpy.ndarray
   :param P: Power spectrum
   :type P: numpy.ndarray
   :param M: Mass of the mass function to be approximated.
   :type M: numpy.ndarray
   :param dndlnM: Mass function to be approximated.
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
   :param Max_inter: Maximum number of iterations used in the minimization. Fiducial value: 100
   :type Max_inter: int
   :param tol: Tolerance for the minimization. Fiducial value: None
   :type tol: Optional[float]

   :return: Best fit of the values of the parameters of the ellipsoidal barrier.
   :rtype: numpy.ndarray


.. py:function:: Fit_HOD(k: numpy.ndarray, P: numpy.ndarray, nbar: Optional[float] = None, posh: Optional[numpy.ndarray] = None, Mh: Optional[numpy.ndarray] = None, velh: Optional[numpy.ndarray] = None, Ch: Optional[numpy.ndarray] = None, nd: int = 256, ndx: Optional[int] = None, ndy: Optional[int] = None, ndz: Optional[int] = None, Lc: float = 2.0, Om0: float = 0.31, z: float = 0.0, x0: Optional[numpy.ndarray] = None, sigma: float = 0.5, Deltah: float = -1.0, seed: int = 12345, USE_VEL: bool = False, l_max: int = 0, direction: str = 'z', window: Union[str, int] = 'cic', R: float = 4.0, R_times: float = 5.0, interlacing: bool = True, Nk: int = 25, k_min: Optional[float] = None, k_max: Optional[float] = None, verbose: bool = False, nthreads: int = 1, Max_inter: int = 100, tol: Optional[float] = None) -> numpy.ndarray

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


.. py:function:: Compute_High_Order_Operators(grid: numpy.ndarray, order: int = 2, nl_order: int = 0, Galileons: bool = False, Renormalized: bool = False, Lc: float = 2.0, verbose: bool = False, nthreads: int = 1) -> Dict[str, numpy.ndarray]

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


.. py:function:: Smooth_Fields(grid: numpy.ndarray, Lc: float = 2.0, k_smooth: float = 10000.0, Input_k: bool = False, Nfields: int = 1, verbose: bool = False, nthreads: int = 1) -> numpy.ndarray

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


.. py:function:: Smooth_and_Reduce_Fields(grid: numpy.ndarray, Lc: float = 2.0, k_smooth: float = 10000.0, Input_k: bool = False, Nfields: int = 1, verbose: bool = False, nthreads: int = 1) -> numpy.ndarray

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


.. py:function:: Compute_Correlation(k: numpy.ndarray, P: numpy.ndarray, direction: str = 'pk2xi', verbose: bool = False) -> dict

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


