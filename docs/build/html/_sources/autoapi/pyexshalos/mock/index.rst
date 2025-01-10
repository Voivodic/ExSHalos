pyexshalos.mock
===============

.. py:module:: pyexshalos.mock

.. autoapi-nested-parse::

   This module create mocks of halos and galaxies.



Functions
---------

.. autoapisummary::

   pyexshalos.mock.Generate_Halos_Box_from_Pk
   pyexshalos.mock.Generate_Halos_Box_from_Grid
   pyexshalos.mock.Generate_Galaxies_from_Halos
   pyexshalos.mock.Split_Galaxies


Module Contents
---------------

.. py:function:: Generate_Halos_Box_from_Pk(k: numpy.ndarray, P: numpy.ndarray, R_max: float = 100000.0, nd: int = 256, ndx: int = 0, ndy: int = 0, ndz: int = 0, Lc: float = 2.0, Om0: float = 0.31, z: float = 0.0, k_smooth: float = 10000.0, delta_c: float = -1.0, Nmin: int = 1, a: float = 1.0, beta: float = 0.0, alpha: float = 0.0, seed: int = 12345, fixed: bool = False, phase: float = 0.0, OUT_DEN: bool = False, OUT_LPT: bool = False, OUT_VEL: bool = False, OUT_PROF: bool = False, DO_2LPT: bool = False, OUT_FLAG: bool = False, verbose: bool = False, nthreads: int = 1) -> Dict[str, numpy.ndarray]

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


.. py:function:: Generate_Halos_Box_from_Grid(grid: numpy.ndarray, k: numpy.ndarray, P: numpy.ndarray, S: Optional[numpy.ndarray] = None, V: Optional[numpy.ndarray] = None, Lc: float = 2.0, Om0: float = 0.31, z: float = 0.0, k_smooth: float = 10000.0, delta_c: float = -1.0, Nmin: int = 1, a: float = 1.0, beta: float = 0.0, alpha: float = 0.0, OUT_LPT: bool = False, OUT_VEL: bool = False, DO_2LPT: bool = False, OUT_FLAG: bool = False, OUT_PROF: bool = False, verbose: bool = False, nthreads: int = 1) -> Dict[str, numpy.ndarray]

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


.. py:function:: Generate_Galaxies_from_Halos(posh: numpy.ndarray, Mh: numpy.ndarray, velh: Optional[numpy.ndarray] = None, Ch: Optional[numpy.ndarray] = None, nd: int = 256, ndx: int = 0, ndy: int = 0, ndz: int = 0, Lc: float = 2.0, Om0: float = 0.31, z: float = 0.0, logMmin: float = 13.25424743, siglogM: float = 0.26461332, logM0: float = 13.28383025, logM1: float = 14.32465146, alpha: float = 1.00811277, sigma: float = 0.5, Deltah: float = -1.0, seed: int = 12345, OUT_VEL: bool = False, OUT_FLAG: bool = False, verbose: bool = False) -> Dict[str, numpy.ndarray]

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


.. py:function:: Split_Galaxies(Mh: numpy.ndarray, Flag: numpy.ndarray, params_cen: numpy.ndarray = np.array([37.10265321, -5.07596644, 0.17497771]), params_sat: numpy.ndarray = np.array([19.84341938, -2.8352781, 0.10443049]), seed: int = 12345, verbose: bool = False) -> numpy.ndarray

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


