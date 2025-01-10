pyexshalos.theory
=================

.. py:module:: pyexshalos.theory

.. autoapi-nested-parse::

   This module compute some theoretical quantities.



Functions
---------

.. autoapisummary::

   pyexshalos.theory.Get_Mcell
   pyexshalos.theory.Get_Lc
   pyexshalos.theory.Get_Omz
   pyexshalos.theory.Get_deltac
   pyexshalos.theory.Get_Hz
   pyexshalos.theory.Get_Ha
   pyexshalos.theory.Get_dHa
   pyexshalos.theory.Get_fz
   pyexshalos.theory.Growth_eq
   pyexshalos.theory.Get_Dz
   pyexshalos.theory.Wth
   pyexshalos.theory.Compute_sigma
   pyexshalos.theory.dlnsdlnm
   pyexshalos.theory.fh
   pyexshalos.theory.dlnndlnm
   pyexshalos.theory.bh1
   pyexshalos.theory.bh2
   pyexshalos.theory.bh3
   pyexshalos.theory.CLPT_Powers
   pyexshalos.theory.Xi_lm
   pyexshalos.theory.Pgg_EFTofLSS


Module Contents
---------------

.. py:function:: Get_Mcell(Om0: float = 0.31, Lc: float = 2.0) -> float

   Compute the mass of a cell in the density grid.

   :param Om0: Omega matter at z=0. Fiducial value: 0.31
   :type Om0: float
   :param Lc: Size of each cell in Mpc/h. Fiducial value: 2.0
   :type Lc: float

   :return: Mass of the cell in units of solar masses per h.
   :rtype: float


.. py:function:: Get_Lc(Om0: float = 0.31, Mcell: float = 85000000000.0) -> float

   Get the size of each cell given its mass.

   :param Om0: Omega matter at z=0. Fiducial value: 0.31
   :type Om0: float
   :param Mcell: Mass of the cell in units of solar masses per h. Fiducial value: 8.5e+10
   :type Mcell: float

   :return: Size of the cell in Mpc/h.
   :rtype: float


.. py:function:: Get_Omz(z: float = 0.0, Om0: float = 0.31) -> float

   Return the value of the matter overdensity at a given redshift.

   :param z: Redshift. Fiducial value: 0.0
   :type z: float
   :param Om0: Omega matter at z=0. Fiducial value: 0.31
   :type Om0: float

   :return: Matter overdensity at redshift z.
   :rtype: float


.. py:function:: Get_deltac(z: float = 0.0, Om0: float = 0.31) -> float

   Return the value of delta_c (matter density contrast for halo formation) following a fit.

   :param z: Redshift. Fiducial value: 0.0
   :type z: float
   :param Om0: Omega matter at z=0. Fiducial value: 0.31
   :type Om0: float

   :return: Value of delta_c at redshift z.
   :rtype: float


.. py:function:: Get_Hz(z: float = 0.0, Om0: float = 0.31) -> float

   Return the Hubble function, in units of 100*h, at a given redshift.

   :param z: Redshift. Fiducial value: 0.0
   :type z: float
   :param Om0: Omega matter at z=0. Fiducial value: 0.31
   :type Om0: float

   :return: Hubble function in units of 100*h at redshift z.
   :rtype: float


.. py:function:: Get_Ha(a: float = 1.0, Om0: float = 0.31) -> float

   Return the Hubble function, in units of 100*h, at a given scale factor.

   :param a: Scale factor. Fiducial value: 1.0
   :type a: float
   :param Om0: Omega matter at z=0. Fiducial value: 0.31
   :type Om0: float

   :return: Hubble function in units of 100*h at scale factor a.
   :rtype: float


.. py:function:: Get_dHa(a: float = 1.0, Om0: float = 0.31) -> float

   Return the derivative of the Hubble's function, with respect to a in units of 100*h, at a given scale factor.

   :param a: Scale factor. Fiducial value: 1.0
   :type a: float
   :param Om0: Omega matter at z=0. Fiducial value: 0.31
   :type Om0: float

   :return: Derivative of the Hubble's function in units of 100*h at scale factor a.
   :rtype: float


.. py:function:: Get_fz(z: float = 0.0, Om0: float = 0.31) -> float

   Return the growth rate at a given redshift.

   :param z: Redshift. Fiducial value: 0.0
   :type z: float
   :param Om0: Omega matter at z=0. Fiducial value: 0.31
   :type Om0: float

   :return: Growth rate at redshift z.
   :rtype: float


.. py:function:: Growth_eq(y: Tuple[float, float], a: float, Om0: float = 0.31) -> numpy.ndarray

   Define the system of differential equations used to compute the growth function.

   :param y: Tuple containing the density contrast (d) and its derivative (v).
   :type y: tuple of float
   :param a: Scale factor. Fiducial value.
   :type a: float
   :param Om0: Omega matter at z=0. Fiducial value: 0.31
   :type Om0: float

   :return: Array containing the derivatives of density contrast and its velocity.
   :rtype: numpy.ndarray


.. py:function:: Get_Dz(Om0: float = 0.31, zmax: float = 1000, zmin: float = -0.5, nzs: int = 1000) -> Dict[str, numpy.ndarray]

   Compute the growth function over a range of redshifts.

   :param Om0: Omega matter at z=0. Fiducial value: 0.31
   :type Om0: float
   :param zmax: Maximum redshift to consider. Fiducial value: 1000
   :type zmax: float
   :param zmin: Minimum redshift to consider. Fiducial value: -0.5
   :type zmin: float
   :param nzs: Number of redshift steps. Fiducial value: 1000
   :type nzs: int

   :return: Dictionary with the keys
            - "z": Ndarray with redshifts
            = "a": Ndarray with scale factors
            - "Dz" Ndarray with growth factors
            - "dDz": Ndarray with derivatives of the growth factor
   :rtype: dict


.. py:function:: Wth(k: numpy.ndarray, R: float) -> numpy.ndarray

   Compute the top-hat window function in Fourier space.

   :param k: Wavenumber.
   :type k: numpy.ndarray
   :param R: Smoothing radius.
   :type R: float

   :return: Window function value.
   :rtype: np.ndarray


.. py:function:: Compute_sigma(k: numpy.ndarray, P: numpy.ndarray, R: Optional[numpy.ndarray] = None, M: Optional[numpy.ndarray] = None, Om0: float = 0.31, z: float = 0.0) -> numpy.ndarray

   Compute the variance of the density field.

   :param k: Wavenumbers of the power spectrum.
   :type k: numpy.ndarray
   :param P: Power spectrum.
   :type P: numpy.ndarray
   :param R: Smoothing radius (used to compute the mass). Fiducial value: None
   :type R: Optional[numpy.ndarray]
   :param M: Mass. Fiducial value: None
   :type M: Optional[numpy.ndarray]
   :param Om0: Omega matter at z=0 (used to compute the mass). Fiducial value: 0.31
   :type Om0: float
   :param z: Redshift. Fiducial value: 0.0
   :type z: float

   :return: Variance of the density field on the given scales.
   :rtype: numpy.ndarray


.. py:function:: dlnsdlnm(M: numpy.ndarray, sigma: numpy.ndarray) -> numpy.ndarray

   Compute the derivative of the logarithm of sigma with respect to the logarithm of mass using finite differences.

   :param M: Mass array.
   :type M: numpy.ndarray
   :param sigma: Variance array.
   :type sigma: numpy.ndarray

   :return: Array containing the derivatives of ln(sigma) with respect to ln(M).
   :rtype: numpy.ndarray


.. py:function:: fh(s: numpy.ndarray, model: Union[int, str] = 'PS', theta: Optional[Union[float, numpy.ndarray]] = None, delta_c: Optional[float] = None, Om0: float = 0.31, z: float = 0.0) -> numpy.ndarray

   Compute the halo mass function (HMF) based on different models.

   :param s: Variance of the density field.
   :type s: numpy.ndarray
   :param model: HMF model to use (0: Press-Schechter, 1: Sheth-Tormen, 2: Tinker, 3: Linear Diffusive Barrier).
                 Can also be a string identifier ("PS", "ST", "Tinker", "2LDB"). Fiducial value: "PS"
   :type model: Union[int, str]
   :param theta: Model parameters. For Sheth-Tormen: [a, b, p], for Tinker: Delta, for Linear Diffusive Barrier: [b, D, dv, J_max].
   :type theta: Optional[Union[float, np.ndarray]]
   :param delta_c: Critical density for collapse. Fiducial value: None
   :type delta_c: Optional[float]
   :param Om0: Omega matter at z=0. Fiducial value: 0.31
   :type Om0: float
   :param z: Redshift. Fiducial value: 0.0
   :type z: float

   :return: Array containing the multiplicity function.
   :rtype: numpy.ndarray


.. py:function:: dlnndlnm(M: numpy.ndarray, sigma: Optional[numpy.ndarray] = None, model: Union[int, str] = 'PS', theta: Optional[Union[float, numpy.ndarray]] = None, delta_c: Optional[float] = None, Om0: float = 0.31, z: float = 0.0, k: Optional[numpy.ndarray] = None, P: Optional[numpy.ndarray] = None) -> numpy.ndarray

   Compute the logarithmic derivative of the halo mass function with respect to mass.

   :param M: Mass array.
   :type M: numpy.ndarray
   :param sigma: Variance of the density field. Fiducial value: None
   :type sigma: Optional[numpy.ndarray]
   :param model: HMF model to use (0: Press-Schechter, 1: Sheth-Tormen, 2: Tinker, 3: Linear Diffusive Barrier).
                 Can also be a string identifier ("PS", "ST", "Tinker", "2LDB"). Fiducial value: "PS"
   :type model: Union[int, str]
   :param theta: Model parameters. For Sheth-Tormen: [a, b, p], for Tinker: Delta, for Linear Diffusive Barrier: [b, D, dv, J_max].
   :type theta: Optional[Union[float, np.ndarray]]
   :param delta_c: Critical density for collapse. Fiducial value: None
   :type delta_c: Optional[float]
   :param Om0: Omega matter at z=0. Fiducial value: 0.31
   :type Om0: float
   :param z: Redshift. Fiducial value: 0.0
   :type z: float
   :param k: Wavenumbers of the power spectrum (required if sigma is None). Fiducial value: None
   :type k: Optional[numpy.ndarray]
   :param P: Power spectrum (required if sigma is None). Fiducial value: None
   :type P: Optional[numpy.ndarray]

   :return: Halo mass function.
   :rtype: numpy.ndarray


.. py:function:: bh1(M: numpy.ndarray, s: Optional[numpy.ndarray] = None, model: Union[int, str] = 'PS', theta: Optional[Union[float, numpy.ndarray]] = None, delta_c: Optional[float] = None, Om0: float = 0.31, z: float = 0.0, k: Optional[numpy.ndarray] = None, P: Optional[numpy.ndarray] = None, Lagrangian: bool = False) -> numpy.ndarray

   Compute the first-order halo bias (b1).

   :param M: Mass array.
   :type M: numpy.ndarray
   :param s: Variance of the linear density field. Fiducial value: None
   :type s: Optional[numpy.ndarray]
   :param model: HMF model to use (0: Press-Schechter, 1: Sheth-Tormen, 2: Tinker, 3: Linear Diffusive Barrier).
                 Can also be a string identifier ("PS", "ST", "Tinker", "2LDB"). Fiducial value: None
   :type model: Union[int, str]
   :param theta: Model parameters. For Sheth-Tormen: [a, b, p], for Tinker: Delta, for Linear Diffusive Barrier: [b, D, dv, J_max].
   :type theta: Optional[Union[float, np.ndarray]]
   :param delta_c: Critical density for collapse. Fiducial value: None
   :type delta_c: Optional[float]
   :param Om0: Omega matter at z=0. Fiducial value: 0.31
   :type Om0: float
   :param z: Redshift. Fiducial value: 0.0
   :type z: float
   :param k: Wavenumbers of the power spectrum (required if s is None). Fiducial value: None
   :type k: Optional[numpy.ndarray]
   :param P: Power spectrum (required if s is none). Fiducial value: None
   :type P: Optional[numpy.ndarray]
   :param Lagrangian: Whether to compute the Lagrangian bias.
   :type Lagrangian: bool

   :return: First-order halo bias (b1).
   :rtype: numpy.ndarray


.. py:function:: bh2(M: numpy.ndarray, s: Optional[numpy.ndarray] = None, model: Union[int, str] = 'PS', theta: Optional[Union[float, numpy.ndarray]] = None, delta_c: Optional[float] = None, Om0: float = 0.31, z: float = 0.0, k: Optional[numpy.ndarray] = None, P: Optional[numpy.ndarray] = None, Lagrangian: bool = False, b1: Optional[numpy.ndarray] = None) -> numpy.ndarray

   Compute the second-order halo bias (b2).

   :param M: Mass array.
   :type M: numpy.ndarray
   :param s: Variance of the density field. Fiducial value: None
   :type s: Optional[numpy.ndarray]
   :param model: HMF model to use (0: Press-Schechter, 1: Sheth-Tormen, 2: Matteo, 3: Lazeyras).
                 Can also be a string identifier ("PS", "ST", "Matteo", "Lazeyras"). Fiducial value: "PS"
   :type model: Union[int, str]
   :param theta: Model parameters. For Sheth-Tormen: [a, b, p], for Matteo: b1, for Lazeyras: b1.
   :type theta: Optional[Union[float, np.ndarray]]
   :param delta_c: Critical density for collapse. Fiducial value: None
   :type delta_c: Optional[float]
   :param Om0: Omega matter at z=0. Fiducial value: 0.31
   :type Om0: float
   :param z: Redshift. Fiducial value: 0.0
   :type z: float
   :param k: Wavenumbers of the power spectrum (required if s is None). Fiducial value: None
   :type k: Optional[numpy.ndarray]
   :param P: Power spectrum (required if s is None). Fiducial value: None
   :type P: Optional[numpy.ndarray]
   :param Lagrangian: Whether to compute the Lagrangian bias.
   :type Lagrangian: bool
   :param b1: First-order halo bias (used in Matteo's and Lazeyras's models). Fiducial value: None
   :type b1: Optional[numpy.ndarray]

   :return: Array containing the second-order halo bias values (b2).
   :rtype: numpy.ndarray


.. py:function:: bh3(M: numpy.ndarray, s: Optional[numpy.ndarray] = None, model: Union[int, str] = 'PS', theta: Optional[Union[float, numpy.ndarray]] = None, delta_c: Optional[float] = None, Om0: float = 0.31, z: float = 0.0, k: Optional[numpy.ndarray] = None, P: Optional[numpy.ndarray] = None, Lagrangian: bool = False, bs2: float = 0.0) -> numpy.ndarray

   Compute the third-order halo bias (b3).

   :param M: Mass array.
   :type M: numpy.ndarray
   :param s: Variance of the density field. Fiducial value: None
   :type s: Optional[numpy.ndarray]
   :param model: HMF model to use (0: Press-Schechter, 1: Sheth-Tormen).
                 Can also be a string identifier ("PS", "ST"). Fiducial value: "PS"
   :type model: Union[int, str]
   :param theta: Model parameters. For Sheth-Tormen: [a, b, p].
   :type theta: Optional[Union[float, np.ndarray]]
   :param delta_c: Critical density for collapse. Fiducial value: None
   :type delta_c: Optional[float]
   :param Om0: Omega matter at z=0. Fiducial value: 0.31
   :type Om0: float
   :param z: Redshift. Fiducial value: 0.0
   :type z: float
   :param k: Wavenumbers of the power spectrum (required if s is None). Fiducial value: None
   :type k: Optional[numpy.ndarray]
   :param P: Power spectrum (required if s is None). Fiducial value: None
   :type P: Optional[numpy.ndarray]
   :param Lagrangian: Whether to compute the Lagrangian bias.
   :type Lagrangian: bool
   :param bs2: Second-order halo bias. Fiducial value: 0.0
   :type bs2: float

   :return: Array containing the third-order halo bias values (b3).
   :rtype: numpy.ndarray


.. py:function:: CLPT_Powers(k: numpy.ndarray, P: numpy.ndarray, Lambda: float = 0.7, kmax: float = 0.7, nmin: int = 5, nmax: int = 10, verbose: bool = False) -> Dict[str, numpy.ndarray]

   Compute the power spectra of the operators using Convolution Lagrangian Perturbation Theory (CLPT).

   :param k: Wavenumber of the power spectrum.
   :type k: numpy.ndarray
   :param P: Linear power spectrum.
   :type P: numpy.ndarray
   :param Lambda: Scale to be used to smooth the power spectrum. Fiducial value: 0.7
   :type Lambda: float
   :param kmax: Maximum wavenumber of the outputs. Fiducial value: 0.7
   :type kmax: float
   :param nmin: Minimum order used in the full computation of the terms of the expansion. Fiducial value: 5
   :type nmin: int
   :param nmax: Maximum order used in the Limber approximation of the terms of the expansion. Fiducial value: 10
   :type nmax: int
   :param verbose: Whether to output information in the C code. Fiducial value: False
   :type verbose: bool

   :return: Dictionary with the power spectra of the operators:
            - "k": Ndarray with the wavenumbers
            - "Plin": Ndarray with linear power spectrum used as input
            - "P11": Ndarray with result for the 11 power spectrum
   :rtype: dict


.. py:function:: Xi_lm(r: numpy.ndarray, k: numpy.ndarray, P: numpy.ndarray, Lambda: float = 0.7, l: int = 0, mk: int = 2, mr: int = 0, K: int = 11, alpha: float = 4.0, Rmax: float = 1.0, verbose: bool = False) -> numpy.ndarray

   Compute the generalized correlation functions (Xi_lm).

   :param r: Radial distances for the output.
   :type r: numpy.ndarray
   :param k: Wavenumber of the power spectrum.
   :type k: numpy.ndarray
   :param P: Linear power spectrum.
   :type P: numpy.ndarray
   :param Lambda: Scale to be used to smooth the power spectrum. Fiducial value: 0.7
   :type Lambda: float
   :param l: Order of the spherical Bessel's function. Fiducial value: 0
   :type l: int
   :param mk: Power of k in the integral. Fiducial value: 2
   :type mk: int
   :param mr: Power of r in the integral. Fiducial value: 0
   :type mr: int
   :param K: Number of points used by the Gaussian smooth. Fiducial value: 11
   :type K: int
   :param alpha: Value of alpha used by the Gaussian smooth. Fiducial value: 4.0
   :type alpha: float
   :param Rmax: Maximum radius for the smoothing. Fiducial value: 1.0
   :type Rmax: float
   :param verbose: Whether to output information in the C code. Fiducial value: False
   :type verbose: bool

   :return: The generalized correlation function :math: 'xi_{lm} = int dk k^{mk} r^{mr} P(k) j_l(kr)'.
   :rtype: numpy.ndarray


.. py:function:: Pgg_EFTofLSS(k: Optional[numpy.ndarray] = None, parameters: Dict[str, float] = {}, b: Optional[numpy.ndarray] = None, cs: Optional[numpy.ndarray] = None, c: Optional[numpy.ndarray] = None, IR_resummation: bool = True, cb: bool = True, RSD: bool = True, AP: bool = False, Om_fid: float = 0.31, z: float = 0.0, ls: Union[List[int], int] = [0, 2, 4], pk_mult: Optional[numpy.ndarray] = None, fz: Optional[float] = None, OUT_MULT: bool = False, h_units: bool = True, vectorized: bool = False) -> Dict[str, numpy.ndarray]

   Compute the 1-loop matter or galaxy power spectrum using classPT.

   :param k: Wavenumbers of the power spectrum (need to run CLASS-PT). Fiducial value: None
   :type k: Optional[numpy.ndarray]
   :param parameters: Cosmological parameters used by CLASS. Fiducial value: {}
   :type parameters: dict
   :param b: Values of the bias parameters (b1, b2, bG2, bGamma3, b4). Fiducial value: None
   :type b: Optional[numpy.ndarray]
   :param cs: Values of the stochastic parameters. 1D or 2D (multitracers) array. Fiducial value: None
   :type cs: Optional[numpy.ndarray]
   :param c: Values of the counterterms. 1D or 2D (multitracers) array. Fiducial value: None
   :type c: Optional[numpy.ndarray]
   :param IR_resummation: Option to do the IR resummation of the spectrum. Fiducial value: True
   :type IR_resummation: bool
   :param cb: Option to add baryons. Fiducial value: True
   :type cb: bool
   :param RSD: Option to give the power spectrum in redshift space. Fiducial value: True
   :type RSD: bool
   :param AP: Option to use the Alcock-Paczynski (AP) effect. Fiducial value: False
   :type AP: bool
   :param Om_fid: Omega matter fiducial for the AP correction. Fiducial value: 0.31
   :type Om_fid: float
   :param z: Redshift of the power spectrum. Fiducial value: 0.0
   :type z: float
   :param ls: The multipoles to be computed [0, 2, 4]. List or int.
   :type ls: Union[List[int], int]
   :param pk_mult: Multipoles of the power spectrum (don't need CLASS-PT). Fiducial value: None
   :type pk_mult: Optional[numpy.ndarray]
   :param fz: Growth rate at redshift z. Fiducial value: None
   :type fz: Optional[float]
   :param OUT_MULT: Whether output multipoles. Fiducial value: False
   :type OUT_MULT: bool
   :param h_units: Whether to use h-units. Fiducial value: True
   :type h_units: bool
   :param vectorized: Whether to use vectorized operations. Fiducial value: False
   :type vectorized: bool

   :return: Dictionary with the computed power spectra and additional information.
   :rtype: dict


