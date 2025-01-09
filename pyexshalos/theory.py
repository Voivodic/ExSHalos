"""
This module compute some theoretical quantities.
"""
from typing import Dict, Optional, Tuple, Union, List

import numpy as np


# Get the mass of each cell given its size
def Get_Mcell(Om0: float = 0.31, Lc: float = 2.0) -> float:
    """
    Compute the mass of a cell in the density grid.

    :param Om0: Omega matter at z=0. Fiducial value: 0.31
    :type Om0: float
    :param Lc: Size of each cell in Mpc/h. Fiducial value: 2.0
    :type Lc: float

    :return: Mass of the cell in units of solar masses per h.
    :rtype: float
    """
    return 2.775e+11*Om0*np.power(Lc, 3.0)

# Get the size of each cell given its mass
def Get_Lc(Om0: float = 0.31, Mcell: float = 8.5e+10) -> float:
    """
    Get the size of each cell given its mass.

    :param Om0: Omega matter at z=0. Fiducial value: 0.31
    :type Om0: float
    :param Mcell: Mass of the cell in units of solar masses per h. Fiducial value: 8.5e+10
    :type Mcell: float

    :return: Size of the cell in Mpc/h.
    :rtype: float
    """
    return np.power(Mcell / (2.775e+11 * Om0), 1.0 / 3.0)

# Get Om at any redshift
def Get_Omz(z: float = 0.0, Om0: float = 0.31) -> float:
    """
    Return the value of the matter overdensity at a given redshift.

    :param z: Redshift. Fiducial value: 0.0
    :type z: float
    :param Om0: Omega matter at z=0. Fiducial value: 0.31
    :type Om0: float

    :return: Matter overdensity at redshift z.
    :rtype: float
    """
    return Om0 * np.power(1.0 + z, 3.0) / (Om0 * np.power(1.0 + z, 3.0) + (1.0 - Om0))

# Get delta_c (critical matter overdensity for halo formation) at any redshift
def Get_deltac(z: float = 0.0, Om0: float = 0.31) -> float:
    """
    Return the value of delta_c (matter density contrast for halo formation) following a fit.

    :param z: Redshift. Fiducial value: 0.0
    :type z: float
    :param Om0: Omega matter at z=0. Fiducial value: 0.31
    :type Om0: float

    :return: Value of delta_c at redshift z.
    :rtype: float
    """
    return 1.686 * np.power(Get_Omz(z, Om0), 0.0055)

# Get H(z) (Hubble constant at z)
def Get_Hz(z: float = 0.0, Om0: float = 0.31) -> float:
    """
    Return the Hubble function, in units of 100*h, at a given redshift.

    :param z: Redshift. Fiducial value: 0.0
    :type z: float
    :param Om0: Omega matter at z=0. Fiducial value: 0.31
    :type Om0: float

    :return: Hubble function in units of 100*h at redshift z.
    :rtype: float
    """
    return np.sqrt(Om0 * np.power(1.0 + z, 3.0) + (1.0 - Om0))

# Get H(a) (Hubble constant at a)
def Get_Ha(a: float = 1.0, Om0: float = 0.31) -> float:
    """
    Return the Hubble function, in units of 100*h, at a given scale factor.

    :param a: Scale factor. Fiducial value: 1.0
    :type a: float
    :param Om0: Omega matter at z=0. Fiducial value: 0.31
    :type Om0: float

    :return: Hubble function in units of 100*h at scale factor a.
    :rtype: float
    """
    return np.sqrt(Om0 * np.power(a, -3.0) + (1.0 - Om0))

# Get dH(a) (derivative of H(a) with respect to a)
def Get_dHa(a: float = 1.0, Om0: float = 0.31) -> float:
    """
    Return the derivative of the Hubble's function, with respect to a in units of 100*h, at a given scale factor.

    :param a: Scale factor. Fiducial value: 1.0
    :type a: float
    :param Om0: Omega matter at z=0. Fiducial value: 0.31
    :type Om0: float

    :return: Derivative of the Hubble's function in units of 100*h at scale factor a.
    :rtype: float
    """
    return -1.5 * Om0 * np.power(a, -4.0) / Get_Ha(a, Om0)

# Get the growth rate at z
def Get_fz(z: float = 0.0, Om0: float = 0.31) -> float:
    """
    Return the growth rate at a given redshift.

    :param z: Redshift. Fiducial value: 0.0
    :type z: float
    :param Om0: Omega matter at z=0. Fiducial value: 0.31
    :type Om0: float

    :return: Growth rate at redshift z.
    :rtype: float
    """
    return np.power(Get_Omz(z, Om0), 0.5454)

# Define the system of differential equations used to compute the growth function
def Growth_eq(y: Tuple[float, float], a: float, Om0: float = 0.31) -> np.ndarray:
    """
    Define the system of differential equations used to compute the growth function.

    :param y: Tuple containing the density contrast (d) and its derivative (v).
    :type y: tuple of float
    :param a: Scale factor. Fiducial value.
    :type a: float
    :param Om0: Omega matter at z=0. Fiducial value: 0.31
    :type Om0: float

    :return: Array containing the derivatives of density contrast and its velocity.
    :rtype: numpy.ndarray
    """
    d, v = y
    dydt = np.array([
        v,
        -(3.0/a + Get_dHa(a, Om0)/Get_Ha(a, Om0)) * v + 3.0/2.0 * Om0 / (np.power(Get_Ha(a, Om0), 2.0) * np.power(a, 5.0)) * d
    ])
    return dydt


# Return the growth function
def Get_Dz(Om0: float = 0.31, zmax: float = 1000, zmin: float = -0.5, nzs: int = 1000) -> Dict[str, np.ndarray]:
    """
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
    """
    from scipy.integrate import odeint

    resp = {}

    # Set the initial conditions
    a = np.logspace(np.log10(1.0 / (zmax + 1.0)), np.log10(1.0 / (zmin + 1.0)), nzs)
    resp['z'] = 1.0 / a - 1.0
    resp['a'] = a
    d0 = a[0]
    dd0 = 1.0
    y0 = [d0, dd0]

    # Solve the Growth function equation
    sol = odeint(Growth_eq, y0, a, args=(Om0,))
    resp['Dz'] = sol[:, 0]
    resp['dDz'] = sol[:, 1]

    return resp

# Window (top-hat) function in Fourier space
def Wth(k: np.ndarray, R: float) -> np.ndarray:
    """
    Compute the top-hat window function in Fourier space.

    :param k: Wavenumber.
    :type k: numpy.ndarray
    :param R: Smoothing radius.
    :type R: float

    :return: Window function value.
    :rtype: np.ndarray
    """
    resp = 3.0 / (np.power(k * R, 2)) * (np.sin(k * R) / (k * R) - np.cos(k * R))
    return resp

# Compute the variance of the linear density field
def Compute_sigma(
    k: np.ndarray,
    P: np.ndarray,
    R: Optional[np.ndarray] = None,
    M: Optional[np.ndarray] = None,
    Om0: float = 0.31,
    z: float = 0.0
) -> np.ndarray:
    """
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
    """
    from scipy.integrate import simpson

    # Compute R(M) if R is not provided
    if R is None:
        if M is None:
            raise ValueError("You have to provide either the mass array (M) or the radius array (R)!")
        else:
            R = np.power(3.0 * M / (4.0 * np.pi * 2.775e+11 * Om0 * np.power(1 + z, 3.0)), 1.0 / 3.0)

    # Evaluate sigma
    Nr = len(R)
    sigma = np.zeros(Nr)
    for j in range(Nr):
        kt = k[k <= 2.0 * np.pi / R[j]]
        Pt = P[k <= 2.0 * np.pi / R[j]]
        
        sigma[j] = np.sqrt(simpson(Pt * kt * kt * np.power(Wth(kt, R[j]), 2.0), kt) / (2.0 * np.pi * np.pi))

    return sigma

# Compute the derivative of sigma with respect to M
def dlnsdlnm(M: np.ndarray, sigma: np.ndarray) -> np.ndarray:
    """
    Compute the derivative of the logarithm of sigma with respect to the logarithm of mass using finite differences.

    :param M: Mass array.
    :type M: numpy.ndarray
    :param sigma: Variance array.
    :type sigma: numpy.ndarray

    :return: Array containing the derivatives of ln(sigma) with respect to ln(M).
    :rtype: numpy.ndarray
    """
    resp = np.zeros(len(M))

    resp[0] = (np.log(sigma[1]) - np.log(sigma[0])) / (np.log(M[1]) - np.log(M[0]))
    resp[1:-1] = (np.log(sigma[2:]) - np.log(sigma[:-2])) / (np.log(M[2:]) - np.log(M[:-2]))
    resp[-1] = (np.log(sigma[-1]) - np.log(sigma[-2])) / (np.log(M[-1]) - np.log(M[-2]))

    return resp

# Multiplicity function
def fh(
    s: np.ndarray,
    model: Union[int, str] = "PS",
    theta: Optional[Union[float, np.ndarray]] = None,
    delta_c: Optional[float] = None,
    Om0: float = 0.31,
    z: float = 0.0
) -> np.ndarray:
    """
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
    """
    from scipy.special import binom

    # Compute critical density if it was not given
    if delta_c is None:
        delta_c = Get_deltac(z=z, Om0=Om0)
    nu = delta_c / s
    resp = np.zeros(len(s))

    # Press-Schechter
    if model == 0 or model in ["ps", "PS", "1SB"]:
        resp = np.sqrt(2.0 / np.pi) * nu * np.exp(-nu * nu / 2)

    # Sheth-Tormen
    elif model == 1 or model in ["ST", "st", "elliptical"]:
        if theta is not None:
            a, b, p = theta
        else:
            a, b, p = np.array([0.7, 0.4, 0.6])

        B = np.sqrt(a) * delta_c * (1.0 + b * np.power(a * nu * nu, -p))
        A = 0.0
        for i in range(6):
            A += np.power(-1, i) * binom(p, i)

        resp = np.sqrt(2.0 * a / np.pi) * nu * np.exp(-B * B / (2.0 * s * s)) * (1.0 + b * A * np.power(a * nu * nu, -p))

    # Tinker
    elif model == 2 or model in ["Tinker", "tinker", "TINKER"]:
        if theta is not None:
            Delta = theta
        else:
            Delta = 300

        if Delta == 200:
            B, d, e, f, g = 0.482, 1.97, 1.0, 0.51, 1.228
        elif Delta == 300:
            B, d, e, f, g = 0.466, 2.06, 0.99, 0.48, 1.310
        elif Delta == 400:
            B, d, e, f, g = 0.494, 2.30, 0.93, 0.48, 1.403

        resp = B * (np.power(s / e, -d) + np.power(s, -f)) * np.exp(-g / (s * s))

    # Linear Diffusive Barrier
    elif model == 3 or model in ["2LDB"]:
        if theta is not None:
            b, D, dv, J_max = theta
        else:
            b, D, dv, J_max = np.array([0.0, 0.0, 2.71, 20])

        resp = np.zeros(len(s))
        dt = delta_c + dv

        for n in range(1, J_max + 1):
            resp += 2.0 * (1.0 + D) * np.exp(-b * b * s * s / (2.0 * (1.0 + D))) * np.exp(-b * delta_c / (1.0 + D)) * (n * np.pi / (dt * dt)) * s * s * np.sin(n * np.pi * delta_c / dt) * np.exp(-n * n * np.pi * np.pi * s * s * (1.0 + D) / (2.0 * dt * dt))

    return resp

# Compute the halo mass function
def dlnndlnm(
    M: np.ndarray,
    sigma: Optional[np.ndarray] = None,
    model: Union[int, str] = "PS",
    theta: Optional[Union[float, np.ndarray]] = None,
    delta_c: Optional[float] = None,
    Om0: float = 0.31,
    z: float = 0.0,
    k: Optional[np.ndarray] = None,
    P: Optional[np.ndarray] = None
) -> np.ndarray:
    """
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
    """
    rhoc = 2.775e+11
    rhom = Om0 * rhoc * np.power(1 + z, 3)

    # Compute sigma if it was not given
    if sigma is None:
        if k is None or P is None:
            raise ValueError("sigma or (k, P) must be provided!")
        sigma = Compute_sigma(k, P, M=M, Om0=Om0, z=z)

    return -fh(sigma, model, theta, delta_c, Om0, z) * rhom / M * dlnsdlnm(M, sigma)

# Halo bias of first order
def bh1(
    M: np.ndarray,
    s: Optional[np.ndarray] = None,
    model: Union[int, str] = "PS",
    theta: Optional[Union[float, np.ndarray]] = None,
    delta_c: Optional[float] = None,
    Om0: float = 0.31,
    z: float = 0.0,
    k: Optional[np.ndarray] = None,
    P: Optional[np.ndarray] = None,
    Lagrangian: bool = False
) -> np.ndarray:
    """
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
    """
    from scipy.special import binom

    # Compute sigma if it was not given
    if s is None:
        if k is None or P is None:
            raise ValueError("s or (k, P) must be provided!")
        s = Compute_sigma(k, P, M=M, Om0=Om0, z=z)

    # Compute delta_c if it was not given
    if delta_c is None:
        delta_c = Get_deltac(Get_Omz(Om0=Om0, z=z))
    nu = delta_c / s
    resp = np.zeros(len(s))

    # Press-Schechter
    if model == 0 or model in ["ps", "PS", "1SB"]:
        resp = 1.0 + (nu * nu - 1.0) / delta_c

    # Sheth-Tormen
    elif model == 1 or model in ["ST", "st", "elliptical"]:
        if theta is not None:
            a, b, p = theta
        else:
            a, b, p = np.array([0.7, 0.4, 0.6])

        A = 0.0
        for i in range(6):
            A += np.power(-1, i) * binom(p, i)

        resp = 1.0 + np.sqrt(a) * nu * nu / delta_c * (1.0 + b * np.power(a * nu * nu, -p)) - 1.0 / (np.sqrt(a) * delta_c * (1.0 + A * np.power(a * nu * nu, -p)))

    # Tinker
    elif model == 2 or model in ["Tinker", "tinker", "TINKER"]:
        if theta is not None:
            Del = theta
        else:
            Del = 300

        y = np.log10(Del)
        A = 1.0 + 0.24 * y * np.exp(-(4.0 / y) ** 4)
        a = 0.44 * y - 0.88
        B = 0.183
        b = 1.5
        C = 0.019 + 0.107 * y + 0.19 * np.exp(-(4.0 / y) ** 4)
        c = 2.4

        resp = 1.0 - A * np.power(nu, a) / (np.power(nu, a) + np.power(delta_c, a)) + B * np.power(nu, b) + C * np.power(nu, c)

    # Linear Diffusive Barrier
    elif model == 3 or model in ["2LDB"]:
        if theta is not None:
            b, D, dv, J_max = theta
        else:
            b, D, dv, J_max = np.array([0.0, 0.0, 2.71, 20])

        resp = np.zeros(len(s))
        tmp = np.zeros(len(s))
        dt = delta_c + dv

        # Halos
        for n in range(1, J_max + 1):
            resp -= (n * np.pi / (dt * dt)) * np.sin(n * np.pi * delta_c / dt) * np.exp(-n * n * np.pi * np.pi * s * s * (1.0 + D) / (2.0 * dt * dt)) * (np.power(np.tan(n * np.pi * delta_c / dt), -1.0) * (n * np.pi / dt) - b / (1.0 + D))

        for n in range(1, J_max + 1):
            tmp += (n * np.pi / (dt * dt)) * np.sin(n * np.pi * delta_c / dt) * np.exp(-n * n * np.pi * np.pi * s * s * (1.0 + D) / (2.0 * dt * dt))

        resp = np.ones(len(s)) + resp / tmp

    # Convert to Lagrangian bias if needed
    if Lagrangian:
        resp -= 1.0

    return resp

# Halo bias of second order
def bh2(
    M: np.ndarray,
    s: Optional[np.ndarray] = None,
    model: Union[int, str] = "PS",
    theta: Optional[Union[float, np.ndarray]] = None,
    delta_c: Optional[float] = None,
    Om0: float = 0.31,
    z: float = 0.0,
    k: Optional[np.ndarray] = None,
    P: Optional[np.ndarray] = None,
    Lagrangian: bool = False,
    b1: Optional[np.ndarray] = None
) -> np.ndarray:
    """
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
    """
    from scipy.special import binom

    # Compute the variance of the linear density field if it was not given
    if s is None:
        s = Compute_sigma(k, P, M=M, Om0=Om0, z=z)

    # Compute the critical density if it was not given
    if delta_c is None:
        delta_c = Get_deltac(Get_Omz(Om0=Om0, z=z))
    nu = delta_c / s
    S = s ** 2
    resp = np.zeros(len(s))

    # Press-Schechter
    if model == 0 or model in ["ps", "PS", "1SB"]:
        resp = np.power(nu * nu / delta_c, 2.0) - 3.0 * np.power(nu / delta_c, 2.0)

    # Sheth-Tormen
    elif model == 1 or model in ["ST", "st", "elliptical"]:
        if theta is not None:
            a, b, p = theta
        else:
            a, b, p = np.array([0.7, 0.4, 0.6])

        A = 0.0
        for i in range(6):
            A += np.power(-1, i) * binom(p, i)

        B = np.sqrt(a) * delta_c * (1.0 + b * np.power(a * nu * nu, -p))
        BP = np.sqrt(a) * delta_c * (1.0 + A * np.power(a * nu * nu, -p))

        resp = np.power(B / S, 2.0) - 1.0 / S - 2.0 * B / (S * BP)

    # Matteo
    elif model == 2 or model in ["matteo", "Matteo"]:
        if b1 is None:
            b1 = bh1(M, s=s, model=2, theta=330, delta_c=delta_c, Om0=Om0, z=z, k=k, P=P, Lagrangian=True)
        resp = -0.09143 * b1 ** 3 + 0.7093 * b1 ** 2 - 0.2607 * b1 - 0.3469

    # Lazeyras
    elif model == 3 or model in ["Lazeyras", "lazeyras"]:
        if b1 is None:
            b1 = bh1(M, s=s, model=2, theta=330, delta_c=delta_c, Om0=Om0, z=z, k=k, P=P, Lagrangian=True)
        resp = 0.412 - 2.143 * b1 + 0.929 * b1 ** 2 + 0.008 * b1 ** 3

    if not Lagrangian:
        if b1 is None:
            b1 = bh1(M, s=s, model=model, theta=theta, delta_c=delta_c, Om0=Om0, z=z, k=k, P=P, Lagrangian=True)
        resp = 4.0 / 21.0 * b1 + 1.0 / 2.0 * resp

    return resp

# Halo bias of third order
def bh3(
    M: np.ndarray,
    s: Optional[np.ndarray] = None,
    model: Union[int, str] = "PS",
    theta: Optional[Union[float, np.ndarray]] = None,
    delta_c: Optional[float] = None,
    Om0: float = 0.31,
    z: float = 0.0,
    k: Optional[np.ndarray] = None,
    P: Optional[np.ndarray] = None,
    Lagrangian: bool = False,
    bs2: float = 0.0
) -> np.ndarray:
    """
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
    """
    from scipy.special import binom

    # Compute the variance of the linear density grid if it was not given
    if s is None:
        s = Compute_sigma(k, P, M=M, Om0=Om0, z=z)

    # Compute the critical density if it was not given
    if delta_c < 0.0:
        delta_c = Get_deltac(Get_Omz(Om0=Om0, z=z))
    nu = delta_c / s
    S = s ** 2
    resp = np.zeros(len(s))

    # Press-Schechter
    if model == 0 or model in ["ps", "PS", "1SB"]:
        resp = np.power(delta_c / S, 3.0) - 6.0 * delta_c / np.power(S, 2.0) + 3.0 / S / delta_c

    # Sheth-Tormen
    elif model == 1 or model in ["ST", "st", "elliptical"]:
        if theta is not None:
            a, b, p = theta
        else:
            a, b, p = np.array([0.7, 0.4, 0.6])

        A = 0.0
        for i in range(6):
            A += np.power(-1, i) * binom(p, i)

        B = np.sqrt(a) * delta_c * (1.0 + b * np.power(a * nu * nu, -p))
        BP = np.sqrt(a) * delta_c * (1.0 + A * np.power(a * nu * nu, -p))

        resp = np.power(B / S, 3.0) - 3.0 * B / np.power(S, 2.0) - 3.0 * B * B / (S * S * BP) + 3.0 / (S * BP)

    if not Lagrangian:
        b2 = bh2(M, s=s, model=model, theta=theta, delta_c=delta_c, Om0=Om0, z=z, k=k, P=P, Lagrangian=True)
        resp = -1.0 / 2.0 * b2 + 1.0 / 6.0 * resp - 2.0 / 3.0 * bs2

    return resp

# Compute the power spectra using CLPT at first order
def CLPT_Powers(
    k: np.ndarray,
    P: np.ndarray,
    Lambda: float = 0.7,
    kmax: float = 0.7,
    nmin: int = 5,
    nmax: int = 10,
    verbose: bool = False
) -> Dict[str, np.ndarray]:
    """
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
    """
    # Call the c function that compute the CLPT
    from .lib.analytical import clpt_compute

    x = clpt_compute(
        k.astype("float64"),
        P.astype("float64"),
        np.float64(Lambda),
        np.float64(kmax),
        np.int32(nmin),
        np.int32(nmax),
        np.int32(verbose)
    )
    
    return x

# Compute the generalized corraletion functions (Xi_lm)
def Xi_lm(
    r: np.ndarray,
    k: np.ndarray,
    P: np.ndarray,
    Lambda: float = 0.7,
    l: int = 0,
    mk: int = 2,
    mr: int = 0,
    K: int = 11,
    alpha: float = 4.0,
    Rmax: float = 1.0,
    verbose: bool = False
) -> np.ndarray:
    """
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
    """
    # Call the c function that compute the Xi_lm
    from .lib.analytical import xilm_compute

    x = xilm_compute(
        r.astype("float64"),
        k.astype("float64"),
        P.astype("float64"),
        np.float64(Lambda),
        np.int32(l),
        np.int32(mk),
        np.int32(mr),
        np.int32(K),
        np.float64(alpha),
        np.float64(Rmax),
        np.int32(verbose)
    )

    return x

# Compute the 1-loop matter or galaxy power spectrum using classPT
def Pgg_EFTofLSS(
    k: Optional[np.ndarray] = None,
    parameters: Dict[str, float] = {},
    b: Optional[np.ndarray] = None,
    cs: Optional[np.ndarray] = None,
    c: Optional[np.ndarray] = None,
    IR_resummation: bool = True,
    cb: bool = True,
    RSD: bool = True,
    AP: bool = False,
    Om_fid: float = 0.31,
    z: float = 0.0,
    ls: Union[List[int], int] = [0, 2, 4],
    pk_mult: Optional[np.ndarray] = None,
    fz: Optional[float] = None,
    OUT_MULT: bool = False,
    h_units: bool = True,
    vectorized: bool = False
) -> Dict[str, np.ndarray]:
    """
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
    """
    #Compute the power spectra using classPT
    if(pk_mult == None):
        try:
	        from classy import Class
        except:
            raise ImportError("classy module is not installed. Please install it using pip: pip install classy")

        #Set the parameters
        M = Class()
        params = {'A_s':2.089e-9, 'n_s':0.9649, 'tau_reio':0.052, 'omega_b':0.02237, 'omega_cdm':0.12, 'h':0.6736, 'YHe':0.2425, 'N_ur':2.0328, 'N_ncdm':1, 'm_ncdm':0.06}
        for key in parameters.keys():
            params[key] = parameters[key]
        params['z_pk'] = z
        M.set(params)
        if(cb == True):
            cb = "Yes"
        else:
            cb = "No"
        if(AP == True):
            AP = "Yes"
        else:
            AP = "No"
        if(IR_resummation == True):
            IR_resummation = "Yes"
        else:
            IR_resummation = "No"
        if(RSD == True):
            M.set({'output':'mPk', 'non linear':'PT', 'IR resummation':IR_resummation, 'Bias tracers':'Yes', 'cb':cb, 'RSD':'Yes', 'AP':AP, 'Omfid':Om_fid})
        else:
            M.set({'output':'mPk', 'non linear':'PT', 'IR resummation':IR_resummation, 'Bias tracers':'Yes', 'cb':cb, 'RSD':'No'})
        M.compute()

        #Compute the spectra of the basis
        if(k is None):
            raise TypeError("You have to give an array of k where to compute the power spectrum")
        h = M.h()
        kh = k*h
        fz = M.scale_independent_growth_factor_f(z)
        M_mult = M.get_pk_mult(kh, z, len(kh))
       
        #Save a dictionary with the spectra
        pk_mult = {}
        spectra_label = ["Id2d2", "Id2", "IG2", "Id2G2", "IG2G2", "FG2", "ctr", "lin", "1loop"]
        spectra_ind = [1, 2, 3, 4, 5, 6, 10, 14, 0]
        for i in range(len(spectra_label)):
            pk_mult[spectra_label[i]] = M_mult[spectra_ind[i]]
        if(RSD == True):
            spectra_label = ["FG2_0b1", "FG2_0", "FG2_2", "ctr_0", "ctr_2", "ctr_4"]
            spectra_ind = [7, 8 , 9, 11, 12, 13]
            for i in range(len(spectra_label)):
                pk_mult[spectra_label[i]] = M_mult[spectra_ind[i]]	
            spectra_label = ["lin_0_vv", "lin_0_vd", "lin_0_dd", "lin_2_vv", "lin_2_vd", "lin_4_vv", "1loop_0_vv", "1loop_0_vd", "1loop_0_dd", "1loop_2_vv", "1loop_2_vd", "1loop_2_dd", "1loop_4_vv", "1loop_4_vd", "1loop_4_dd", "Idd2_0", "Id2_0", "IdG2_0", "IG2_0", "Idd2_2", "Id2_2", "IdG2_2", "IG2_2", "Id2_4", "IG2_4"]
            for i in range(len(spectra_label)):
                pk_mult[spectra_label[i]] = M_mult[15+i]	

    else:
        h = 1
        if(fz == None):
            fz = pow((Om_fid*pow(1+z, 3.0))/(Om_fid*pow(1+z, 3.0) + 1.0 - Om_fid), 0.5454)
        if(RSD == True and len(pk_mult.keys()) < 10):
            raise ValueError("There are not all spectra needed for the computations in redshift space")

	#Get the number of tracers
    if(b is None):
	    raise TypeError("You have to give an array with the values of the bias parameters")
	
    if((len(b.shape) == 1 and vectorized == False) or (len(b.shape) == 2 and vectorized == True)):
        Ntracers = 1
        if(vectorized == False):
            b = b.reshape([1, 1, b.shape[0]])
            cs = cs.reshape([1, 1])
            if(RSD == True):
                c = np.array(c).reshape([1, 1, c.shape[0], c.shape[1]])
            else:
                c = np.array(c).reshape([1, 1, c.shape[0]])
        else:
            b = b.reshape([b.shape[0], 1, b.shape[1]])
            cs = cs.reshape([cs.shape[0], 1])
            if(RSD == True):
                c = np.array(c).reshape([c.shape[0], 1, c.shape[1], c.shape[2]])
            else:
                c = np.array(c).reshape([c.shape[0], 1, c.shape[1]])
    else: 
        if(vectorized == False):
            Ntracers = b.shape[0]
            b = b.reshape([1, b.shape[0], b.shape[1]])
            cs = cs.reshape([1, cs.shape[0]])
            if(RSD == True):
                c = np.array(c).reshape([1, c.shape[0], c.shape[1], c.shape[2]])
            else:
                c = np.array(c).reshape([1, c.shape[0], c.shape[1]])
        else:
            Ntracers = b.shape[1]

	#Set all combinations of the bias parameters (b1, b2, bG2, bGamm3, b4)
	#(b1, b1^2, b2^2, b1*b2, b2, b1*bG2, bG2, b2*bG2, bG2^2, b1*bGamma3, bGamma3, b4, b1*b4, b1^2*b4)
    if(RSD == True):
        bias = np.zeros([14, int(Ntracers*(Ntracers+1)/2), b.shape[0]])
        ctrs = np.zeros([3, int(Ntracers*(Ntracers+1)/2), cs.shape[0]])
        count = 0
        for i in range(Ntracers):
            for j in range(i):
                bias[:, count, :] = np.array([(b[:,i,0] + b[:,j,0])/2.0, b[:,i,0]*b[:,j,0], b[:,i,1]*b[:,j,1], (b[:,i,0]*b[:,j,1] + b[:,i,1]*b[:,j,0])/2.0, (b[:,i,1] + b[:,j,1])/2.0, (b[:,i,0]*b[:,j,2] + b[:,i,2]*b[:,j,0])/2.0, (b[:,i,2] + b[:,j,2])/2.0, (b[:,i,1]*b[:,j,2] + b[:,i,2]*b[:,j,1])/2.0, b[:,i,2]*b[:,j,2], (b[:,i,0]*b[:,j,3] + b[:,i,3]*b[:,j,0])/2.0, (b[:,i,3] + b[:,j,3])/2.0, (b[:,i,4] + b[:,j,4])/2.0, (b[:,i,0]*b[:,j,4] + b[:,i,4]*b[:,j,0])/2.0, (b[:,i,0]**2*b[:,j,4] + b[:,i,4]*b[:,j,0]**2)/2.0])
                for l in range(3):
                    ctrs[:, count, l] = (cs[:,i, l]*b[:,j,0] + cs[:,j, l]*b[:,i,0])/2.0
                count += 1
            bias[:, count, :] = np.array([b[:,i,0], b[:,i,0]**2, b[:,i,1]**2, b[:,i,0]*b[:,i,1], b[:,i,1], b[:,i,0]*b[:,i,2], b[:,i,2], b[:,i,1]*b[:,i,2], b[:,i,2]**2, b[:,i,0]*b[:,i,3], b[:,i,3], b[:,i,4], b[:,i,0]*b[:,i,4], b[:,i,0]**2*b[:,i,4]])
            for l in range(3):
                ctrs[:, count, l] = cs[:,i, l]*b[:,i,0]
            count += 1		
    #(b1^2, b1*b2, b1*bG2, b1*bGamma3, b2^2, bG2^2, b2*bG2)
    else:
        bias = np.zeros([7, int(Ntracers*(Ntracers+1)/2), b.shape[0]])
        ctrs = np.zeros([int(Ntracers*(Ntracers+1)/2), cs.shape[0]])
        count = 0
        for i in range(Ntracers):
            for j in range(i):
                bias[:,count, :] = np.array([b[:,i,0]*b[:,j,0], (b[:,i,0]*b[:,j,1] + b[:,i,1]*b[:,j,0])/2.0, (b[:,i,0]*b[:,j,2] + b[:,i,2]*b[:,j,0])/2.0, (b[:,i,0]*b[:,j,3] + b[:,i,3]*b[:,j,0])/2.0, b[:,i,1]*b[:,j,1], b[:,i,2]*b[:,j,2], (b[:,i,1]*b[:,j,2] + b[:,i,2]*b[:,j,1])/2.0])
                ctrs[count,:] = (cs[:,i]*b[:,j,0] + cs[:,j]*b[:,i,0])/2.0
                count += 1
            bias[:,count, :] = np.array([b[:,i,0]**2, b[:,i,0]*b[:,i,1], b[:,i,0]*b[:,i,2], b[:,i,0]*b[:,i,3], b[:,i,1]**2, b[:,i,2]**2, b[:,i,1]*b[:,i,2]])
            ctrs[count, :] = cs[:,i]*b[:,i,0]
            count += 1

				
    #Define the functions to compute each power spectra
    #Compute Pgg in real space
    def Pgg(ind):
        resp = (bias[0,ind,:]*(pk_mult["lin"] + pk_mult["1loop"]) + bias[1,ind,:]*pk_mult["Id2"] + 2.0*bias[2,ind,:]*pk_mult["IG2"] + 2.0*bias[2,ind,:]*pk_mult["FG2"] + 0.8*bias[3,ind,:]*pk_mult["FG2"] + 0.25*bias[4,ind,:]*pk_mult["Id2d2"] + bias[5,ind,:]*pk_mult["IG2G2"] + bias[6,ind,:]*pk_mult["Id2G2"])*pow(h, 3.0*h_units) + 2.0*ctrs[ind,:]*pk_mult["ctr"]*pow(h, h_units)
        for i in range(len(c[0,ind,:])):
            resp += c[:,ind,i]*np.power(k, 2*i)

        return resp

    #Compute the monopole of the power spectrum
    def Pgg_l0(ind):
        resp =  (pk_mult["lin_0_vv"] + pk_mult["1loop_0_vv"] + bias[0,ind,:]*(pk_mult["lin_0_vd"] + pk_mult["1loop_0_vd"]) + bias[1,ind,:]*(pk_mult["lin_0_dd"] + pk_mult["1loop_0_dd"]) + 0.25*bias[2,ind,:]*pk_mult["Id2d2"] + bias[3,ind,:]*pk_mult["Idd2_0"] + bias[4,ind,:]*pk_mult["Id2_0"] + bias[5,ind,:]*pk_mult["IdG2_0"] + bias[6,ind,:]*pk_mult["IG2_0"] + bias[7,ind,:]*pk_mult["Id2G2"] + bias[8,ind,:]*pk_mult["IG2G2"] + 2.0*bias[5,ind,:]*pk_mult["FG2_0b1"] + 2.0*bias[6,ind,:]*pk_mult["FG2_0"] + 0.8*bias[9,ind,:]*pk_mult["FG2_0b1"] + 0.8*bias[10,ind,:]*pk_mult["FG2_0"])*pow(h, 3.0*h_units) + 2.0*ctrs[0,ind,:]*pk_mult["ctr_0"]*pow(h, h_units) + fz**2*np.power(k, 2.0)*35/8.0*pk_mult["ctr_4"]*(1.0/9.0*bias[11,ind,:]*fz**2 + 2.0/7.0*fz*bias[12,ind,:] + 1.0/5.0*bias[13,ind,:])*pow(h, h_units)
        for i in range(len(c[0,ind,:,1])):
            resp += c[:,ind,i,1]*np.power(k, 2*i)

        return resp

    #Compute the quadrupole of the power spectrum
    def Pgg_l2(ind):
        resp = (pk_mult["lin_2_vv"] + pk_mult["1loop_2_vv"] + bias[0,ind,:]*(pk_mult["lin_2_vd"] + pk_mult["1loop_2_vd"]) + bias[1,ind,:]*pk_mult["1loop_2_dd"] + bias[3,ind,:]*pk_mult["Idd2_2"] + bias[4,ind,:]*pk_mult["Id2_2"] + bias[5,ind,:]*pk_mult["IdG2_2"] + bias[6,ind,:]*pk_mult["IG2_2"] + (2.0*bias[6,ind,:] + 0.8*bias[10,ind,:])*pk_mult["FG2_2"])*pow(h, 3.0*h_units) + 2.0*ctrs[1,ind,:]*pk_mult["ctr_2"]*pow(h, h_units) + fz**2*np.power(k, 2.0)*35/8.0*pk_mult["ctr_4"]*(70.0*bias[11,ind,:]*fz**2 + 165.0*fz*bias[12,ind,:] + 99.0*bias[13,ind,:])*(4.0/693.0)*pow(h, h_units)
        for i in range(len(c[0,ind,:,1])):
            resp += c[:,ind,i,1]*np.power(k, 2*i)

        return resp

    #Compute the hexadecapole of the power spectrum
    def Pgg_l4(ind):
        resp = (pk_mult["lin_4_vv"] + pk_mult["1loop_4_vv"] + bias[0,ind,:]*pk_mult["1loop_4_vd"] + bias[1,ind,:]*pk_mult["1loop_4_dd"] + bias[4,ind,:]*pk_mult["Id2_4"] + bias[6,ind,:]*pk_mult["IG2_4"])*pow(h, 3.0*h_units) + 2.0*ctrs[2,ind,:]*pk_mult["ctr_4"]*pow(h, h_units) + fz**2*np.power(k, 2.0)*35/8.0*pk_mult["ctr_4"]*(210.0*bias[11,ind,:]*fz**2 + 390.0*fz*bias[12,ind,:] + 143.0*bias[13,ind,:])*(8.0/5005.0)*pow(h, h_units)
        for i in range(len(c[0,ind,:,2])):
            resp += c[:,ind,i,2]*np.power(k, 2*i)

        return resp

    #Compute the spectra and save in the dictionary
    x = {}
    if(RSD == True):
        if(0 in ls):
            P = np.zeros([b.shape[0], int(Ntracers*(Ntracers+1)/2), len(k)])
            ind = 0
            for i in range(Ntracers):
                for j in range(i+1):
                    P[:,ind,:] = Pgg_l0(ind)
                    ind += 1
            x["Pgg_l0"] = P	
        if(2 in ls):
            P = np.zeros([b.shape[0], int(Ntracers*(Ntracers+1)/2), len(k)])
            ind = 0
            for i in range(Ntracers):
                for j in range(i+1):
                    P[:,ind,:] = Pgg_l2(ind)
                    ind += 1
            x["Pgg_l2"] = P	
        if(4 in ls):
            P = np.zeros([b.shape[0], int(Ntracers*(Ntracers+1)/2), len(k)])
            ind = 0
            for i in range(Ntracers):
                for j in range(i+1):
                    P[:,ind,:] = Pgg_l4(ind)
                    ind += 1
            x["Pgg_l4"] = P	
    else:
        P = np.zeros([b.shape[0], int(Ntracers*(Ntracers+1)/2), len(k)])
        ind = 0
        for i in range(Ntracers):
            for j in range(i+1):
                P[:,ind,:] = Pgg(ind)
                ind += 1
        x["Pgg"] = P

    #Output the spectra
    if(OUT_MULT == True):
        for key in pk_mult.keys():
            if(key == "ctr" or key == "ctr_0" or key == "ctr_2" or key == "ctr_4"):
                x[key] = pk_mult[key]*pow(h, h_units)
            else:
                x[key] = pk_mult[key]*pow(h, 3.0*h_units)

    return x
