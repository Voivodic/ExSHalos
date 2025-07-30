"""
Test the theory module functions using pytest.
"""

# Import the core modules
import numpy as np
import pytest
from numpy.typing import NDArray

# Import the module with the theory functions
from pyexshalos import theory


# --- Fixtures for test setup ---


@pytest.fixture(scope="module")
def test_params() -> dict[str, float]:
    """Defines common parameters for all tests."""
    return {
        "omega_m0": 0.31,
        "z": 0.0,
        "cell_size": 2.0,
        "m_cell": 8.5e10,
    }


@pytest.fixture(scope="module")
def sample_arrays() -> tuple[NDArray[np.floating], NDArray[np.floating]]:
    """Creates sample arrays for testing."""
    k = np.logspace(-3, 1, 100)
    pk = np.power(k, -1.5)
    return k, pk


# --- Test Functions ---


def test_get_cell_mass(test_params: dict[str, float]) -> None:
    """Test the get_cell_mass function."""
    mass = theory.get_cell_mass(
        omega_m0=test_params["omega_m0"], cell_size=test_params["cell_size"]
    )
    expected = (
        2.775e11 * test_params["omega_m0"] * np.power(test_params["cell_size"], 3.0)
    )
    assert np.isclose(mass, expected)


def test_get_cell_size(test_params: dict[str, float]) -> None:
    """Test the get_cell_size function."""
    size = theory.get_cell_size(
        omega_m0=test_params["omega_m0"], m_cell=test_params["m_cell"]
    )
    expected = np.power(
        test_params["m_cell"] / (2.775e11 * test_params["omega_m0"]), 1.0 / 3.0
    )
    assert np.isclose(size, expected)


def test_get_omz(test_params: dict[str, float]) -> None:
    """Test the get_omz function."""
    z_array = np.array([0.0, 0.5, 1.0])
    omz = theory.get_omz(z_array, omega_m0=test_params["omega_m0"])

    # Test at z=0
    expected_z0 = test_params["omega_m0"]
    assert np.isclose(omz[0], expected_z0)

    # Test at z=1
    expected_z1 = (
        test_params["omega_m0"]
        * 8.0
        / (test_params["omega_m0"] * 8.0 + (1.0 - test_params["omega_m0"]))
    )
    assert np.isclose(omz[2], expected_z1)


def test_get_deltac(test_params: dict[str, float]) -> None:
    """Test the get_deltac function."""
    z_array = np.array([0.0])
    deltac = theory.get_deltac(z_array, omega_m0=test_params["omega_m0"])
    expected = 1.686 * np.power(
        theory.get_omz(z_array, omega_m0=test_params["omega_m0"]), 0.0055
    )
    assert np.isclose(deltac, expected)


def test_get_hz(test_params: dict[str, float]) -> None:
    """Test the get_hz function."""
    z_array = np.array([0.0])
    hz = theory.get_hz(z_array, omega_m0=test_params["omega_m0"])
    expected = np.sqrt(test_params["omega_m0"] + (1.0 - test_params["omega_m0"]))
    assert np.isclose(hz, expected)


def test_get_ha(test_params: dict[str, float]) -> None:
    """Test the get_ha function."""
    a_array = np.array([1.0])
    ha = theory.get_ha(a_array, omega_m0=test_params["omega_m0"])
    expected = np.sqrt(test_params["omega_m0"] + (1.0 - test_params["omega_m0"]))
    assert np.isclose(ha, expected)


def test_wth() -> None:
    """Test the wth function."""
    k = np.array([0.1, 0.5, 1.0])
    r = 1.0
    result = theory.wth(k, r)
    expected = 3.0 / (np.power(k * r, 2)) * (np.sin(k * r) / (k * r) - np.cos(k * r))
    assert np.allclose(result, expected)


def test_fh_ps_model() -> None:
    """Test the fh function with PS model."""
    s = np.array([0.5, 1.0, 1.5])
    result = theory.fh(s, model="PS")
    delta_c = 1.686
    nu = delta_c / s
    expected = np.sqrt(2.0 / np.pi) * nu * np.exp(-nu * nu / 2)
    assert np.allclose(result, expected)


def test_dlnsdlnm() -> None:
    """Test the dlnsdlnm function."""
    m = np.logspace(10, 15, 10)
    sigma = np.logspace(-1, -3, 10)
    result = theory.dlnsdlnm(m, sigma)

    # Test first element
    expected_first = (np.log(sigma[1]) - np.log(sigma[0])) / (
        np.log(m[1]) - np.log(m[0])
    )
    assert np.isclose(result[0], expected_first)

    # Test last element
    expected_last = (np.log(sigma[-1]) - np.log(sigma[-2])) / (
        np.log(m[-1]) - np.log(m[-2])
    )
    assert np.isclose(result[-1], expected_last)


def test_get_bh1_ps_model() -> None:
    """Test the get_bh1 function with PS model."""
    m = np.logspace(10, 15, 10)
    sigma = np.array([0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4])
    result = theory.get_bh1(m, sigma=sigma, model="PS")
    delta_c = 1.686
    nu = delta_c / sigma
    expected = 1.0 + (nu * nu - 1.0) / delta_c
    assert np.allclose(result, expected)


def test_get_bh2_ps_model() -> None:
    """Test the get_bh2 function with PS model."""
    m = np.logspace(10, 15, 10)
    sigma = np.array([0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4])
    result = theory.get_bh2(m, sigma=sigma, model="PS")
    delta_c = 1.686
    nu = delta_c / sigma
    expected = np.power(nu * nu / delta_c, 2.0) - 3.0 * np.power(nu / delta_c, 2.0)
    assert np.allclose(result, expected)
