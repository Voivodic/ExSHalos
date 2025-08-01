"""
Test the theory module functions using pytest.
"""

# Import the core modules
from typing import Any

import numpy as np
import pytest

# Import the module with the theory functions
from pyexshalos import theory

# --- Fixtures for test setup ---


@pytest.fixture(scope="module")
def test_params() -> dict[str, Any]:
    """Defines common parameters for all tests."""
    return {
        "omega_m0": 0.31,
        "omega_mz": 0.78233,
        "delta_c": 1.68372,
        "hz": 1.78045,
        "radial_distance": 2.29911e3,
        "angular_distance": 1.47909e3,
        "delta_ra": np.pi / 3.0,
        "delta_dec": np.pi / 4.0,
        "z_min": 0.3,
        "z_max": 0.5,
        "box_shape": (1.37967e3, 1.03475e3, 0.48322e3),
        "z": 1.0,
        "cell_size": 2.0,
        "cell_mass": 6.882e11,
        "tolerance": 1e-5,
    }


# --- Test Functions ---


def test_get_cell_mass(test_params: dict[str, Any]) -> None:
    """Test the get_cell_mass function."""
    mass = theory.get_cell_mass(
        cell_size=test_params["cell_size"], omega_m0=test_params["omega_m0"]
    )

    assert np.isclose(
        test_params["cell_mass"], mass, rtol=test_params["tolerance"]
    )


def test_get_cell_size(test_params: dict[str, Any]) -> None:
    """Test the get_cell_size function."""
    size = theory.get_cell_size(
        omega_m0=test_params["omega_m0"], m_cell=test_params["cell_mass"]
    )

    assert np.isclose(
        test_params["cell_size"], size, rtol=test_params["tolerance"]
    )


def test_get_omz(test_params: dict[str, Any]) -> None:
    """Test the get_omz function."""
    omz = theory.get_omz(
        np.array(test_params["z"]), omega_m0=test_params["omega_m0"]
    )

    assert np.isclose(
        test_params["omega_mz"], omz.item(), rtol=test_params["tolerance"]
    )


def test_get_deltac(test_params: dict[str, Any]) -> None:
    """Test the get_deltac function."""
    deltac = theory.get_deltac(
        np.array(test_params["z"]), omega_m0=test_params["omega_m0"]
    )

    assert np.isclose(
        test_params["delta_c"], deltac.item(), rtol=test_params["tolerance"]
    )


def test_get_hz(test_params: dict[str, Any]) -> None:
    """Test the get_hz function."""
    hz = theory.get_hz(
        np.array(test_params["z"]), omega_m0=test_params["omega_m0"]
    )

    assert np.isclose(
        test_params["hz"], hz.item(), rtol=test_params["tolerance"]
    )


def test_get_ha(test_params: dict[str, Any]) -> None:
    """Test the get_ha function."""
    a = 1.0 / (1.0 + test_params["z"])
    ha = theory.get_ha(np.array(a), omega_m0=test_params["omega_m0"])

    assert np.isclose(
        test_params["hz"], ha.item(), rtol=test_params["tolerance"]
    )


def test_get_radial_distance(test_params: dict[str, Any]) -> None:
    """Test the get_radial_distance function."""
    dist = theory.get_radial_distance(
        np.array(test_params["z"]), omega_m0=test_params["omega_m0"]
    )

    assert np.isclose(
        test_params["radial_distance"],
        dist.item(),
        rtol=test_params["tolerance"],
    )


def test_get_angular_distance(test_params: dict[str, Any]) -> None:
    """Test the get_radial_distance function."""
    dist = theory.get_angular_distance(
        np.array(test_params["z"]),
        ra_0=0.0,
        ra_1=test_params["delta_ra"],
        dec_0=-test_params["delta_dec"] / 2.0,
        dec_1=test_params["delta_dec"] / 2.0,
        omega_m0=test_params["omega_m0"],
    )

    assert np.isclose(
        test_params["angular_distance"],
        dist.item(),
        rtol=test_params["tolerance"],
    )


def test_get_box_shape(test_params: dict[str, Any]) -> None:
    """Test the get_box_shape function."""
    shape = theory.get_box_shape(
        ra_min=0.0,
        ra_max=test_params["delta_ra"],
        dec_min=-test_params["delta_dec"] / 2.0,
        dec_max=test_params["delta_dec"] / 2.0,
        z_min=test_params["z_min"],
        z_max=test_params["z_max"],
        omega_m0=test_params["omega_m0"],
        in_rad=True,
    )

    assert np.allclose(
        shape,
        test_params["box_shape"],
        rtol=test_params["tolerance"],
    )
