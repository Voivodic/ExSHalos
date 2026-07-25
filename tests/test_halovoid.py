"""
Test the halovoid module
"""

import pytest
import numpy as np
import pyexshalos as exs


@pytest.fixture(scope="module")
def params() -> dict:
    """
    Defines the parameters used by the tests.
    """
    return {
        "L": 100.0,
        "Nd": 20,
        "Np": 1_00_000,
        "delta_h": 4.0,
        "delta_v": 0.4,
        "r_max": 20.0,
    }


def test_total_cell_volume(params: dict) -> None:
    """
    Test if the total volume is recovered correctly.
    """
    x = params["L"] * np.random.random((params["Np"], 3))
    volume = exs.simulation.total_volume(x, params["Nd"], params["L"])

    assert np.isclose(volume, params["L"] ** 3, rtol=1e-3)


def test_voronoi_computation(params: dict) -> None:
    """
    Test if the Voronoi diagram is computed correctly.
    """
    x = params["L"] * np.random.random((params["Np"], 3))
    resp = exs.simulation.halo_void_finder(
        x,
        params["L"],
        params["delta_h"],
        params["delta_v"],
        params["Nd"],
        params["r_max"],
    )

    rho_voids = params["delta_v"] * params["Np"] / pow(params["L"], 3)
    rho_halos = params["delta_h"] * params["Nd"] / pow(params["L"], 3)
    print(resp["halos_den"].shape, resp["voids_den"].shape)
    print(resp["voids_pos"][:,0].shape , np.unique(resp["voids_pos"][:,0]).shape)
    print(resp["voids_pos"][:,1].shape , np.unique(resp["voids_pos"][:,1]).shape)
    print(resp["voids_pos"][:,2].shape , np.unique(resp["voids_pos"][:,2]).shape)
    print(resp["voids_den"].shape , np.unique(resp["voids_den"]).shape)

    assert False#np.all(resp["voids_den"] <= rho_voids) and np.all(resp["halos_den"] >= rho_halos)
