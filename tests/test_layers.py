"""
Test the layers of the SCNF using pytest.
"""

# Import the core modules
from typing import Any

# Jax related modules
import e3nn_jax as e3nn
import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jrandom
import pytest
from jaxtyping import Array, Float, PRNGKeyArray

# Import the module with the lauers
from scnf import layers


# Define a function to transform a grid under E(3)
def grid_transform(
    grid: Float[Array, "ND ND ND"],
    rotations: list[int] = [0, 0, 0],
    shifts: list[int] = [0, 0, 0],
) -> jnp.ndarray:
    """Transform a grid under E(3) operations (rotations and shifts).

    This function applies a series of 90-degree rotations and cyclic shifts
    to a 3D JAX array (grid), simulating E(3) transformations.

    :param grid: The input 3D JAX array (grid) to be transformed.
    :type grid: Float[Array, "ND ND ND"]
    :param rotations: A list of three integers [k0, k1, k2] specifying the
        number of 90-degree rotations to apply around axes (1,2), (0,2), and (0,1)
        respectively. Defaults to [0, 0, 0] (no rotations).
    :type rotations: list
    :param shifts: A list of three integers [s0, s1, s2] specifying the
        number of cyclic shifts to apply along each of the three axes.
        Defaults to [0, 0, 0] (no shifts).
    :type shifts: list

    :returns: The transformed 3D JAX array.
    :rtype: jnp.ndarray
    """
    # Do not transform inplace
    transformed_grid = grid.copy()

    # # Apply the shifts
    for axis in range(3):
        transformed_grid = jnp.roll(
            transformed_grid, shift=shifts[axis], axis=axis
        )

    # Apply the rotations
    transformed_grid = jnp.rot90(transformed_grid, k=rotations[0], axes=(1, 2))
    transformed_grid = jnp.rot90(transformed_grid, k=rotations[1], axes=(0, 2))
    transformed_grid = jnp.rot90(transformed_grid, k=rotations[2], axes=(0, 1))

    return transformed_grid


# --- Fixtures for test setup ---


@pytest.fixture(scope="module")
def test_params() -> dict[str, Any]:
    """Defines common parameters for all tests."""
    return {
        "ND": 32,
        "NGRIDS": 10,
        "SEED": 12345,
        "CONV_IRREPS": [
            e3nn.Irreps("1x0e"),
            e3nn.Irreps("4x0e"),
            e3nn.Irreps("11x0e+1x2e"),
            e3nn.Irreps("19x0e+1x1o+2x2e"),
            e3nn.Irreps("64x0e"),
        ],
        "CONV_CHANNELS": [1, 4, 16, 32, 64],
        "KERNEL_SIZE": 3,
        "KERNEL_FOURIER_SIZE": 32,
        "STRIDE": 1,
        "CELL_SIZE": 1.0,
        "N_NEURONS_LINS": [64, 32, 16, 8],
        "N_NEURONS_RADIAL": [4, 4],
        "POOLING_SIZE": 1,
        "KERNEL_POOLING_SIZE": 1,
        "IN_SIZE": 10,
        "OUT_SIZE": 10,
        "TIME_SIZE": 100,
        "TOLERANCE": 1e-5,
        "KERNEL_POOLING_TYPE": "exp",
    }


@pytest.fixture(scope="module")
def rng_key(test_params: dict[str, int]) -> PRNGKeyArray:
    """Provides a JAX PRNG key."""
    return jrandom.PRNGKey(test_params["SEED"])


@pytest.fixture(scope="module")
def rotation_and_shift_arrays(
    rng_key: PRNGKeyArray, test_params: dict[str, Any]
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Generates random rotation and shift arrays."""
    key_rotation, key_shift = jrandom.split(rng_key, num=2)
    rotations = jrandom.randint(key_rotation, (test_params["NGRIDS"], 3), 0, 4)
    shifts = jrandom.randint(
        key_shift,
        (test_params["NGRIDS"], 3),
        -test_params["ND"] // 2,
        test_params["ND"] // 2 + 1,
    )
    return rotations, shifts


@pytest.fixture(scope="module")
def input_grids(
    rng_key: PRNGKeyArray,
    rotation_and_shift_arrays: tuple[jnp.ndarray, jnp.ndarray],
    test_params: dict[str, Any],
) -> tuple[e3nn.IrrepsArray, e3nn.IrrepsArray]:
    """Creates initial and transformed grids."""
    key = rng_key
    rotations, shifts = rotation_and_shift_arrays

    grid1_array: Float[Array, "NGRIDS ND ND ND"] = jrandom.normal(
        key,
        shape=(
            test_params["NGRIDS"],
            test_params["ND"],
            test_params["ND"],
            test_params["ND"],
        ),
    )
    grid1 = e3nn.IrrepsArray(
        irreps=test_params["CONV_IRREPS"][0],
        array=grid1_array[:, :, :, :, jnp.newaxis],
    )

    grid2_list: list[Float[Array, "ND ND ND"]] = []
    for i in range(test_params["NGRIDS"]):
        grid2_list.append(
            grid_transform(
                grid1.array[i, :, :, :, 0],
                rotations=rotations[i].tolist(),
                shifts=shifts[i].tolist(),
            )
        )
    grid2_array: Float[Array, "NGRIDS ND ND ND"] = jnp.array(grid2_list)
    grid2 = e3nn.IrrepsArray(
        irreps=test_params["CONV_IRREPS"][0],
        array=grid2_array[:, :, :, :, jnp.newaxis],
    )
    return grid1, grid2


@pytest.fixture(scope="module")
def compress_nd_layer(
    rng_key: PRNGKeyArray, test_params: dict[str, Any]
) -> eqx.Module:
    """Initializes a non-equivariant compression layer."""
    key_compress = rng_key
    compress_nd = layers.compress_nd(
        key=key_compress,
        dimension=3,
        kernel_size=test_params["KERNEL_SIZE"],
        conv_channels=test_params["CONV_CHANNELS"],
        n_neurons_lins=test_params["N_NEURONS_LINS"],
        pooling_stride=test_params["POOLING_SIZE"],
        kernel_pooling_size=test_params["KERNEL_POOLING_SIZE"],
    )
    return compress_nd


@pytest.fixture(scope="module")
def compress_e3_layer(
    rng_key: PRNGKeyArray, test_params: dict[str, Any]
) -> eqx.Module:
    """Initializes an E(3)-equivariant compression layer."""
    key_compress = rng_key
    compress_e3 = layers.compress_3d_e3(
        key=key_compress,
        kernel_size=test_params["KERNEL_SIZE"],
        cell_size=test_params["CELL_SIZE"],
        conv_irreps=test_params["CONV_IRREPS"],
        n_neurons_lins=test_params["N_NEURONS_LINS"],
        n_neurons_radial=test_params["N_NEURONS_RADIAL"],
        pooling_stride=test_params["POOLING_SIZE"],
        kernel_pooling_size=test_params["KERNEL_POOLING_SIZE"],
    )
    compress_e3.compute_kernels()
    return compress_e3


@pytest.fixture(scope="module")
def compress_fourier_e3_layer(
    rng_key: PRNGKeyArray, test_params: dict[str, Any]
) -> eqx.Module:
    """Initializes an E(3)-equivariant compression layer."""
    key_compress = rng_key
    compress_e3 = layers.compress_fourier_3d_e3(
        key=key_compress,
        grid_size=[test_params["ND"], test_params["ND"], test_params["ND"]],
        cell_size=test_params["CELL_SIZE"],
        conv_irreps=test_params["CONV_IRREPS"],
        n_neurons_lins=test_params["N_NEURONS_LINS"],
        n_neurons_radial=test_params["N_NEURONS_RADIAL"],
        downsampling_factor=test_params["POOLING_SIZE"],
    )
    compress_e3.compute_kernels()
    return compress_e3


@pytest.fixture(scope="module")
def pool_layer(test_params: dict[str, Any]) -> eqx.nn.AvgPool3d:
    """Initializes a pooling layer."""
    pool = eqx.nn.AvgPool3d(
        kernel_size=test_params["KERNEL_POOLING_SIZE"],
        stride=test_params["POOLING_SIZE"],
        padding=0,
    )
    return pool


@pytest.fixture(scope="module")
def pool_e3_layer(test_params: dict[str, Any]) -> eqx.Module:
    """Initializes a pooling layer."""
    pool = layers.pool_e3_layer(
        kernel_size=test_params["KERNEL_POOLING_SIZE"],
        stride=test_params["POOLING_SIZE"],
        cell_size=test_params["CELL_SIZE"],
        kernel_type=test_params["KERNEL_POOLING_TYPE"],
    )
    return pool


@pytest.fixture(scope="module")
def concat_layer(
    rng_key: PRNGKeyArray, test_params: dict[str, Any]
) -> eqx.Module:
    """Initializes concatenation layers (non-equivariant and E(3)-equivariant)."""
    key_concat = rng_key

    concat = layers.concat_layer(
        key=key_concat,
        in_size=test_params["IN_SIZE"],
        out_size=test_params["OUT_SIZE"],
        compressed_grid_size=test_params["N_NEURONS_LINS"][-1],
    )
    return concat


# --- Test Functions ---


def test_grid_transform(
    input_grids: tuple[e3nn.IrrepsArray, e3nn.IrrepsArray],
    rotation_and_shift_arrays: tuple[jnp.ndarray, jnp.ndarray],
    test_params: dict[str, Any],
) -> None:
    """Test the grid_transform function for basic functionality.

    This test is mainly to ensure the helper function works as expected.
    The primary invariance tests are for the layers.

    :param input_grids: A tuple containing original and transformed input grids.
    :type input_grids: tuple
    :param rotation_and_shift_arrays: A tuple containing arrays of rotations and shifts.
    :type rotation_and_shift_arrays: tuple
    :param test_params: Dictionary of test parameters.
    :type test_params: dict
    """
    grid1, grid2 = input_grids
    rotations, shifts = rotation_and_shift_arrays

    # Pick one grid and transform it
    idx = 0
    original_grid: Float[Array, "ND ND ND"] = grid1.array[idx, :, :, :, 0]
    transformed_grid_manual = grid_transform(
        original_grid,
        rotations=rotations[idx].tolist(),
        shifts=shifts[idx].tolist(),
    )

    # Assert that the manually transformed grid is close to grid2 (which was pre-transformed)
    assert jnp.allclose(
        transformed_grid_manual,
        grid2.array[idx, :, :, :, 0],
        atol=test_params["TOLERANCE"],
    )


def test_conv_fourier_e3(
    input_grids: tuple[e3nn.IrrepsArray, e3nn.IrrepsArray],
    rng_key: PRNGKeyArray,
    test_params: dict[str, Any],
) -> None:
    """Test the E(3)-equivariant Fourier convolution layer.

    This test verifies that the `conv_fourier_e3_layer` produces
    invariant outputs under E(3) transformations (rotations and shifts).

    :param input_grids: A tuple containing original and transformed input grids.
    :type input_grids: tuple
    :param rng_key: JAX PRNG key for random number generation.
    :type rng_key: jax.random.PRNGKey
    :param test_params: Dictionary of test parameters.
    :type test_params: dict
    """
    # Initialize the layer
    conv_e3 = layers.conv_fourier_e3_layer(
        key=rng_key,
        grid_size=[test_params["ND"], test_params["ND"], test_params["ND"]],
        irreps_in=test_params["CONV_IRREPS"][0],
        irreps_out=test_params["CONV_IRREPS"][0],
        cell_size=test_params["CELL_SIZE"],
        kernel_size=test_params["KERNEL_FOURIER_SIZE"],
        n_neurons_radial=test_params["N_NEURONS_RADIAL"],
    )

    # Compute the kernel
    conv_e3.compute_kernel()

    # Apply the layer
    grid1, grid2 = input_grids
    out1: jnp.ndarray = jax.vmap(conv_e3)(grid1.array)
    out2: jnp.ndarray = jax.vmap(conv_e3)(grid2.array)

    # Flatten the output
    out1 = jnp.mean(out1, axis=(1, 2, 3, 4))
    out2 = jnp.mean(out2, axis=(1, 2, 3, 4))

    assert jnp.allclose(out1, out2, atol=test_params["TOLERANCE"])


def test_compression_invariance(
    input_grids: tuple[e3nn.IrrepsArray, e3nn.IrrepsArray],
    compress_nd_layer: layers.compress_nd,
    compress_e3_layer: layers.compress_3d_e3,
    test_params: dict[str, Any],
) -> None:
    """Test the invariance of the compression layers.

    This test verifies that the E(3)-equivariant compression layer
    (compress_e3_layer) is more invariant to E(3) transformations
    (rotations and shifts) than the non-equivariant compression layer
    (compress_nd_layer). It does this by comparing the difference between
    compressed versions of an original grid and its transformed counterpart.

    :param input_grids: A tuple containing original and transformed input grids.
    :type input_grids: tuple
    :param compress_nd_layer: An instance of the non-equivariant compression layer.
    :type compress_nd_layer: equinox.Module
    :param compress_e3_layer: An instance of the E(3)-equivariant compression layer.
    :type compress_e3_layer: equinox.Module
    :param test_params: Dictionary of test parameters.
    :type test_params: dict
    """
    # Compress the grids
    grid1, grid2 = input_grids
    compressed1: jnp.ndarray = jax.vmap(compress_nd_layer)(grid1.array)
    compressed2: jnp.ndarray = jax.vmap(compress_nd_layer)(grid2.array)
    compressed1_eq: jnp.ndarray = jax.vmap(compress_e3_layer)(grid1.array)
    compressed2_eq: jnp.ndarray = jax.vmap(compress_e3_layer)(grid2.array)

    # Calculate the relative difference
    rate = jnp.mean(jnp.power((compressed1 / compressed2 - 1.0), 2))
    rate_eq = jnp.mean(jnp.power((compressed1_eq / compressed2_eq - 1.0), 2))

    # Check that the equivariant layer is more invariant than the non-equivariant one
    assert rate + test_params["TOLERANCE"] >= rate_eq


def test_compression_fourier_invariance(
    input_grids: tuple[e3nn.IrrepsArray, e3nn.IrrepsArray],
    compress_nd_layer: layers.compress_nd,
    compress_fourier_e3_layer: layers.compress_fourier_3d_e3,
    test_params: dict[str, Any],
) -> None:
    """Test the invariance of the Fourier-based compression layers.

    This test verifies that the E(3)-equivariant Fourier compression layer
    (compress_fourier_e3_layer) is more invariant to E(3) transformations
    (rotations and shifts) than the non-equivariant compression layer
    (compress_nd_layer). It does this by comparing the difference between
    compressed versions of an original grid and its transformed counterpart.

    :param input_grids: A tuple containing original and transformed input grids.
    :type input_grids: tuple
    :param compress_nd_layer: An instance of the non-equivariant compression layer.
    :type compress_nd_layer: equinox.Module
    :param compress_fourier_e3_layer: An instance of the E(3)-equivariant Fourier compression layer.
    :type compress_fourier_e3_layer: equinox.Module
    :param test_params: Dictionary of test parameters.
    :type test_params: dict
    """
    # Compress the grids
    grid1, grid2 = input_grids
    compressed1_eq: Array = jax.vmap(compress_fourier_e3_layer)(grid1.array)
    compressed2_eq: Array = jax.vmap(compress_fourier_e3_layer)(grid2.array)
    compressed1: Array = jax.vmap(compress_nd_layer)(grid1.array)
    compressed2: Array = jax.vmap(compress_nd_layer)(grid2.array)

    # Calculate the relative difference
    rate = jnp.mean(jnp.power(compressed1 / compressed2 - 1.0, 2))
    rate_eq = jnp.mean(jnp.power(compressed1_eq / compressed2_eq - 1.0, 2))

    # Check that the equivariant layer is more invariant than the non-equivariant one
    assert rate + test_params["TOLERANCE"] >= rate_eq


def test_pooling_layer(
    input_grids: tuple[e3nn.IrrepsArray, e3nn.IrrepsArray],
    pool_layer: eqx.nn.AvgPool3d,
    pool_e3_layer: layers.pool_e3_layer,
    test_params: dict[str, Any],
) -> None:
    """Test the invariance of the pooling layers.

    This test verifies that the E(3)-equivariant pooling layer (pool_e3_layer)
    is more invariant to E(3) transformations (rotations and shifts) than
    the non-equivariant pooling layer (pool_layer). It does this by comparing
    the difference between pooled versions of an original grid and its
    transformed counterpart.

    :param input_grids: A tuple containing original and transformed input grids.
    :type input_grids: tuple
    :param pool_layer: An instance of the non-equivariant pooling layer.
    :type pool_layer: equinox.nn.AvgPool3d
    :param pool_e3_layer: An instance of the E(3)-equivariant pooling layer.
    :type pool_e3_layer: equinox.Module
    """
    # Pool the grids
    grid1, grid2 = input_grids
    gridt1: jnp.ndarray = jnp.transpose(grid1.array, (0, 4, 1, 2, 3))
    gridt2: jnp.ndarray = jnp.transpose(grid2.array, (0, 4, 1, 2, 3))

    pooled1: jnp.ndarray = jnp.transpose(
        jax.vmap(pool_layer)(gridt1), (0, 2, 3, 4, 1)
    )
    pooled2: jnp.ndarray = jnp.transpose(
        jax.vmap(pool_layer)(gridt2), (0, 2, 3, 4, 1)
    )
    pooled1_eq: jnp.ndarray = jax.vmap(pool_e3_layer)(grid1.array)
    pooled2_eq: jnp.ndarray = jax.vmap(pool_e3_layer)(grid2.array)

    # Calculate the relative difference
    mean1 = jnp.mean(pooled1, axis=(1, 2, 3, 4))
    mean2 = jnp.mean(pooled2, axis=(1, 2, 3, 4))
    mean1_eq = jnp.mean(pooled1_eq, axis=(1, 2, 3, 4))
    mean2_eq = jnp.mean(pooled2_eq, axis=(1, 2, 3, 4))
    rate = jnp.mean(jnp.power((mean1 - mean2), 2))
    rate_eq = jnp.mean(jnp.power((mean1_eq - mean2_eq), 2))

    # Check that the equivariant layer is more invariant than the non-equivariant one
    assert rate + test_params["TOLERANCE"] >= rate_eq


def test_concat_invariance(
    input_grids: tuple[e3nn.IrrepsArray, e3nn.IrrepsArray],
    concat_layer: layers.concat_layer,
    compress_nd_layer: layers.compress_nd,
    compress_e3_layer: layers.compress_3d_e3,
    rng_key: PRNGKeyArray,
    test_params: dict[str, Any],
) -> None:
    """Test the invariance of the concatenation layers.

    This test verifies that the E(3)-equivariant concatenation layer
    (concat_eq) is more invariant to E(3) transformations
    (rotations and shifts) than the non-equivariant concatenation layer
    (concat). It does this by comparing the difference between
    concatenated outputs of an original grid and its transformed counterpart.

    :param input_grids: A tuple containing original and transformed input grids.
    :type input_grids: tuple
    :param concat_layer: An instance of the concatenation layer.
    :type concat_layer: equinox.Module
    :param compress_nd_layer: An instance of the non-equivariant compression layer.
    :type compress_nd_layer: equinox.Module
    :param compress_e3_layer: An instance of the E(3)-equivariant compression layer.
    :type compress_e3_layer: equinox.Module
    :param rng_key: JAX PRNG key for random number generation.
    :type rng_key: jax.random.PRNGKey
    :param test_params: Dictionary of test parameters.
    :type test_params: dict
    """
    grid1, grid2 = input_grids

    # Create the input array and time
    key_inputs = rng_key
    inputs: jnp.ndarray = jrandom.normal(
        key_inputs, shape=(test_params["NGRIDS"], test_params["IN_SIZE"])
    )
    times: jnp.ndarray = jnp.linspace(0.0, 1.0, test_params["TIME_SIZE"])

    # Compress the grids
    compressed1: jnp.ndarray = jax.vmap(compress_nd_layer)(grid1.array)
    compressed2: jnp.ndarray = jax.vmap(compress_nd_layer)(grid2.array)
    compressed1_eq: jnp.ndarray = jax.vmap(compress_e3_layer)(grid1.array)
    compressed2_eq: jnp.ndarray = jax.vmap(compress_e3_layer)(grid2.array)

    # Compute the concatenated output
    output1: jnp.ndarray = jax.vmap(
        jax.vmap(concat_layer, in_axes=(0, None, None)), in_axes=(None, 0, 0)
    )(times, inputs, compressed1)
    output2: jnp.ndarray = jax.vmap(
        jax.vmap(concat_layer, in_axes=(0, None, None)), in_axes=(None, 0, 0)
    )(times, inputs, compressed2)
    output1_eq: jnp.ndarray = jax.vmap(
        jax.vmap(concat_layer, in_axes=(0, None, None)), in_axes=(None, 0, 0)
    )(times, inputs, compressed1_eq)
    output2_eq: jnp.ndarray = jax.vmap(
        jax.vmap(concat_layer, in_axes=(0, None, None)), in_axes=(None, 0, 0)
    )(times, inputs, compressed2_eq)

    # Calculate the relative difference
    rate = jnp.mean(jnp.power(output1 / output2 - 1.0, 2))
    rate_eq = jnp.mean(jnp.power(output1_eq / output2_eq - 1.0, 2))

    # The output shape should be (OUT_SIZE,)
    assert rate + test_params["TOLERANCE"] >= rate_eq


def test_concat_fourier_invariance(
    input_grids: tuple[e3nn.IrrepsArray, e3nn.IrrepsArray],
    concat_layer: layers.concat_layer,
    compress_nd_layer: layers.compress_nd,
    compress_fourier_e3_layer: layers.compress_fourier_3d_e3,
    rng_key: PRNGKeyArray,
    test_params: dict[str, Any],
) -> None:
    """Test the invariance of the Fourier-based concatenation layers.

    This test verifies that the E(3)-equivariant Fourier concatenation layer
    (concat_eq) is more invariant to E(3) transformations
    (rotations and shifts) than the non-equivariant concatenation layer
    (concat). It does this by comparing the difference between
    concatenated outputs of an original grid and its transformed counterpart.

    :param input_grids: A tuple containing original and transformed input grids.
    :type input_grids: tuple
    :param concat_layer: An instance of the concatenation layer.
    :type concat_layer: equinox.Module
    :param compress_nd_layer: An instance of the non-equivariant compression layer.
    :type compress_nd_layer: equinox.Module
    :param compress_fourier_e3_layer: An instance of the E(3)-equivariant Fourier compression layer.
    :type compress_fourier_e3_layer: equinox.Module
    :param rng_key: JAX PRNG key for random number generation.
    :type rng_key: jax.random.PRNGKey
    :param test_params: Dictionary of test parameters.
    :type test_params: dict
    """
    grid1, grid2 = input_grids

    # Create the input array and time
    key_inputs = rng_key
    inputs: jnp.ndarray = jrandom.normal(
        key_inputs, shape=(test_params["NGRIDS"], test_params["IN_SIZE"])
    )
    times: jnp.ndarray = jnp.linspace(0.0, 1.0, test_params["TIME_SIZE"])

    # Compress the grids
    compressed1: jnp.ndarray = jax.vmap(compress_nd_layer)(grid1.array)
    compressed2: jnp.ndarray = jax.vmap(compress_nd_layer)(grid2.array)
    compressed1_eq: jnp.ndarray = jax.vmap(compress_fourier_e3_layer)(
        grid1.array
    )
    compressed2_eq: jnp.ndarray = jax.vmap(compress_fourier_e3_layer)(
        grid2.array
    )

    # Compute the concatenated output
    output1: jnp.ndarray = jax.vmap(
        jax.vmap(concat_layer, in_axes=(0, None, None)), in_axes=(None, 0, 0)
    )(times, inputs, compressed1)
    output2: jnp.ndarray = jax.vmap(
        jax.vmap(concat_layer, in_axes=(0, None, None)), in_axes=(None, 0, 0)
    )(times, inputs, compressed2)
    output1_eq: jnp.ndarray = jax.vmap(
        jax.vmap(concat_layer, in_axes=(0, None, None)), in_axes=(None, 0, 0)
    )(times, inputs, compressed1_eq)
    output2_eq: jnp.ndarray = jax.vmap(
        jax.vmap(concat_layer, in_axes=(0, None, None)), in_axes=(None, 0, 0)
    )(times, inputs, compressed2_eq)

    # Calculate the relative difference
    rate = jnp.mean(jnp.power(output1 / output2 - 1.0, 2))
    rate_eq = jnp.mean(jnp.power(output1_eq / output2_eq - 1.0, 2))

    # The output shape should be (OUT_SIZE,)
    assert rate + test_params["TOLERANCE"] >= rate_eq
