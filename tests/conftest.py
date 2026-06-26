"""conftest.py

Shared fixtures and configuration for the popinn test suite.

Centralizes two things every test file would otherwise repeat:

  * The process-global float64 flag. PINNs need x64 (see design notes); it
    must be set before any array is created, and setting it once here -- the
    earliest-imported module pytest loads -- guarantees that ordering for the
    whole suite. If a float32-assuming suite is ever added, it must run as a
    separate pytest invocation, since this flag cannot be toggled per test.

  * The small reference models (PINN / P2INN / DeepONet) and grids. These are
    session-scoped: built once and reused across every test file. The models
    are immutable eqx.Modules and tests never mutate them, so sharing one
    instance is safe and keeps the suite fast.

Fixtures defined locally in a test file shadow these, so existing files that
still declare their own (e.g. test_eval.py) keep working unchanged until they
are migrated to rely on these.
"""

import jax
# Must run before any arrays are created anywhere in the suite.
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import jax.random as jr

from jaxtyping import Array

import equinox as eqx

import pytest

from popinn import PINN, P2INN, DeepONet, ResidualTerm, eval_grid, eval_grid_flat_aux, FixedWeights, Loss

# Small dims keep everything fast; tests exercise wiring, not capacity.
NX, NT = 5, 4              # coordinate axis lengths
NA, NB = 4, 6              # scalar-parameter axis lengths (P2INN)
NF1, NF2 = 3, 2            # function-valued axis lengths (DeepONet)
NF1_SEN, NF2_SEN = 7, 5    # sensor counts per branch (deliberately unequal)

class TrainingData(eqx.Module):
    dummy_coords: tuple[Array]
    pde_coords: tuple[Array]
    aux: tuple


@pytest.fixture(scope="session")
def grids():
    x = jnp.linspace(0.0, 1.0, NX)
    t = jnp.linspace(0.0, 1.0, NT)
    a = jnp.linspace(0.5, 2.0, NA)
    b = jnp.linspace(-1.0, 1.0, NB)
    return x, t, a, b


@pytest.fixture(scope="session")
def fn_data():
    k1, k2 = jr.split(jr.PRNGKey(1))
    f1 = jr.normal(k1, (NF1, NF1_SEN))
    f2 = jr.normal(k2, (NF2, NF2_SEN))
    return f1, f2


@pytest.fixture(scope="session")
def training_data(grids):
    x, t, a, b = grids
    return TrainingData((2.*x, 2.*t), (x,t), (a,b))


@pytest.fixture(scope="session")
def pinn():
    return PINN(jr.PRNGKey(0), num_coords=2, hidden_dim=8, depth=2)


@pytest.fixture(scope="session")
def p2inn():
    return P2INN(
        jr.PRNGKey(0), num_params=2, num_coords=2,
        param_hidden_dim=8, param_depth=2,
        coord_hidden_dim=8, coord_depth=2,
        manifold_inner_dim=8, manifold_depth=2,
    )


@pytest.fixture(scope="session")
def deeponet():
    return DeepONet(
        jr.PRNGKey(2),
        branch_input_dim=(NF1_SEN, NF2_SEN),
        trunk_input_dim=2,
        branch_trunk_output_dim=8,
        branch_depth=(2, 2),
        trunk_depth=2,
    )

@pytest.fixture(scope = 'session')
def pde_residual_fn():
    def outer(model):
        def r(x, t, aux):
            return model.D(1)(x, t, aux) - model(x, t, aux)
        return r
    return outer

@pytest.fixture(scope = 'session')
def dummy_residual_fn():
    def outer(model):
        def r(x, t, aux):
            return model(x, t, aux)
        return r
    return outer

@pytest.fixture(scope="session")
def pde_residual_term(pde_residual_fn):
    fn = pde_residual_fn
    return ResidualTerm(
        name = 'pde',
        residual_fn = fn,
        metric = 'mse',
        eval_fn = eval_grid
    )

@pytest.fixture(scope="session")
def dummy_residual_term(dummy_residual_fn):
    fn = dummy_residual_fn
    return ResidualTerm(
        name = 'dummy',
        residual_fn = fn,
        metric = 'mse',
        eval_fn = eval_grid
    )

@pytest.fixture(scope="session")
def pde_residual_term_flat_aux(pde_residual_fn):
    fn = pde_residual_fn
    return ResidualTerm(
        name = 'pde',
        residual_fn = fn,
        metric = 'mse',
        eval_fn = eval_grid_flat_aux
    )

@pytest.fixture(scope="session")
def fixed_weights():
    return FixedWeights(
        values = {'pde': 2., 'dummy': 1.}
    )

@pytest.fixture(scope="session")
def loss(pde_residual_term, dummy_residual_term, fixed_weights):
    return Loss(
        [pde_residual_term, dummy_residual_term],
        fixed_weights
        )