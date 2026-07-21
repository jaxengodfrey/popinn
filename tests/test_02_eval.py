"""test_eval.py

Pytest suite for `eval_grid` and `eval_grid_flat_aux` in eval.py, under the
tuple-aux convention: per-point models take fn(*coords, aux) where aux is a
TUPLE of auxiliary components (scalar parameters or sensor-sampled function
arrays of any shapes). Each top-level aux element is one grid axis; a
sub-tuple element is zipped (its leaves share one axis).

Run with:  pytest test_eval.py -v

Coverage, organized by what it pins down:

  Correctness & layout   -- values and axis order vs. an explicit Python loop,
                            for the three model types (PINN/P2INN/DeepONet).
  Strategy equivalence   -- full vmap vs. outer-chunk vs. joint-chunk agree,
                            including non-dividing chunk sizes (remainders).
  aux structure          -- vector/ragged components, and grouped (zipped)
                            components that share one axis.
  Per-point contract     -- scalar output, derivatives via model.D batched
                            through eval, and gradient flow through the
                            chunked loss path (the training use case).

Precision note: float64 is enabled suite-wide in conftest.py. Different execution strategies
compile to different XLA kernels whose floating-point reduction orders differ,
so float32 results agree only to ~1e-5 relative; in float64 the reordering
noise is ~1e-13, letting the tolerances below be tight enough that a real bug
(wrong parameter/output pairing -> O(1) relative error) fails unambiguously.
The x64 flag is process-global and is set once in conftest.py; a float32-assuming
suite would need a separate pytest invocation.
"""

import itertools

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

# Models, grids, fn_data and the axis-length constants are shared via conftest.py
# (which also enables float64 before any array is created). Importing the
# constants here keeps the shape assertions and parametrizations in sync with
# the fixtures that produce the data.
from conftest import NA, NB, NF1, NF1_SEN, NF2, NF2_SEN, NT, NX

from popinn import eval_grid, eval_grid_flat_aux

# Tight tolerances are intentional -- see precision note above.
RTOL, ATOL = 1e-9, 1e-12


# ──────────────────────────────────────────────────────────────
# Reference implementation: explicit per-point loop
# ──────────────────────────────────────────────────────────────


def reference_grid(fn, coords_1d, aux):
    """Slow, unambiguous ground truth.

    Calls fn one point at a time with a tuple aux slice and writes the result
    at the index the documented convention prescribes -- axes in REVERSED
    argument order, aux block leading:

        out[i_last_aux, ..., i_first_aux, j_last_coord, ..., j_first_coord]

    Any disagreement with eval_grid is a real bug (axis order or
    parameter/output pairing), not floating-point noise. Handles grouped
    (sub-tuple) aux components: a component's axis length is its first leaf's
    leading dim, and slicing indexes every leaf by the same index.
    """

    def axis_len(comp):
        return jax.tree_util.tree_leaves(comp)[0].shape[0]

    def slice_comp(comp, i):
        return jax.tree_util.tree_map(lambda leaf: leaf[i], comp)

    aux_lens = [axis_len(c) for c in aux]
    coord_lens = [c.shape[0] for c in coords_1d]
    out = np.zeros(tuple(reversed(aux_lens)) + tuple(reversed(coord_lens)))

    for aux_idx in itertools.product(*(range(n) for n in aux_lens)):
        aux_slice = tuple(slice_comp(c, i) for c, i in zip(aux, aux_idx))
        for coord_idx in itertools.product(*(range(n) for n in coord_lens)):
            pt = [c[j] for c, j in zip(coords_1d, coord_idx)]
            out[tuple(reversed(aux_idx)) + tuple(reversed(coord_idx))] = fn(*pt, aux_slice)
    return jnp.asarray(out)


# ──────────────────────────────────────────────────────────────
# Correctness: values and axis order vs. the reference loop
# ──────────────────────────────────────────────────────────────


def test_p2inn_matches_reference(p2inn, grids):
    """Scalar-parameter aux: crossed (a, b) grid, P2INN."""
    x, t, a, b = grids
    got = eval_grid(p2inn, (x, t), (a, b))
    want = reference_grid(p2inn, (x, t), (a, b))
    assert got.shape == (NB, NA, NT, NX)
    assert jnp.allclose(got, want, rtol=RTOL, atol=ATOL)


def test_deeponet_matches_reference(deeponet, grids, fn_data):
    """Vector-valued aux with unequal sensor counts (the ragged case is not a
    special code path under the tuple convention), DeepONet."""
    x, t, _, _ = grids
    f1, f2 = fn_data
    got = eval_grid(deeponet, (x, t), (f1, f2))
    want = reference_grid(deeponet, (x, t), (f1, f2))
    assert got.shape == (NF2, NF1, NT, NX)
    assert jnp.allclose(got, want, rtol=RTOL, atol=ATOL)


def test_axis_order_single_point(p2inn, grids):
    """Spot-check the layout convention directly:
    out[l, k, j, i] == fn(x[i], t[j], (a[k], b[l]))."""
    x, t, a, b = grids
    u = eval_grid(p2inn, (x, t), (a, b))
    i, j, k, l = 4, 1, 2, 1
    assert jnp.allclose(u[l, k, j, i], p2inn(x[i], t[j], (a[k], b[l])), rtol=RTOL, atol=ATOL)


# ──────────────────────────────────────────────────────────────
# Strategy equivalence: chunked paths == full vmap
# ──────────────────────────────────────────────────────────────


# 1 = fully sequential; 5 = non-divisor of NA*NB = 24 (remainder); NA*NB = single chunk.
@pytest.mark.parametrize("batch_size", [1, 5, NA * NB])
def test_joint_chunk_matches_full_vmap(p2inn, grids, batch_size):
    x, t, a, b = grids
    full = eval_grid(p2inn, (x, t), (a, b))
    print(batch_size)
    flat = eval_grid_flat_aux(p2inn, (x, t), (a, b), batch_size=batch_size)
    assert flat.shape == full.shape
    assert jnp.allclose(flat, full, rtol=RTOL, atol=ATOL)


@pytest.mark.parametrize("outer_batch_size", [1, 2])
def test_outer_chunk_matches_full_vmap(p2inn, grids, outer_batch_size):
    x, t, a, b = grids
    full = eval_grid(p2inn, (x, t), (a, b))
    chunked = eval_grid(p2inn, (x, t), (a, b), outer_batch_size=outer_batch_size)
    assert chunked.shape == full.shape
    assert jnp.allclose(chunked, full, rtol=RTOL, atol=ATOL)


def test_chunked_vector_aux(deeponet, grids, fn_data):
    """Both chunking paths handle vector/ragged components, which are gathered
    by integer index rather than meshgridded."""
    x, t, _, _ = grids
    f1, f2 = fn_data
    full = eval_grid(deeponet, (x, t), (f1, f2))
    assert jnp.allclose(eval_grid_flat_aux(deeponet, (x, t), (f1, f2), batch_size=4), full, rtol=RTOL, atol=ATOL)
    assert jnp.allclose(eval_grid(deeponet, (x, t), (f1, f2), outer_batch_size=1), full, rtol=RTOL, atol=ATOL)


# ──────────────────────────────────────────────────────────────
# aux structure: grouped (zipped) components
# ──────────────────────────────────────────────────────────────


def test_grouped_aux_zips_paired_components(deeponet, grids, fn_data):
    """A grouped sub-tuple shares ONE axis (zipped), crossed against an
    independent component. Models the DeepONet case where a PDE coefficient
    is paired with the function it generated: aux = (f1, (f2, gamma)) ->
    f1 crossed with the (f2, gamma) pairs. Checked against the reference loop
    and across all execution strategies."""
    x, t, _, _ = grids
    f1, f2 = fn_data
    gammas = jnp.linspace(0.5, 2.0, NF2)  # paired with f2 along axis 0

    # Residual-shaped fn that consumes both the grouped function and its scalar.
    def fn(x_, t_, aux):
        f1_i, (f2_i, gamma) = aux
        return deeponet(x_, t_, (f1_i, f2_i)) * gamma

    aux = (f1, (f2, gammas))
    full = eval_grid(fn, (x, t), aux)
    assert full.shape == (NF2, NF1, NT, NX)
    assert jnp.allclose(full, reference_grid(fn, (x, t), aux), rtol=RTOL, atol=ATOL)

    # Chunking must preserve the zip pairing (leaf-wise gather by one index).
    assert jnp.allclose(eval_grid_flat_aux(fn, (x, t), aux, batch_size=4), full, rtol=RTOL, atol=ATOL)
    assert jnp.allclose(eval_grid(fn, (x, t), aux, outer_batch_size=1), full, rtol=RTOL, atol=ATOL)


# ──────────────────────────────────────────────────────────────
# Per-point contract: scalar output, optional aux, derivatives, grads
# ──────────────────────────────────────────────────────────────


def test_scalar_output_contract(pinn, p2inn, deeponet):
    """jax.grad / model.D require a true scalar (shape (), not (1,)). Catches
    the eqx.nn.MLP out_size=1 vs 'scalar' trap before it surfaces as a cryptic
    grad error deep in the vmap stack."""
    assert pinn(0.3, 0.7).shape == ()
    assert p2inn(0.3, 0.7, (1.0, -0.5)).shape == ()
    assert deeponet(0.3, 0.7, (jnp.zeros(NF1_SEN), jnp.zeros(NF2_SEN))).shape == ()


def test_empty_aux_pure_pinn(pinn, grids):
    """A PINN has no aux. model(x, t), model(x, t, ()), and eval with aux
    omitted are all equivalent; every execution path degrades to a plain
    coordinate-grid evaluation of shape (Nt, Nx). With no aux axes,
    outer_batch_size chunks the last COORDINATE (t), checked for a
    non-dividing chunk size."""
    x, t, _, _ = grids

    # Dispatch: omitted aux == explicit empty aux, per point and through D.
    assert jnp.allclose(pinn(0.3, 0.7), pinn(0.3, 0.7, ()), rtol=RTOL, atol=ATOL)
    assert jnp.allclose(pinn.D(0, 0)(0.3, 0.7), pinn.D(0, 0)(0.3, 0.7, ()), rtol=RTOL, atol=ATOL)

    full = eval_grid(pinn, (x, t))  # aux defaults to ()
    assert full.shape == (NT, NX)
    assert jnp.allclose(full, eval_grid(pinn, (x, t), ()), rtol=RTOL, atol=ATOL)
    assert jnp.allclose(full[1, 2], pinn(x[2], t[1]), rtol=RTOL, atol=ATOL)

    # Both chunking paths, including a chunk size that doesn't divide NT = 4.
    assert jnp.allclose(eval_grid_flat_aux(pinn, (x, t), batch_size=2), full, rtol=RTOL, atol=ATOL)
    for k in (1, 3):
        assert jnp.allclose(eval_grid(pinn, (x, t), outer_batch_size=k), full, rtol=RTOL, atol=ATOL)


def test_derivative_through_eval_grid(p2inn, grids):
    """model.D preserves the per-point signature, so it batches through
    eval_grid like the model itself. Verify a batched derivative against a
    direct jax.grad at one point and a central finite difference, and confirm
    the layout places the result at the expected index."""
    x, t, a, b = grids
    u_t = eval_grid(p2inn.D(1), (x, t), (a, b))
    assert u_t.shape == (NB, NA, NT, NX)

    i, j, k, l = 2, 3, 1, 0
    params = (a[k], b[l])
    direct = jax.grad(p2inn, argnums=1)(x[i], t[j], params)
    assert jnp.allclose(u_t[l, k, j, i], direct, rtol=RTOL, atol=ATOL)

    eps = 1e-6
    fd = (p2inn(x[i], t[j] + eps, params) - p2inn(x[i], t[j] - eps, params)) / (2 * eps)
    assert jnp.allclose(u_t[l, k, j, i], fd, rtol=1e-5, atol=1e-7)


def test_grad_flows_through_chunked_loss(p2inn, grids):
    """The training use case: a per-point residual (model's own signature,
    parameters as plain scalars) batched by eval_grid_flat_aux inside a loss.
    jax.grad must flow through the internal lax.map to finite, nonzero
    gradients. Also covers the residual-aux-can-equal-model-aux path."""
    x, t, a, b = grids

    def heat_residual(model):
        def r(x_, t_, aux):
            a_, _ = aux
            return model.D(1)(x_, t_, aux) - a_ * model.D(0, 0)(x_, t_, aux)

        return r

    def loss(model):  # model is the differentiated argument
        res = eval_grid_flat_aux(heat_residual(model), (x, t), (a, b), batch_size=2)
        return jnp.mean(res**2)

    grads = eqx.filter_grad(loss)(p2inn)
    leaves = jax.tree_util.tree_leaves(eqx.filter(grads, eqx.is_array))
    assert all(jnp.all(jnp.isfinite(g)) for g in leaves)
    assert any(jnp.any(g != 0) for g in leaves)
