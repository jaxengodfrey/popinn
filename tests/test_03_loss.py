import equinox as eqx
import jax
import jax.numpy as jnp
import pytest

from popinn import FixedWeights, Loss, ResidualTerm, eval_grid, eval_grid_flat_aux
from popinn.loss import _reduce

# Tight for exact/closed-form checks; looser for finite differences.
RTOL, ATOL = 1e-9, 1e-12

# ──────────────────────────────────────────────────────────────
# _reduce - metric dispatch & error paths
# ──────────────────────────────────────────────────────────────

TEST_VALS = jnp.array([2, -9, 5, 6])


def test_metric_str_flags():
    """The built in metrics, `mse` & `mae`, should return mean of squared
    values and mean of absolute values, respectively.
    Unknown metric string will raise a ValueError
    """
    true_mse = 36.5
    true_mae = 5.5

    assert _reduce(TEST_VALS, metric="mse") == true_mse
    assert _reduce(TEST_VALS, metric="mae") == true_mae
    with pytest.raises((ValueError)):
        _reduce(TEST_VALS, metric="abc")


def test_custom_metric_callable():
    """Check that a user-supplied metric is handled properly."""
    test_metric = lambda x: jnp.max(x)
    true_max = 6
    assert _reduce(TEST_VALS, metric=test_metric) == true_max


def test_metric_scalar_output():
    """Output of _reduce must be a scalar, as it is passed to jax.grad through
    the loss. Checks that _reduce produces a value error when custom metric
    does not return scalar.
    """
    assert _reduce(TEST_VALS, metric="mse").shape == ()
    assert _reduce(TEST_VALS, metric="mae").shape == ()
    with pytest.raises((ValueError)):
        _reduce(TEST_VALS, metric=lambda x: x)


# ──────────────────────────────────────────────────────────────
# ResidualTerm - static fields; coordinate, metric, and batch_size plumbing
# ──────────────────────────────────────────────────────────────


def test_residual_term_static_fields(pde_residual_term):
    """ResidualTerm must not contain any array data and all fields should be
    marked as static, therefore it should contain no leaves.
    """
    assert jax.tree_util.tree_leaves(pde_residual_term) == []


def test_residual_term_coordinate_selector(training_data, pde_residual_fn, pde_residual_term, p2inn):
    """`name` supplied to ResidualTerm should select the correct field: e.g., name = 'foo' selects data.foo_coords."""
    check = pde_residual_term(p2inn, training_data)
    correct = _reduce(eval_grid(pde_residual_fn(p2inn), training_data.pde_coords, training_data.aux), "mse")
    incorrect = _reduce(eval_grid(pde_residual_fn(p2inn), training_data.dummy_coords, training_data.aux), "mse")
    assert jnp.allclose(check, correct, rtol=RTOL, atol=ATOL)
    assert not jnp.allclose(check, incorrect, rtol=RTOL, atol=ATOL)


def test_residual_term_invalid_name(training_data, pde_residual_fn, p2inn):
    """`name` supplied to ResidualTerm paired with a dataset with no attribute <name>_coords
    should return an AttributeError.
    """
    res_term = ResidualTerm(name="abc", residual_fn=pde_residual_fn, metric="mse", eval_fn=eval_grid)

    with pytest.raises((AttributeError)):
        res_term(p2inn, training_data)


def test_residual_term_custom_metric(training_data, pde_residual_fn, p2inn):
    """ResidualTerm initialized with a custom metric should pass that metric to _reduce
    when called on a model and dataset.
    """
    custom_metric = lambda x: jnp.max(x)
    res_term = ResidualTerm(name="pde", residual_fn=pde_residual_fn, metric=custom_metric, eval_fn=eval_grid)
    check = res_term(p2inn, training_data)
    correct = _reduce(eval_grid(pde_residual_fn(p2inn), training_data.pde_coords, training_data.aux), custom_metric)
    assert jnp.allclose(check, correct, rtol=RTOL, atol=ATOL)


def test_residual_term_batch_size_routing(training_data, pde_residual_fn, pde_residual_term, p2inn):
    """When batch_size != None in ResidualTerm's initialization call, it chooses the correct
    kwarg ('batch_size' or 'outer_batch_size') for eval_grid or eval_grid_flat_aux. This checks
    that the logic doesn't get flipped, which will throw a TypeError in the assert line below
    """
    res_term_eval_grid = ResidualTerm(name="pde", residual_fn=pde_residual_fn, eval_fn=eval_grid, batch_size=2)

    res_term_eval_grid_flat_aux = ResidualTerm(name="pde", residual_fn=pde_residual_fn, eval_fn=eval_grid_flat_aux, batch_size=2)

    assert jnp.allclose(res_term_eval_grid(p2inn, training_data), res_term_eval_grid_flat_aux(p2inn, training_data), rtol=RTOL, atol=ATOL)
    assert jnp.allclose(res_term_eval_grid(p2inn, training_data), pde_residual_term(p2inn, training_data), rtol=RTOL, atol=ATOL)


# ──────────────────────────────────────────────────────────────
# Weights - static fields, .combine method, scalar output
# ──────────────────────────────────────────────────────────────


def test_fixed_weights_static_fields(fixed_weights):
    """FixedWeights must not contain any array data and all fields should be
    marked as static, therefore it should contain no leaves.
    """
    assert jax.tree_util.tree_leaves(fixed_weights) == []


def test_fixed_weights_combine(fixed_weights):
    """Check that FixedWeights.combine returns the weighted sum on the correct name paths
    and that it returns a KeyError when the residual dict and weights dict have miss-matched
    keys.
    """
    res = {"pde": jnp.array(1.0), "dummy": jnp.array(3.0)}
    check = fixed_weights.combine(res)
    correct = 5.0
    assert jnp.allclose(check, correct, rtol=RTOL, atol=ATOL)
    with pytest.raises((KeyError)):
        res = {"abc": jnp.array(1.0), "dummy": jnp.array(3.0)}
        check = fixed_weights.combine(res)


def test_fixed_weights_scalar_output(fixed_weights):
    """FixedWeights.combine must return scalar output."""
    res = {"pde": jnp.array(1.0), "dummy": jnp.array(3.0)}
    assert fixed_weights.combine(res).shape == ()


# ──────────────────────────────────────────────────────────────
# Loss -
# ──────────────────────────────────────────────────────────────


def test_loss_scalar_dict_output(loss, p2inn, training_data, fixed_weights, pde_residual_term):
    """Loss should return a tuple where the first element is the scalar
    total loss and the second element is a dictionary with each loss component.
    """
    total, component_dict = loss(p2inn, training_data)
    assert total.shape == ()
    assert set(component_dict.keys()) == set(["pde", "dummy"])

    assert jnp.allclose(total, fixed_weights.combine(component_dict), rtol=RTOL, atol=ATOL)
    assert jnp.allclose(component_dict["pde"], pde_residual_term(p2inn, training_data), rtol=RTOL, atol=ATOL)


def test_loss_initialization(pde_residual_term, dummy_residual_term, fixed_weights):
    """Loss makes various assumptions about the names and keys of residual terms
    and weights that ensure the computation is correct.
    """

    # duplicate residual terms
    with pytest.raises((ValueError)):
        Loss([pde_residual_term, pde_residual_term, dummy_residual_term], fixed_weights)

    # weights = None defaults to FixedWeights
    assert Loss([pde_residual_term, dummy_residual_term]).weights == FixedWeights({"pde": 1.0, "dummy": 1.0})

    # the set of weight keys must equal the set residual term names
    with pytest.raises((ValueError)):
        Loss([pde_residual_term, dummy_residual_term], FixedWeights({"abc": 1.0, "dummy": 1.0}))

    # order of residual terms and weight key names doesn't matter:
    loss1 = Loss([pde_residual_term, dummy_residual_term], FixedWeights({"dummy": 3.0, "pde": 5.0}))
    loss2 = Loss([pde_residual_term, dummy_residual_term], FixedWeights({"pde": 5.0, "dummy": 3.0}))

    assert loss1 == loss2


def test_loss_model_differentiable(loss, training_data, p2inn):
    """Make sure the model is differentiable through Loss and that the
    gradients are non-zero and finite.
    """

    (loss_val, loss_dict), grads = eqx.filter_value_and_grad(lambda m: loss(m, training_data), has_aux=True)(p2inn)
    leaves = jax.tree_util.tree_leaves(eqx.filter(grads, eqx.is_array))
    assert leaves  # got gradients at all
    assert all(jnp.all(jnp.isfinite(g)) for g in leaves)  # every leaf finite
    assert any(jnp.any(g != 0) for g in leaves)  # at least one nonzero
