"""test_models.py

Pytest suite for models.py: AbstractModel's per-point machinery (call
dispatch, `D`) and the three concrete models (PINN, P2INN, DeepONet).

These are UNIT tests: each pins down one property of the model layer in
isolation, so a failure names the broken thing directly rather than surfacing
as a cryptic error several vmaps deep inside eval/loss/train. The coverage,
organized by what it pins down:

  Per-point contract -- scalar output, shape () not (1,). Everything
                        downstream (jax.grad, D, eval batching) assumes it.
  Call dispatch      -- the (x, t) / (x, t, ()) / (x, t, aux) conventions
                        resolve correctly; aux must be a tuple, not a list.
  Derivatives        -- D differentiates the right coordinate to the right
                        order, checked EXACTLY against a known closed form and
                        approximately against finite differences on a real net.
  Sub-network wiring -- each encoder/branch is actually connected (perturbing
                        its input moves the output).
  Construction       -- dims wire up for varying coord/param/branch counts;
                        a wrong-length aux tuple fails loudly.
  Characterization   -- the default softplus final activation forces positive
                        output (a documented gotcha), and overriding it works.

float64 is enabled process-wide in conftest.py; the closed-form derivative
checks below rely on it to hold a tight tolerance.
"""

import jax
import jax.numpy as jnp
import jax.random as jr
import pytest

from popinn import AbstractModel, PINN, P2INN, DeepONet
from conftest import NF1_SEN, NF2_SEN

# Tight for exact/closed-form checks; looser for finite differences.
RTOL, ATOL = 1e-9, 1e-12
FD_RTOL, FD_ATOL = 1e-5, 1e-7


# ──────────────────────────────────────────────────────────────
# Per-point contract: scalar output
# ──────────────────────────────────────────────────────────────

def test_scalar_output_contract(pinn, p2inn, deeponet):
    """Each model must return a true scalar, shape () not (1,).

    This is the single most load-bearing invariant in the package: jax.grad
    (and therefore model.D, and therefore every residual) requires a scalar
    output. The trap is eqx.nn.MLP(out_size=1), which yields shape (1,) and
    only fails much later as an opaque grad error deep in the vmap stack.
    Asserting () here is a one-line backstop that catches it at the source.
    """
    assert pinn(0.3, 0.7).shape == ()
    assert p2inn(0.3, 0.7, (1.0, -0.5)).shape == ()
    assert deeponet(0.3, 0.7, (jnp.zeros(NF1_SEN), jnp.zeros(NF2_SEN))).shape == ()


# ──────────────────────────────────────────────────────────────
# Call dispatch: how __call__ separates coords from aux
# ──────────────────────────────────────────────────────────────

def test_pinn_aux_optional_and_ignored(pinn):
    """model(x, t), model(x, t, ()), and model(x, t, <anything>) agree.

    __call__ treats a trailing tuple as aux and everything else as coords, so
    the three spellings should dispatch identically for a PINN -- and since a
    PINN ignores aux entirely, even a non-empty aux must not change the
    output. A failure here means the coord/aux split is miscounting arguments.
    """
    bare = pinn(0.3, 0.7)
    explicit_empty = pinn(0.3, 0.7, ())
    nonempty_aux = pinn(0.3, 0.7, (1.0, -0.5))
    assert jnp.allclose(bare, explicit_empty, rtol=RTOL, atol=ATOL)
    assert jnp.allclose(bare, nonempty_aux, rtol=RTOL, atol=ATOL)


def test_aux_must_be_tuple_not_list(pinn):
    """A list passed where aux belongs is a documented footgun, not silent.

    Dispatch keys on `isinstance(args[-1], tuple)`. A list fails that check,
    so it is swept into coords and dies in jnp.stack (mismatched shapes).
    Pinning this down documents *why* aux must be a tuple: the failure is
    loud and immediate, not a wrong-but-plausible number.
    """
    with pytest.raises((TypeError, ValueError)):
        pinn(0.3, 0.7, [1.0, -0.5])


# ──────────────────────────────────────────────────────────────
# Derivatives via D
# ──────────────────────────────────────────────────────────────

class _Quadratic(AbstractModel):
    """Closed-form model u(x, t) = x^2 * t + t, with no parameters.

    A network's derivatives have no analytic form, so to test that `D`
    differentiates the *right coordinate to the right order* we need a model
    whose derivatives we know exactly. This isolates the differentiation
    machinery (grad chaining, the coord-stacking in __call__) from MLP
    randomness:  u_x = 2xt,  u_xx = 2t,  u_t = x^2 + 1.
    """

    def _eval(self, coords, aux_inputs):
        x, t = coords[0], coords[1]
        return x ** 2 * t + t


def test_D_exact_on_closed_form():
    """D selects the coordinate and differentiation order correctly.

    Checked against the hand-computed derivatives of _Quadratic, so any
    mistake -- differentiating the wrong argnum, wrong chaining order for
    D(0, 0), or a coord-stacking bug -- shows up as an exact mismatch rather
    than tolerance noise.
    """
    m = _Quadratic()
    x, t = 0.4, 1.3
    assert jnp.allclose(m.D(0)(x, t),    2 * x * t,      rtol=RTOL, atol=ATOL)
    assert jnp.allclose(m.D(0, 0)(x, t), 2 * t,          rtol=RTOL, atol=ATOL)
    assert jnp.allclose(m.D(1)(x, t),    x ** 2 + 1.0,   rtol=RTOL, atol=ATOL)


def test_D_matches_finite_difference_on_real_model(p2inn):
    """On a real network, D agrees with a central finite difference.

    """
    x, t, params = 0.3, 0.6, (1.2, -0.4)
    eps = 1e-6
    fd = (p2inn(x, t + eps, params) - p2inn(x, t - eps, params)) / (2 * eps)
    assert jnp.allclose(p2inn.D(1)(x, t, params), fd, rtol=FD_RTOL, atol=FD_ATOL)


def test_D_preserves_call_signature(p2inn):
    """D returns a function with the model's own signature and scalar output.

    D is meant to be a drop-in for the model inside residuals and eval, so the
    derivative must accept the same (coords..., aux) call and return shape ().
    """
    assert p2inn.D(0)(0.3, 0.6, (1.2, -0.4)).shape == ()
    assert p2inn.D(1,1)(0.3, 0.6, (1.2, -0.4)).shape == ()


# ──────────────────────────────────────────────────────────────
# Sub-network wiring: every encoder/branch is connected
# ──────────────────────────────────────────────────────────────

def test_p2inn_depends_on_params_and_coords(p2inn):
    """Output moves when either a parameter or a coordinate changes.

    Guards against a silently severed branch -- e.g. a param encoder whose
    embedding never reaches the manifold. If parameters didn't matter the net
    would be a plain PINN with extra dead weights, and a parametrized-PINN
    test elsewhere could still pass by coincidence; this catches it directly.
    """
    base = p2inn(0.3, 0.6, (1.0, -0.5))
    diff_param = p2inn(0.3, 0.6, (1.7, -0.5))   # changed a only
    diff_coord = p2inn(0.8, 0.6, (1.0, -0.5))   # changed x only
    assert not jnp.allclose(base, diff_param, rtol=RTOL, atol=ATOL)
    assert not jnp.allclose(base, diff_coord, rtol=RTOL, atol=ATOL)


def test_deeponet_depends_on_every_branch(deeponet):
    """Output moves when either branch's function input changes.

    The DeepONet multiplies trunk and branch embeddings elementwise; if a
    branch were dropped from the product, its input would have no effect.
    Perturbing each branch in turn confirms both feed the output.
    """
    f1 = jnp.zeros(NF1_SEN)
    f2 = jnp.zeros(NF2_SEN)
    base = deeponet(0.3, 0.6, (f1, f2))
    perturb_b1 = deeponet(0.3, 0.6, (f1 + 1.0, f2))
    perturb_b2 = deeponet(0.3, 0.6, (f1, f2 + 1.0))
    assert not jnp.allclose(base, perturb_b1, rtol=RTOL, atol=ATOL)
    assert not jnp.allclose(base, perturb_b2, rtol=RTOL, atol=ATOL)


# ──────────────────────────────────────────────────────────────
# Construction: dims wire up; wrong-length aux fails loudly
# ──────────────────────────────────────────────────────────────

def test_pinn_arbitrary_coord_count():
    """num_coords sizes the input layer; a 3-coord PINN takes three scalars."""
    model = PINN(jr.PRNGKey(0), num_coords=3, hidden_dim=8, depth=2)
    assert model(0.1, 0.2, 0.3).shape == ()


def test_p2inn_multiple_params_and_coords():
    """num_params / num_coords size the two encoders independently."""
    model = P2INN(
        jr.PRNGKey(0), num_params=2, num_coords=3,
        param_hidden_dim=8, param_depth=2,
        coord_hidden_dim=8, coord_depth=2,
        manifold_inner_dim=8, manifold_depth=2,
    )
    assert model(0.1, 0.2, 0.3, (1.0, -0.5)).shape == ()


def test_deeponet_wrong_aux_length_raises(deeponet):
    """Calling with fewer functions than branches fails loudly.

    branch_input_dim has length two, so _eval indexes aux_inputs[0] and [1];
    a one-element aux tuple raises rather than silently using one branch.
    """
    with pytest.raises((IndexError, ValueError, TypeError)):
        deeponet(0.3, 0.6, (jnp.zeros(NF1_SEN),))


# ──────────────────────────────────────────────────────────────
# Characterization: default final activation (a documented gotcha)
# ──────────────────────────────────────────────────────────────

def test_default_final_activation_is_positive(pinn, p2inn):
    """PINN/P2INN default to a softplus final activation -> output > 0.

    This is a *characterization* test: it documents intended current behavior
    rather than asserting correctness. The notes flag softplus as a fine
    default but "silently wrong for sign-changing solutions"; pinning the
    positivity here means anyone who changes the default sees a test move,
    prompting them to reconsider the sign-changing case on purpose.
    """
    assert pinn(0.3, 0.7) > 0
    assert p2inn(0.3, 0.7, (1.0, -0.5)) > 0


def test_final_activation_override_is_applied():
    """Overriding final_activation actually changes the output map.

    Swapping softplus for tanh must bound the PINN's output to (-1, 1),
    confirming the final activation is genuinely applied (and giving the
    escape hatch for sign-changing solutions the positivity test warns about).
    """
    model = PINN(jr.PRNGKey(0), num_coords=2, hidden_dim=8, depth=2,
                 final_activation=jnp.tanh)
    out = model(0.3, 0.7)
    assert jnp.abs(out) <= 1.0